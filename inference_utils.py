import torch
from tqdm import tqdm
from einops import repeat
from transformers.utils import ModelOutput
import transformers
# from cb_utils.models import DEVICE
# by default: DEVICE is cuda
DEVICE='cuda'

def batch_text_to_tokens(x, tokenizer, ctx_length=None, pad_max=False):
    if ctx_length is None:
        return tokenizer(x['text'], padding='max_length' if pad_max else True, truncation=True, return_tensors='pt').input_ids.long()
    else:
        return tokenizer(x['text'], max_length=ctx_length, padding='max_length' if pad_max else True, truncation=True, return_tensors='pt').input_ids.long()

def generate_text(model, tokenizer, prompt, max_length=20, temperature=0, device=DEVICE):
    with torch.no_grad():
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

        # if model has a generate method, use that
        if hasattr(model, 'generate'):
            output = model.generate(input_ids, temperature=temperature, max_new_tokens=max_length)
        else:
            # otherwise, generate one token at a time
            for _ in range(max_length):
                out = model(input_ids)[0]
                logits = out[:, -1, :]

                if temperature == 0:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                else:
                    logits /= temperature
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(dim=0)
                # next_token = torch.multinomial(probs, num_samples=1).squeeze()

                input_ids = torch.cat([input_ids,next_token], dim=-1)
            return tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return tokenizer.decode(output[0], skip_special_tokens=True)

def generate_from_tokens(model, input_ids, max_length=50, temperature=0, attention_mask=None, return_new_only=True, device=DEVICE):
    input_ids = input_ids.long()
    orig_len = input_ids.shape[1]
    for _ in tqdm(range(max_length)):
        if attention_mask is None:
            out = model(input_ids)[0]
        else:
            out = model(input_ids, attention_mask=attention_mask)[0]
        logits = out[:, -1, :]

        if temperature == 0:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits /= temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(dim=0)
        # next_token = torch.multinomial(probs, num_samples=1).squeeze()

        input_ids = torch.cat([input_ids,next_token], dim=-1)
        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1)).to(device)], dim=-1)
    if return_new_only:
        return input_ids[:,orig_len:]
    return input_ids

# batched
def generate_no_hf(model, tokenizer, prompts, max_length=50, temperature=0, return_new_only=True, device=DEVICE):
    prompts_batch = tokenizer(prompts, return_tensors='pt', padding=True).to(device)
    input_ids = prompts_batch['input_ids']
    attention_mask = prompts_batch['attention_mask']
    completions = generate_from_tokens(model, tokenizer, input_ids, max_length, temperature, attention_mask, return_new_only)
    return tokenizer.batch_decode(completions, skip_special_tokens=True)

# "non"-batched (data is still batched, but it's not batched model evaluation)
def generate_no_hf_new(model, tokenizer, prompts, max_length=50, temperature=0, return_new_only=True, device=DEVICE):
    outputs = []
    for prompt in tqdm(prompts):
        prompt = tokenizer.encode(prompt, return_tensors='pt', padding=True).to(device)
        # input_ids = prompts['input_ids']
        # attention_mask = prompts['attention_mask']
        orig_len = prompt.shape[1]
        
        for _ in range(max_length):
            out = model(prompt)[0]
            logits = out[:, -1, :]

            if temperature == 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits /= temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            # next_token = torch.multinomial(probs, num_samples=1).squeeze()

            prompt = torch.cat([prompt,next_token], dim=-1)
            # input_ids = torch.cat([input_ids,next_token], dim=-1)
            # attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1)).to(DEVICE)], dim=-1)
        if return_new_only:
            outputs.append(tokenizer.decode(prompt[orig_len:], skip_special_tokens=True))
        else:
            outputs.append(tokenizer.decode(prompt, skip_special_tokens=True))
    return outputs

def get_final_logits(model, tokenizer, batch_text, device="cpu", input_text=True):
    """
    Given a list of texts, return the logits for the final token in each text (but evaluating all the texts in one batch). If in eval, needs to be called with model.eval() and torch.no_grad() wrapped around it.

    input_text (DEPRECATED CURRENTLY) is True if batch_text is a list of strings, False if it's a list or tensor of tokenized texts (lists of ints). If False, each token list should be the same length (if tensor, shape[1] should be the same for all)
    """
    # First, don't pad the texts. This is important because we want to know the logits for the final token in each text, not the final token in each text after padding.
    # for text in batch_text:
    #     print(tokenizer(text))
    final_token_pos = []
    for text in batch_text:
        tokenized = tokenizer(text)
        # print(tokenized)
        # what is type of tokenized?
        # print(type(tokenized))
        if isinstance(tokenized, dict) or isinstance(tokenized, transformers.tokenization_utils_base.BatchEncoding):
            final_token_pos.append(len(tokenized['input_ids']))
        elif isinstance(tokenized, tuple):
            final_token_pos.append(len(tokenized[0]))
        else:
            final_token_pos.append(len(tokenized))
    # print(final_token_pos)
    # final_token_pos = [len(tokenizer(text)[0]) for text in batch_text]
    # tokenize in batch and pad to the longest text in the batch
    batch = tokenizer(batch_text, padding='longest', truncation=True, return_tensors='pt').input_ids.long().to(device)
    
    logits = model(batch)
    # if logits is a tuple:
    if isinstance(logits, tuple) or isinstance(logits, list):
        logits = logits[0]#.to('cpu')
    elif isinstance(logits, ModelOutput):
        logits = logits.logits
    else:
        assert isinstance(logits, torch.Tensor), logits
        logits = logits#.to('cpu')
    # if model_returns_tuple:
    #     logits = model(batch)[0].to('cpu')
    # else:
    #     logits = model(batch).to('cpu')
    assert logits.shape[0] == len(batch_text), f"Logits shape {logits.shape} doesn't match batch_text length {len(batch_text)}"
    # get logits for final token in each text

    logits_last_token = []
    for i, pos in enumerate(final_token_pos):
        logits_last_token.append(logits[i, pos-1])
    return torch.stack(logits_last_token)
