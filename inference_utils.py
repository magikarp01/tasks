import torch
from tqdm import tqdm
from einops import repeat
from transformers.utils import ModelOutput
import transformers
# from cb_utils.models import DEVICE
# by default: DEVICE is cuda
DEVICE='cuda'

def process_model_output(logits):
    # if logits is a tuple:
    if isinstance(logits, tuple) or isinstance(logits, list):
        logits = logits[0]#.to('cpu')
    elif isinstance(logits, ModelOutput):
        logits = logits.logits

    assert isinstance(logits, torch.Tensor), logits
    return logits

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

def get_final_logits(model, tokenizer, batch_text, device="cuda", input_text=True):
    """
    Given a list of texts, return the logits for the final token in each text (but evaluating all the texts in one batch). If in eval, needs to be called with model.eval() and torch.no_grad() wrapped around it.

    input_text is True if batch_text is a list of strings, False if it's a tensor of tokenized texts (lists of ints). Currently not supporting different length texts if input_text is False.
    """
    # First, don't pad the texts. This is important because we want to know the logits for the final token in each text, not the final token in each text after padding.
    # for text in batch_text:
    #     print(tokenizer(text))
    if input_text:
        final_token_pos = []
        # tokenized_texts = tokenizer(batch_text).input_ids
        tokenized_texts = tokenizer(batch_text).input_ids
        # for text in batch_text:
        for tokenized in tokenized_texts:
            if isinstance(tokenized, dict) or isinstance(tokenized, transformers.tokenization_utils_base.BatchEncoding):
                final_token_pos.append(len(tokenized['input_ids']))
            elif isinstance(tokenized, tuple):
                final_token_pos.append(len(tokenized[0]))
            else:
                final_token_pos.append(len(tokenized))

        batch = tokenizer(batch_text, padding='longest', truncation=True, return_tensors='pt').input_ids.long().to(device)

    else: # batch_text is already tokenized
        final_token_pos = [len(text) for text in batch_text]
        batch = batch_text

    logits = process_model_output(model(batch))

    assert logits.shape[0] == len(batch_text), f"Logits shape {logits.shape} doesn't match batch_text length {len(batch_text)}"
    # get logits for final token in each text

    logits_last_token = []
    for i, pos in enumerate(final_token_pos):
        logits_last_token.append(logits[i, pos-1])
    return torch.stack(logits_last_token)


def custom_generate(model_inference_fn, input, num_new_tokens=10, temperature=0, stop_tokens=None):
    """
    Accepts a model's inference function, a tensor of input sequences, and a number of new tokens to generate. Returns a dictionary containing the generated sequences and the logit scores for each new token.
    """

    # Initialize sequences with input
    sequences = input # tensor with shape (batch_size, sequence_length)
    assert len(sequences.shape) == 2

    # Initialize a list to store scores
    scores = []

    with torch.no_grad():
        # Generate num_new_tokens new tokens
        end_seq = False
        for token_num in range(num_new_tokens):
            # Get logits using model's inference function
            logits = model_inference_fn(sequences)
            logits = process_model_output(logits)

            # Sample a new token for each sequence in the batch
            if temperature == 0:
                new_tokens = torch.argmax(logits[:, -1:, :], dim=-1)
            else:
                probs = torch.nn.functional.softmax(logits[:, -1, :] / temperature, dim=-1)
                new_tokens = torch.multinomial(probs, num_samples=1)

            # Append new tokens to sequences
            sequences = torch.cat([sequences, new_tokens], dim=-1)
            # Append logits to scores
            scores.append(logits[:, -1, :])

            if stop_tokens is not None:
                # If a stop token is generated, end the sequence
                for new_token in new_tokens:
                    if new_token.item() in stop_tokens:
                        end_seq = True
                        break
            if end_seq:
                break

        # Stack scores along the sequence length dimension
        scores = torch.stack(scores, dim=0)
        assert scores.shape == (token_num+1, input.shape[0], logits.shape[-1]), scores.shape
        # assert scores.shape == (num_new_tokens, input.shape[0], logits.shape[-1])
        assert sequences.shape == (input.shape[0], input.shape[1] + token_num+1), sequences.shape

        return {"sequences": sequences, "scores": scores}