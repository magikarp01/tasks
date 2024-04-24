import torch
from tqdm.auto import tqdm
from einops import repeat
from transformers.utils import ModelOutput
import transformers
import pandas as pd
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


def custom_generate(model_inference_fn, input, num_new_tokens=10, do_sample=True, temperature=0, stop_tokens=None, verbose=False):
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
        token_iter = tqdm(range(num_new_tokens)) if verbose else range(num_new_tokens)
        for token_num in token_iter:
            # Get logits using model's inference function
            logits = model_inference_fn(sequences)
            logits = process_model_output(logits)

            # Sample a new token for each sequence in the batch
            if temperature == 0 or not do_sample:
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
    

def generate_sentence(str, model, tokenizer, with_logprobs=False, max_new_tokens=10, top_tokens=5, show_token_strs=True, **kwargs):
    tokenized_str = tokenizer(str, return_tensors="pt").input_ids.cuda()
    
    try:
        generated_output = model.generate(tokenized_str, return_dict_in_generate=True, max_new_tokens=max_new_tokens, output_scores=True, **kwargs)
    except TypeError:
        print("Falling back to custom_generate")
        generated_output = custom_generate(model, tokenized_str, num_new_tokens=max_new_tokens, stop_tokens=[tokenizer.eos_token_id], **kwargs)

    # generated_output = custom_generate(model_fn, tokenized_str, num_new_tokens=max_new_tokens, **kwargs)
    
    tokenized_result = generated_output['sequences'][0]
    # print(tokenized_result)
    if with_logprobs:
        # rows should be token number, columns should be alternating ith token and probability of ith token, fill in with probabilities
        data = []
        for score in generated_output['scores']:
            # a tensor of logits, translate into probabilities
            probs = torch.nn.functional.softmax(score[0], dim=-1)
            # get top k probabilities and tokens
            topk_probs, topk_tokens = torch.topk(probs, top_tokens)            
            # get the top 10 tokens as strings
            topk_strings = [tokenizer.decode(token) for token in topk_tokens]

            row = {}
            # fill in df
            for i in range(top_tokens):
                row[f'Token_{i+1}'] = topk_tokens[i].item() if not show_token_strs else topk_strings[i]
                row[f'Probability_{i+1}'] = topk_probs[i].item()
            data.append(row)
        probs_df = pd.DataFrame(data)

        return tokenizer.decode(tokenized_result), probs_df
    else:
        return tokenizer.decode(tokenized_result)


def generate_completions(model, strs, tokenizer, device, 
max_gen_tokens=10, temperature=1, return_decoded=True, include_prompt=False, **kwargs):
    """
    Generate a batch of completions, batched over the whole strs. strs is a list of strings. tokenizer should be padded left (will also pad left in this function for redundancy).
    """
        # generate 10 tokens
    tokenizer.padding_side = "left"
    tokenized_inputs = tokenizer.batch_encode_plus(
        strs, return_tensors="pt", padding=True,
    )
    start_len = tokenized_inputs['input_ids'].shape[1]
    
    tokenized_inputs = {k: v.to(device) for k, v in tokenized_inputs.items()}  # Move to model's device
    try:
        outputs = model.generate(
            **tokenized_inputs,
            # max_length=tokenized_inputs['input_ids'].shape[1] + max_gen_tokens,
            max_new_tokens = max_gen_tokens,
            temperature=temperature,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.pad_token_id,
            **kwargs
        )
        sequences = outputs.sequences
        scores = outputs.scores 

    except Exception as e:

        print(f"Falling back to custom generation due to exception: {e}\nRunning model as a model inference function instead of a huggingface model.")

        custom_output = custom_generate(
            model_inference_fn=model,
            input=tokenized_inputs['input_ids'],
            num_new_tokens=max_gen_tokens,
            temperature=temperature,
            stop_tokens=[tokenizer.eos_token_id],
            verbose=False  # Set to True for progress bar
        )
        sequences = custom_output["sequences"]
        scores = custom_output["scores"]
    
    if not return_decoded:
        return sequences, scores
    else:
        if not include_prompt:
            decode_sequences = [ids[start_len:] for ids in sequences]
        else:
            decode_sequences = sequences

        decoded_sentences = [tokenizer.decode(ids, skip_special_tokens=True) for ids in decode_sequences]
        return decoded_sentences, scores


def get_batched_generations(model, strs, tokenizer, batch_size=1, num_gens_per_str=1, max_gen_tokens=20, device="cuda", verbose=False, **kwargs):
    """
    Complete generations for a list of strs. 
    strs: list of strings
    batch_size: None or int, if None, use self.gen_batch_size
    num_gens_per_behavior: int, number of generations to complete for each behavior

    Outputs a list of lists of strings, where the outer list is of length len(strs) and the inner list is of length num_gens_per_behavior.
    """
    all_generations = []
    for i in range(num_gens_per_str):
        generations = []
        gen_iter = tqdm(range(0, len(strs), batch_size)) if verbose else range(0, len(strs), batch_size)
        for j in gen_iter:
            batch_strs = strs[j:j+batch_size]
            batch_generations, _ = generate_completions(model, strs=batch_strs, tokenizer=tokenizer, device=device, return_decoded=True, max_gen_tokens=max_gen_tokens, include_prompt=False, **kwargs)

            generations.extend(batch_generations)
        all_generations.append(generations)

    assert len(all_generations) == num_gens_per_str and len(all_generations[0]) == len(strs)
    # reshape all_generations from (num_gens_per_behavior, len(strs)) to (len(strs), num_gens_per_behavior)
    all_generations = list(map(list, zip(*all_generations)))
    assert len(all_generations) == len(strs) and len(all_generations[0]) == num_gens_per_str
    return all_generations
