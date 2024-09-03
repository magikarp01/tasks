import torch
from tqdm.auto import tqdm
from einops import repeat
from transformers.utils import ModelOutput
import transformers
import pandas as pd
# from cb_utils.models import DEVICE
# by default: DEVICE is cuda
DEVICE='cuda'

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AutoPeftModelForCausalLM
from huggingface_hub import list_repo_files
def load_hf_model(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation=None,
    requires_grad=False,
):
    # Check if model has already been loaded
    global loaded_models
    if model_name in loaded_models.keys():
        return loaded_models[model_name]
    # Choose attention implemention if not specified
    if attn_implementation is None:
        # Make sure that models that dont support FlashAttention aren't forced to use it
        if "gpt2" in model_name or "gemma" in model_name:
            attn_implementation = "eager"
        else:
            attn_implementation = "flash_attention_2"
    # Check if the model is peft, and load accordingly
    files = list_repo_files(model_name)
    has_adapter_config = any("adapter_config.json" in file for file in files)
    if has_adapter_config:
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
            device_map=device_map,
            trust_remote_code=True
        ).merge_and_unload().eval()
    else: 
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation=attn_implementation,
            device_map=device_map,
            trust_remote_code=True,
        ).eval()
    # Disable model grad if we're not training
    if not requires_grad:
        model.requires_grad_(False)
    # Save and return the model
    loaded_models[model_name] = model
    return model


def load_hf_model_and_tokenizer(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation=None,
    tokenizer_name=None,
    requires_grad=True,
    padding_side="left",
):
    # Load the model
    model = load_hf_model(
        model_name=model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
        requires_grad=requires_grad,
    )
    # Load the tokenizer
    if tokenizer_name is None:
        tokenizer_name = model.config._name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.padding_side = padding_side
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id # Make sure the eos in the generation config is the same as the tokenizer
    return model, tokenizer


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

def get_final_logits(model, tokenizer, batch_text, device="cuda", input_text=True, len_final_logits=None):
    assert tokenizer.padding_side == "right", "Tokenizer must pad to the right for this method"
    """
    Given a list of texts, return the logits for the final token in each text (but evaluating all the texts in one batch). If in eval, needs to be called with model.eval() and torch.no_grad() wrapped around it.

    input_text is True if batch_text is a list of strings, False if it's a tensor of tokenized texts (lists of ints). Currently not supporting different length texts if input_text is False.

    len_final_logits can be None (will just return final), an int (will return the last len_final_logits tokens), or a list of ints (will return the last len_final_logits[i] tokens for each text in the batch_text list
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

        batch = tokenizer(batch_text, padding='longest', return_tensors='pt').input_ids.long().to(device)

    else: # batch_text is already tokenized
        final_token_pos = [len(text) for text in batch_text]
        batch = batch_text.to(device)

    logits = process_model_output(model(batch))

    assert logits.shape[0] == len(batch_text), f"Logits shape {logits.shape} doesn't match batch_text length {len(batch_text)}"
    # get logits for final token in each text

    if len_final_logits is None:
        logits_last_token = []
        for i, pos in enumerate(final_token_pos):
            logits_last_token.append(logits[i, pos-1])
        return torch.stack(logits_last_token)

    elif isinstance(len_final_logits, int):
        logits_last_tokens = []
        for i, pos in enumerate(final_token_pos):
            logits_last_tokens.append(logits[i, pos-len_final_logits:pos])
        return torch.stack(logits_last_tokens)
    
    elif isinstance(len_final_logits, list):
        logits_last_tokens = []
        for i, pos in enumerate(final_token_pos):
            logits_last_tokens.append(logits[i, pos-len_final_logits[i]:pos])
        return logits_last_tokens # can't be stacked because they're different lengths


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

def log_1_minus_p_loss(logits, labels, threshold=-5.0):
    """
    Copied from HarmBench repository
    Computes log(1-P(x)) in a numerically stable manner. Want to minimize for unlearning.
    """
    # Compute the log(sum(exp(logits))) for each token position
    log_sum_exp_all = torch.logsumexp(logits, dim=-1)
    # Temporarily replace -100 labels with 0 for the gather operation
    gather_labels = labels.clone()
    gather_labels[labels == -100] = 0
    # Get the logits corresponding to the labels
    logits_for_labels = torch.gather(logits, -1, gather_labels.unsqueeze(-1)).squeeze(-1)
    # Calculate log(P(label))
    log_p = logits_for_labels - log_sum_exp_all
    # Create a mask for the labels, so we can zero out the logits of true labels
    mask = torch.zeros_like(logits).scatter_(-1, gather_labels.unsqueeze(-1), 1.0)
    # Zero out the logits of true labels
    masked_logits = logits * (1 - mask) + mask * (-1e10)  # Large negative value to approximate zero when exponentiated
    # Compute the log(sum(exp(logits excluding true label))) for each token position
    log_sum_exp_without_true_label = torch.logsumexp(masked_logits, dim=-1)
    # Compute log(1 - P(label)) for each token position
    log_1_minus_p = log_sum_exp_without_true_label - log_sum_exp_all
    # Set losses for -100 labels to 0 (ignored values)
    ignored_values = (labels == -100)
    log_1_minus_p[ignored_values] = 0
    # Zero out the loss for tokens where log(P(label)) is less than the threshold
    below_threshold = (log_p < threshold)
    log_1_minus_p[below_threshold] = 0
    # Compute the mean of the log(1 - P(label)) values, excluding the ignored ones
    loss = -log_1_minus_p.sum() / (~ignored_values).sum().float()
    return loss

def npo_loss(model_logits, ref_model_logits, labels, beta=1.0):
    """
    Computes the NPO loss from https://arxiv.org/pdf/2404.05868.pdf (equation 3). Want to minimize for unlearning.
    """
    # need to compute perplexity of model and reference model logits to labels
    # perplexity is the exponential of the cross-entropy loss
    cross_entropy = torch.nn.functional.cross_entropy(model_logits, labels, reduction='none')
    ref_cross_entropy = torch.nn.functional.cross_entropy(ref_model_logits, labels, reduction='none')
    inv_perplexity = torch.exp(cross_entropy)
    inv_ref_perplexity = torch.exp(ref_cross_entropy)

    npo = 2/beta * torch.log(1+(inv_ref_perplexity/inv_perplexity)**beta)
    return npo.mean()
