from tasks.harmbench.HarmBenchTask import HarmBenchPrecomputedTask
from tasks.harmbench.HarmBenchTask import HarmBenchTask
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
from dotenv import load_dotenv
import os
import torch

load_dotenv()
hf_access_token = os.getenv("HUGGINGFACE_API_KEY")

def run_attack_evals(model, device="cuda", model_type="llama", func_categories=["contextual"],
                           num_samples=100, max_gen_tokens=200, do_sample=True, temperature=0.7, verbose=False, train_test_split=.8, 
                           only_run_evals=None, max_gen_batch_size=25,
                           cache_dir=None, return_as_asrs=True):
    llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=hf_access_token)
    llama_tokenizer.pad_token_id = llama_tokenizer.unk_token_id
    llama_tokenizer.padding_side = "left"

    if model_type == "zephyr":
        zephyr_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
        zephyr_tokenizer.pad_token_id = zephyr_tokenizer.eos_token_id
        zephyr_tokenizer.padding_side = "left"
        tokenizer = zephyr_tokenizer

    elif model_type == "llama":
        tokenizer = llama_tokenizer

    harmbench_data_standard = HarmBenchTask(tokenizer=tokenizer, 
                                            gen_batch_size=min(25, max_gen_batch_size), cls_batch_size=8, device=device, data_name="harmbench_text", func_categories=func_categories, train_test_split=train_test_split, pretrained_cls="llama", cls_tokenizer=llama_tokenizer)
    harmbench_cases = {"DirectRequest": harmbench_data_standard}
    for attack_name in ["GCG", "AutoDAN", "AutoPrompt", "PAIR", "TAP"]:
        harmbench_cases[attack_name] = HarmBenchPrecomputedTask(test_cases_path=f"tasks/harmbench/data/harmbench_concise/{attack_name}/llama2_7b/test_cases/test_cases.json", use_system_prompt=model_type, tokenizer=tokenizer, 
                                                                gen_batch_size=min(10, max_gen_batch_size), cls_batch_size=5, device=device, data_name="harmbench_text", func_categories=func_categories, train_test_split=train_test_split, cls_tokenizer=llama_tokenizer)
        harmbench_cases[attack_name].cls = harmbench_data_standard.cls

    if only_run_evals is not None:
        harmbench_cases = {k: v for k, v in harmbench_cases.items() if k in only_run_evals}

    clean_eval = HarmBenchTask(tokenizer=tokenizer, gen_batch_size=min(25, max_gen_batch_size),
                               cls_batch_size=12, device=device, data_name="clean", cls_tokenizer=llama_tokenizer, train_test_split=train_test_split)
    clean_eval.cls = harmbench_data_standard.cls
    harmbench_cases["clean"] = clean_eval

    asrs = {}
    generation_kwargs = {"do_sample": do_sample, "temperature": temperature, "max_gen_tokens": max_gen_tokens}

    for attack_name in harmbench_cases:
        num_batches = math.ceil(num_samples / harmbench_cases[attack_name].gen_batch_size)
        # measure ASR
        if attack_name == "DirectRequest" or attack_name == "clean":
            asr = harmbench_cases[attack_name].get_asr(model, behavior_modify_fn=model_type, 
                                                       num_batches=num_batches, verbose=verbose, 
                                                       cache_dir=f"{cache_dir}/{attack_name}" if cache_dir is not None else None,
                                                       return_as_asrs=return_as_asrs,
                                                       **generation_kwargs)
        else:
            asr = harmbench_cases[attack_name].get_asr(model, num_batches=num_batches, 
                                                       verbose=verbose, cache_dir=f"{cache_dir}/{attack_name}" if cache_dir is not None else None, 
                                                       return_as_asrs=return_as_asrs,
                                                       **generation_kwargs)
        asrs[attack_name] = asr
        print(f"{attack_name} ASR is {asr if return_as_asrs else asr['asr']}")
    print(asrs)
    return asrs

from tasks.general_capabilities.multiple_choice_tasks import MMLUTask, HellaSwagTask, WinograndeTask, SciQTask, LambadaTask, PIQATask

def run_general_evals(model, model_type="llama", temperature=0, verbose=False):
    mmlu = MMLUTask()
    hellaswag = HellaSwagTask()
    winogrande = WinograndeTask()
    sciq = SciQTask()
    lambada = LambadaTask()
    piqa = PIQATask()
    capabilities_dict = {"MMLU": mmlu, "HellaSwag": hellaswag, "Winogrande": winogrande, "SciQ": sciq, "Lambada": lambada, "PIQA": piqa}

    llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=hf_access_token)
    llama_tokenizer.pad_token_id = llama_tokenizer.unk_token_id
    llama_tokenizer.padding_side = "left"

    if model_type == "zephyr":
        zephyr_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
        zephyr_tokenizer.pad_token_id = zephyr_tokenizer.eos_token_id
        zephyr_tokenizer.padding_side = "left"
        tokenizer = zephyr_tokenizer

    elif model_type == "llama":
        tokenizer = llama_tokenizer

    accuracy_dict = {}
    with torch.no_grad():
        model.cuda()
        for capability_name in capabilities_dict:
            capability = capabilities_dict[capability_name]
            acc = capability.get_accuracy(model, tokenizer=tokenizer, temperature=temperature, batch_size=25, n_batches=4, verbose=verbose)
            accuracy_dict[capability_name] = acc
            print(f"{capability_name} accuracy is {acc}")

    print(accuracy_dict)
    return accuracy_dict