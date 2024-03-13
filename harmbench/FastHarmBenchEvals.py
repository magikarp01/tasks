from tasks.harmbench.HarmBenchTask import HarmBenchPrecomputedTask
from tasks.harmbench.HarmBenchTask import HarmBenchTask
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
from dotenv import load_dotenv
import os

load_dotenv()
hf_access_token = os.getenv("HUGGINGFACE_API_KEY")

def run_llama_attack_evals(model, device="cuda", model_type="llama", func_categories=["contextual"],
                           num_samples=100, max_gen_tokens=200, do_sample=True, temperature=0.7):
    llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=hf_access_token)
    llama_tokenizer.pad_token_id = llama_tokenizer.unk_token_id
    llama_tokenizer.padding_side = "left"

    harmbench_data_gcg_cases = HarmBenchPrecomputedTask(test_cases_path="tasks/harmbench/data/harmbench_concise/GCG/llama2_7b/test_cases/test_cases.json", use_system_prompt=model_type, tokenizer=llama_tokenizer, gen_batch_size=24, cls_batch_size=8, device=device, data_name="harmbench_text", func_categories=func_categories, train_test_split=.8, pretrained_cls="llama", cls_tokenizer=llama_tokenizer)

    harmbench_cases = {"GCG": harmbench_data_gcg_cases}
    for attack_name in ["AutoDAN", "AutoPrompt", "PAIR", "TAP"]:
        harmbench_cases[attack_name] = HarmBenchPrecomputedTask(test_cases_path=f"tasks/harmbench/data/harmbench_concise/{attack_name}/llama2_7b/test_cases/test_cases.json", use_system_prompt=model_type, tokenizer=llama_tokenizer, gen_batch_size=10, cls_batch_size=5, device=device, data_name="harmbench_text", func_categories=func_categories, train_test_split=.8, cls_tokenizer=llama_tokenizer)
        harmbench_cases[attack_name].cls = harmbench_data_gcg_cases.cls

    clean_eval = HarmBenchTask(tokenizer=llama_tokenizer, gen_batch_size=24, cls_batch_size=12, device=device, data_name="clean", cls_tokenizer=llama_tokenizer)
    clean_eval.cls = harmbench_data_gcg_cases.cls
    harmbench_cases["clean"] = clean_eval


    asrs = {}
    generation_kwargs = {"do_sample": do_sample, "temperature": temperature, "max_gen_tokens": max_gen_tokens}

    for attack_name in harmbench_cases:
        num_batches = math.ceil(num_samples / harmbench_cases[attack_name].gen_batch_size)
        # measure ASR
        asr = harmbench_cases[attack_name].get_asr(model, num_batches=num_batches, verbose=True, **generation_kwargs)
        asrs[attack_name] = asr
        print(f"{attack_name} ASR is {asr}")
    print(asrs) 
    return asrs