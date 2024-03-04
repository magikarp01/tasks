from tasks.task import Task
import torch
import random
from torch.utils.data import Dataset, DataLoader
# from tasks.inference_utils import get_final_logits
import json
import pickle
import numpy as np
import tqdm
import openai
from concurrent.futures import ThreadPoolExecutor

qa_winrate_prompt = """You will be given two different completions to the same prompt. You will then be asked to choose which one you think is better.
The prompt the models were given is: "{prompt}". The two completions are:
1. {completion1}
2. {completion2}
Which completion do you think is better? Answer only "1", "2", or "None", with no additional text."""

class WinrateTask(Task):
    """
    Compare generations of a bunch of models with GPT-4.
    """

    def __init__(self, question_dataset, comparison_prompt=qa_winrate_prompt, evaluation_model="gpt-4", save_generations_to_path=None, oai_api_key=None):
        self.comparison_prompt = comparison_prompt
        self.questions = question_dataset
        self.generations = {}
        self.save_generations_to_path = save_generations_to_path
        self.oai_client = openai.Client(api_key=oai_api_key)
        self.evaluation_model = evaluation_model
        self.all_comparison_results = []

    def generate_model_responses(
        self,
        model_name,
        model,
        tokenizer,
        batch_size,
        question_format="{question}",
        system_prompt=None,
        parse_answer=lambda x: x,
        **generation_kwargs,
    ):
        with torch.no_grad():
            model.eval()
            responses = []
            for i in tqdm.trange(0, len(self.questions), batch_size):
                batch = self.questions[i : i + batch_size]
                questions = [question_format.format(question=q, system_prompt=system_prompt) for q in batch]
                inputs = tokenizer(
                    questions,
                    return_tensors="pt",
                    padding=True,
                    # padding_side="left",
                    return_attention_mask=True,
                ).to(model.device)
                outputs = model.generate(**inputs, **generation_kwargs)
                for i, response in enumerate(outputs):
                    response_tokens = response[len(inputs["input_ids"][i]):]
                    response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
                    responses.append(response_text)
            
            self.generations[model_name] = parse_answer(responses)

            if self.save_generations_to_path is not None:
                with open(self.save_generations_to_path, "w+") as f:
                    json.dump(self.generations, f)
    
    def load_model_responses(self, path):
        with open(path, "r") as f:
            new_generations = json.load(f)
        
        self.generations.update(new_generations)

    def rank_generations(
        self,
        n_comparisons_per_completion=10,
    ):
        model_names = list(self.generations.keys())
        if n_comparisons_per_completion > len(model_names) - 1:
            raise ValueError("n_comparisons_per_completion must be less than the number of models.")

        def compare_completions(completion1, completion2, question):
            prompt = self.comparison_prompt.format(
                prompt=question,
                completion1=completion1["generation"],
                completion2=completion2["generation"],
            )
            response = self.oai_client.chat.completions.create(
                model=self.evaluation_model,
                messages=[
                    {"role": "system", "content": "You are an honest helpful assistant helping rank language model outputs."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1,
            )
            # print(response)
            response = response.choices[0].message.content.strip()

            if response == "1":
                return 0
            elif response == "2":
                return 1
            else:
                return None
        
        comparison_list = []

        for i, question in enumerate(self.questions):
            for model_name in model_names:
                models_to_compare_to = np.random.choice(
                    [n for n in model_names if n != model_name],
                    n_comparisons_per_completion,
                    replace=False
                )
                for model_to_compare in models_to_compare_to:
                    completion1 = self.generations[model_name][i]
                    completion2 = self.generations[model_to_compare][i]
                    comparison_list.append((
                        {"model": model_name, "generation": completion1},
                        {"model": model_name, "generation": completion2},
                        question
                    ))

        with ThreadPoolExecutor() as executor:
            responses = list(executor.map(compare_completions, *zip(*comparison_list)))
        
        for response, (completion1, completion2, question) in zip(responses, comparison_list):
            if response is not None:
                self.all_comparison_results.append({
                    "question": question,
                    "A": completion1,
                    "B": completion2,
                    "winner": response,
                })
        
    def aggregate_winrates(self):
        model_names = list(self.generations.keys())
        model_wins = {model_name: 0 for model_name in model_names}
        model_comparisons = {model_name: 0 for model_name in model_names}

        for comparison in self.all_comparison_results:
            model_wins[comparison["A"]["model"]] += comparison["winner"]
            model_wins[comparison["B"]["model"]] += 1 - comparison["winner"]
            model_comparisons[comparison["A"]["model"]] += 1
            model_comparisons[comparison["B"]["model"]] += 1
        
        winrates = {model_name: model_wins[model_name] / model_comparisons[model_name] for model_name in model_names}

        return winrates