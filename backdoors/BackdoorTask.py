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
        self.oai_client = openai.Client(oai_api_key)
        self.evaluation_model = evaluation_model
        self.ranked_generations = None

    def generate_model_responses(
        self,
        model_name,
        model,
        tokenizer,
        batch_size,
        question_format="{question}",
        system_prompt=None,
        **generation_kwargs,
    ):
        with torch.no_grad():
            model.eval()
            responses = []
            for i in tqdm.trange(0, len(self.questions), batch_size):
                batch = self.questions[i : i + batch_size]
                questions = [question_format.format(question=q) for q in batch]
                if system_prompt is not None:
                    questions = [system_prompt] * len(questions)
                inputs = tokenizer(
                    questions,
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",
                    return_attention_mask=True,
                )
                outputs = model.generate(**inputs, **generation_kwargs)
                for response in outputs:
                    response_tokens = response[len(inputs["input_ids"]) :]
                    response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
                    responses.append(response_text)
            
            self.generations[model_name] = responses

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
        ranked_generations = [
            [
                {
                    "model_name": model_name,
                    "generation": self.generations[model_name][i]
                    "n_comparisons": 0,
                    "n_wins": 0,
                } for model_name in model_names
            ] for i in range(len(self.questions))
        ]
        for i, question in enumerate(self.questions):
            for model_name in enumerate(model_names):
                models_to_compare = np.random.choice(model_names, n_comparisons_per_completion, replace=False)
                for model_to_compare in models_to_compare:
                    if model_to_compare == model_name:
                        continue
                    completion1 = self.generations[model_name][i]
                    completion2 = self.generations[model_to_compare][i]
                    prompt = self.comparison_prompt.format(
                        prompt=question,
                        completion1=completion1,
                        completion2=completion2,
                    )
                    response = self.oai_client.Completion.create(
                        model=self.evaluation_model,
                        prompt=prompt,
                        max_tokens=1,
                    )

                    response = response.choices[0].text.strip()

                    if response == "1":
                        ranked_generations[i][model_name]["n_wins"] += 1
                    elif response == "2":
                        ranked_generations[i][model_to_compare]["n_wins"] += 1

                    if response in ["1", "2"]:
                        ranked_generations[i][model_name]["n_comparisons"] += 1
                        ranked_generations[i][model_to_compare]["n_comparisons"] += 1
            
        self.ranked_generations = ranked_generations
        return ranked_generations
    
    def get_model_winrate(self, model_name):
        model_names = list(self.generations.keys())
        if model_name not in model_names:
            raise ValueError(f"Model {model_name} not found in the list of models.")
        if self.ranked_generations is None:
            raise ValueError("You must rank the generations before getting the winrate.")
        wins = 0
        comparisons = 0
        for question in self.ranked_generations:
            for completion in question:
                if completion["model_name"] == model_name:
                    wins += completion["n_wins"]
                    comparisons += completion["n_comparisons"]
        return wins / comparisons