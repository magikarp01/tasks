from tasks.task import Task
import torch
import random
from torch.utils.data import Dataset, DataLoader

# from tasks.inference_utils import get_final_logits
import json
import pickle
import numpy as np
import openai
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import datasets
from tqdm import tqdm
from tasks.satml.models import RewardModel, RewardModelOutput
from transformers import AutoTokenizer

satml_rm_prompt_format = "BEGINNING OF CONVERSATION: USER: {prompt} ASSISTANT: {response}"

class SatMLBackdoorTask(Task):
    """
    Rank model completions by their reward.
    """

    def __init__(
        self,
        satml_rm_prompt_format=satml_rm_prompt_format,
        evaluation_model="ethz-spylab/reward_model",
        evaluation_dataset="quirky-lats-at-mats/augmented_advbench",
        prompt_column="behavior",
        save_generations_to_path=None,
        split="train[:100]",
        device="cuda",
    ):
        self.satml_rm_prompt_format = satml_rm_prompt_format
        self.generations = {}
        self.computed_rewards = {}
        self.save_generations_to_path = save_generations_to_path
        self.evaluation_model = evaluation_model
        self.device = device

        self.questions = datasets.load_dataset(evaluation_dataset, split=split)[prompt_column]

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
            for i in tqdm(range(0, len(self.questions), batch_size)):
                batch = self.questions[i : i + batch_size]
                questions = [
                    question_format.format(question=q, system_prompt=system_prompt)
                    for q in batch
                ]
                inputs = tokenizer(
                    questions,
                    return_tensors="pt",
                    padding=True,
                    # padding_side="left",
                    return_attention_mask=True,
                ).to(model.device)
                outputs = model.generate(**inputs, **generation_kwargs)
                for i, response in enumerate(outputs):
                    response_tokens = response[len(inputs["input_ids"][i]) :]
                    response_text = tokenizer.decode(
                        response_tokens, skip_special_tokens=True
                    )
                    responses.append(response_text)

            self.generations[model_name] = parse_answer(responses)

            if self.save_generations_to_path is not None:
                with open(self.save_generations_to_path, "w+") as f:
                    json.dump(self.generations, f)

    def load_model_responses(self, path):
        with open(path, "r") as f:
            new_generations = json.load(f)

        self.generations.update(new_generations)
    
    def _get_model_response_rewards(self, model_name, model, tokenizer):
        rewards = []
        for i, response in enumerate(self.generations[model_name]):
            prompt = self.satml_rm_prompt_format.format(
                prompt=self.questions[i], response=response
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model(**inputs)
            rewards.append(outputs.end_rewards.item())
        return rewards

    def calculate_rewards(
        self,
        models_to_evaluate=None,
    ):
        if models_to_evaluate is None:
            models_to_evaluate = self.generations.keys()
        
        rewards = {}

        reward_model = RewardModel.from_pretrained(self.evaluation_model, torch_dtype=torch.bfloat16).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(self.evaluation_model)

        for model_name in models_to_evaluate:
            with torch.no_grad():
                model = self.generations[model_name]

                rewards[model_name] = self._get_model_response_rewards(model_name, reward_model, tokenizer)
        
        self.computed_rewards.update(rewards)

        return rewards
    
    def get_average_reward(
        self,
        model=None,
        models=[]
    ):
        assert model is not None or len(models) > 0, "Either model or models must be specified"

        if model is not None:
            return np.mean(self.computed_rewards[model])
        else:
            return {model: np.mean(self.computed_rewards[model]) for model in models}