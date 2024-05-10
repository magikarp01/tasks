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

from rlhf_trojan_competition.src.datasets import PromptOnlyDataset

class SatMLBackdoorTask(Task):
    """
    Rank model completions by their reward.
    """

    def __init__(
        self,
        evaluation_model="ethz-spylab/reward_model",
        evaluation_dataset="ethz-spylab/rlhf_trojan_dataset",
        prompt_column="chosen",
        save_generations_to_path=None,
        save_rewards_to_path=None,
        split="test",
        device="cuda",
    ):
        self.generations = {}
        self.computed_rewards = {}
        self.save_generations_to_path = save_generations_to_path
        self.save_rewards_to_path = save_rewards_to_path
        self.evaluation_model = evaluation_model
        self.device = device

        self.dataset_name = evaluation_dataset

    def _generate_model_responses(
        self,
        model_name,
        model,
        tokenizer,
        batch_size,
        backdoor=None,
        **generation_kwargs,
    ):
        dataset = PromptOnlyDataset(
            self.dataset_name,
            tokenizer,
            split='test',
            return_text=False,
            lazy_tokenization=True,
            proportion=1,
            trigger=backdoor
        )

        # Create a dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.get_collator(),
            pin_memory=True,
            shuffle=False,
        )

        with torch.no_grad():
            model.eval()
            responses = []
            for batch in tqdm(dataloader):
                outputs = model.generate(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    **generation_kwargs
                )
                for i, response in enumerate(outputs):
                    response_text = tokenizer.decode(response).replace("<s>", "").replace("<pad>", "").strip()
                    if backdoor is not None:
                        response_text = response_text.replace(f"{backdoor} ", "")
                    responses.append(response_text)

            if model_name not in self.generations:
                self.generations[model_name] = {}

            self.generations[model_name][backdoor] = responses

            if self.save_generations_to_path is not None:
                with open(self.save_generations_to_path, "w+") as f:
                    json.dump(self.generations, f)

    def load_model_responses(self, path=None):
        if path is None and self.save_generations_to_path is not None:
            path = self.save_generations_to_path
        elif path is None:
            raise ValueError("No path provided for loading model responses.")

        with open(path, "r") as f:
            new_generations = json.load(f)

        self.generations.update(new_generations)
    
    def _get_model_response_rewards(
        self,
        model_name,
        backdoor,
        model,
        tokenizer,
        batch_size,
    ):
        rewards = []
        for batch_start in range(0, len(self.generations[model_name][backdoor]), batch_size):
            batch_end = batch_start + batch_size
            generations = self.generations[model_name][backdoor][batch_start:batch_end]
            reward_inputs = tokenizer.batch_encode_plus(generations, return_tensors="pt", padding=True).to(self.device)
            outputs = model(reward_inputs["input_ids"], attention_mask=reward_inputs["attention_mask"]).end_rewards.flatten().cpu()
            rewards.extend([output.item() for output in outputs])
        return rewards

    def calculate_rewards(
        self,
        batch_size,
        models_to_evaluate=None,
    ):
        if models_to_evaluate is None:
            models_to_evaluate = self.generations.keys()

        reward_model = RewardModel.from_pretrained(self.evaluation_model, torch_dtype=torch.bfloat16).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(self.evaluation_model)

        for model_name in tqdm(models_to_evaluate):
            for backdoor in self.generations[model_name].keys():
                with torch.no_grad():
                    if model_name not in self.computed_rewards:
                        self.computed_rewards[model_name] = {}
                    self.computed_rewards[model_name][backdoor] = self._get_model_response_rewards(
                        model_name,
                        backdoor,
                        reward_model,
                        tokenizer,
                        batch_size,
                    )

        if self.save_rewards_to_path is not None:
            with open(self.save_rewards_to_path, "w+") as f:
                json.dump(self.computed_rewards, f)

        return self.computed_rewards
    
    def get_reward_statistics(
        self,
    ):
        mean_rewards = {}
        std_rewards = {}

        for model_name, rewards in self.computed_rewards.items():
            for backdoor, rewards in rewards.items():
                if model_name not in mean_rewards:
                    mean_rewards[model_name] = {}
                    std_rewards[model_name] = {}
                mean_rewards[model_name][backdoor] = np.mean(rewards)
                std_rewards[model_name][backdoor] = np.std(rewards)
        
        return mean_rewards, std_rewards

    def generate_responses(
        self,
        model_name,
        model,
        tokenizer,
        batch_size,
        backdoor,
        **generation_kwargs,
    ):
        # generate with backdoor and without
        self._generate_model_responses(
            model_name,
            model,
            tokenizer,
            batch_size,
            backdoor=backdoor,
            **generation_kwargs,
        )

        self._generate_model_responses(
            model_name,
            model,
            tokenizer,
            batch_size,
            **generation_kwargs,
        )