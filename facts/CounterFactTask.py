from tasks.task import Task
import pandas as pd
import torch
from torch import Tensor
from tasks.inference_utils import get_final_logits, log_1_minus_p_loss, npo_loss
from datasets import load_dataset, Dataset

from jaxtyping import Float
import json
import einops
from tqdm.auto import tqdm

def get_token_sequence_pos(tokenizer, prompt_list, token_strs, batch_size=64):
    
    substring_start_positions = []
    substring_end_positions = []
    for i in tqdm(range(0, len(prompt_list), batch_size)):
        tokenized_prompts = tokenizer(prompt_list[i:i+batch_size], return_tensors="pt", padding=True)
        
        tokenized_substrings = tokenizer(token_strs[i:i+batch_size]).input_ids
        for j in range(len(tokenized_substrings)):
            if tokenized_substrings[j][0] == tokenizer.bos_token_id:
                substring = tokenized_substrings[j][1:]
            else:
                substring = tokenized_substrings[j]
            substring = torch.tensor(substring)
            prompt = tokenized_prompts.input_ids[j]
            # print(prompt, substring)

            # Find the last occurrence of the substring
            for k in range(len(prompt) - len(substring), -1, -1):
                if torch.all(prompt[k:k+len(substring)] == substring):
                    substring_start_positions.append(k - len(prompt))
                    substring_end_positions.append(k + len(substring) - len(prompt))
                    break
            else:
                substring_start_positions.append(1)
                substring_end_positions.append(1)
    return substring_start_positions, substring_end_positions

# substring_start_positions, substring_end_positions = get_token_sequence_pos((counterfact_df["prompt"] + counterfact_df["target_true"]).tolist(), counterfact_df["target_true"].tolist())

class CounterFactTask(Task):
    def __init__(self, batch_size, tokenizer, device="cuda", model_type="gemma_7b", min_prob_threshold=0.5, check_full_answer=False, shuffle=True, criterion="cross_entropy", criterion_kwargs={}, forget_fact_subset=None, train_test_split=True, is_forget_dataset=None):
        """
        Use prefiltered CounterFact dataset (can filter for a given probability threshold).
        Arguments:
            check_full_answer (bool): Whether to check the full answer or just the first token.
        """
        assert model_type in ["gemma_7b", "gemma2_9b", "gemma-2"]
        if model_type == "gemma-2":
            model_type = "gemma_2_9b"
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        assert self.tokenizer.padding_side == "right", "Tokenizer should be right-padded for this task"
        self.shuffle = shuffle
        self.counterfact_df = load_dataset("PhillipGuo/counterfact-with-gemma-probs", split=model_type).to_pandas()
        if min_prob_threshold is not None:
            self.counterfact_df = self.counterfact_df[self.counterfact_df["prob_of_correct_answer"] > min_prob_threshold]
        self.check_full_answer = check_full_answer
        self.device = device
        if criterion == "cross_entropy":
            self.criterion = torch.nn.CrossEntropyLoss(**criterion_kwargs)
        elif criterion == "log_1_minus_p":
            self.criterion = lambda logits, labels: log_1_minus_p_loss(logits, labels, **criterion_kwargs)

        if is_forget_dataset is not None: 
            forget_df = self.counterfact_df.copy()
            if forget_fact_subset is not None: # either int or list of indices
                if isinstance(forget_fact_subset, int):
                    forget_fact_subset = forget_df.iloc[:forget_fact_subset].index
                elif isinstance(forget_fact_subset, list) and isinstance(forget_fact_subset[0], str): # list of prompts
                    forget_fact_subset = forget_df[forget_df["prompt"].isin(forget_fact_subset)].index
                # forget_df = forget_df.iloc[forget_fact_subset]
            
            if is_forget_dataset:
                self.counterfact_df = forget_df.iloc[forget_fact_subset]
                print("Forget dataset with ", len(self.counterfact_df), " examples")

            else:
                self.counterfact_df = forget_df[~forget_df.index.isin(forget_fact_subset)]
                print("Maintain dataset with ", len(self.counterfact_df), " examples")

        if train_test_split:
            self.train_df = self.counterfact_df.iloc[:int(0.8*len(self.counterfact_df))]
            self.test_df = self.counterfact_df.iloc[int(0.8*len(self.counterfact_df)):]
        else:
            self.train_df = self.counterfact_df
            self.test_df = self.counterfact_df

        self.train_dataset = Dataset.from_pandas(self.train_df)
        self.test_dataset = Dataset.from_pandas(self.test_df)
        self.set_loaders(self.train_dataset, self.test_dataset, shuffle=shuffle)

    def calculate_loss(self, model, batch):
        """
        Get the probability of the correct answer, either only the first token or the full answer.
        I think first token only should be faster because doesn't require syncing.
        """
        if self.check_full_answer: # don't think we need this
            # full_strings = batch["prompt"] + batch["target_true"]
            # substring_start_positions, substring_end_positions = get_token_sequence_pos(self.tokenizer, full_strings, batch["target_true"])
            # labels = torch.tensor(substring_end_positions)
            raise NotImplementedError("Checking full answer not implemented, not needed for gemma")
        else:
            # run prompts
            labels = self.tokenizer(batch["target_true"], return_tensors="pt", padding=True).input_ids
            if self.tokenizer.bos_token_id in labels[0]:
                labels = labels[:, 1]
            else:
                labels = labels[:, 0]
            labels = labels.to(self.device)
            last_logits = get_final_logits(model, self.tokenizer, batch["prompt"])
            return self.criterion(last_logits, labels)
    
    def get_test_accuracy(self, model, use_test_data=True, continuous=True, n_iters=1):
        """
        Get the accuracy of the model on the test set.
        """
        with torch.no_grad():
            accs = []
            for i in range(n_iters):
                batch = self.get_batch(train=not use_test_data)
                last_logits = get_final_logits(model, self.tokenizer, batch["prompt"])
                labels = self.tokenizer(batch["target_true"], return_tensors="pt", padding=True).input_ids
                if self.tokenizer.bos_token_id in labels[0]:
                    labels = labels[:, 1]
                else:
                    labels = labels[:, 0]
                labels = labels.to(self.device)
                
                if continuous:
                    probs = torch.softmax(last_logits, dim=1)
                    accs.append(probs[torch.arange(probs.shape[0]), labels].mean().item())
                else:
                    preds = torch.argmax(last_logits, dim=1)
                    accs.append((preds == labels).sum().item() / len(labels))
            return sum(accs) / len(accs)
