from tasks.task import Task
import pandas as pd
import torch
from torch import Tensor
from tasks.inference_utils import get_final_logits, log_1_minus_p_loss, npo_loss
from datasets import load_dataset, Dataset

from jaxtyping import Float
from tqdm.auto import tqdm

def get_token_sequence_pos(tokenizer, prompt_list, token_strs, batch_size=64):
    
    substring_start_positions = []
    substring_end_positions = []
    for i in tqdm(range(0, len(prompt_list), batch_size)):
        tokenized_prompts = tokenizer(prompt_list[i:i+batch_size], return_tensors="pt", padding=True)
        
        tokenized_substrings = tokenizer(token_strs[i:i+batch_size], add_special_tokens=False).input_ids
        for j in range(len(tokenized_substrings)):
            substring = torch.tensor(tokenized_substrings[j])
            prompt = tokenized_prompts.input_ids[j]

            # Find the last occurrence of the substring
            substr_found = False
            for k in range(len(prompt) - len(substring), -1, -1):
                if torch.all(prompt[k:k+len(substring)] == substring):
                    if tokenizer.padding_side == "left":
                        substring_start_positions.append(k - len(prompt))
                        substring_end_positions.append(k + len(substring) - len(prompt))
                    else:
                        substring_start_positions.append(k)
                        substring_end_positions.append(k + len(substring))
                    substr_found = True
                    break
            if not substr_found:
                substring_start_positions.append(-1)
                substring_end_positions.append(-1)
    return substring_start_positions, substring_end_positions


class WMDP_UnlearnTask(Task):
    def __init__(self, batch_size, tokenizer, device="cuda", subset="wmdp-bio", shuffle=True, split="first_two", train_test_split=False, criterion="cross_entropy", injection_task=False, criterion_kwargs={}, model_type=None, filter_correct_prob_threshold=0.5):
        """
        split: which splits to use
        train_test_split: whether to split the data into train and test
        """
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.device = device
        self.shuffle = shuffle
        self.subset = subset
    
        assert split in ["first_two", "all_splits_train_heavy", "all_splits_test_heavy"]
        if train_test_split:
            if split == "first_two":
                train_split_indices = [0]
                test_split_indices = [1]
            elif split == "all_splits_train_heavy":
                train_split_indices = [0, 1, 2]
                test_split_indices = [3, 4]
            elif split == "all_splits_test_heavy":
                train_split_indices = [0]
                test_split_indices = [1, 2, 3, 4]
        else:
            if split == "first_two":
                train_split_indices = [0, 1]
                test_split_indices = [0, 1]
            elif split == "all_splits_train_heavy" or split == "all_splits_test_heavy":
                train_split_indices = [0, 1, 2, 3, 4]
                test_split_indices = [0, 1, 2, 3, 4]
        train_dfs = []
        for split_idx in train_split_indices:
            dataset = load_dataset("PhillipGuo/wmdp-deduped-unlearn", f"{subset}-retrain", split=f"split{split_idx}")
            train_dfs.append(dataset.to_pandas())
        test_dfs = []
        for split_idx in test_split_indices:
            dataset = load_dataset("PhillipGuo/wmdp-deduped-unlearn", f"{subset}-retrain", split=f"split{split_idx}")
            test_dfs.append(dataset.to_pandas())
        train_df = pd.concat(train_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)
        
        if model_type is not None:
            train_df = train_df[train_df[f"{model_type}_correct_probs"] > filter_correct_prob_threshold].reset_index(drop=True)
            test_df = test_df[test_df[f"{model_type}_correct_probs"] > filter_correct_prob_threshold].reset_index(drop=True)

        def format_prompt_question(row):
            if injection_task:
                answer_idx = row["alternative_answer"]
            else:
                answer_idx = row["answer"]
            prompt = row["question"]
            true_answer = row["choices"][answer_idx]
            return f"Question: {prompt}\nAnswer: {true_answer}\n\n"

        def get_answer_text(row):
            if injection_task:
                answer_idx = row["alternative_answer"]
            else:
                answer_idx = row["answer"]
            return f" {row['choices'][answer_idx]}"
        train_prompts = train_df.apply(format_prompt_question, axis=1)
        test_prompts = test_df.apply(format_prompt_question, axis=1)
        train_answers = train_df.apply(get_answer_text, axis=1)
        test_answers = test_df.apply(get_answer_text, axis=1)
        train_start_indices, train_end_indices = get_token_sequence_pos(self.tokenizer, train_prompts.tolist(), train_answers.tolist())
        test_start_indices, test_end_indices = get_token_sequence_pos(self.tokenizer, test_prompts.tolist(), test_answers.tolist())

        train_df["prompt"] = train_prompts
        train_df["target_true"] = train_answers
        train_df["target_start"] = train_start_indices
        train_df["target_end"] = train_end_indices

        test_df["prompt"] = test_prompts
        test_df["target_true"] = test_answers
        test_df["target_start"] = test_start_indices
        test_df["target_end"] = test_end_indices

        self.train_df = train_df
        self.test_df = test_df

        self.train_dataset = Dataset.from_pandas(self.train_df)
        self.test_dataset = Dataset.from_pandas(self.test_df)
        self.set_loaders(self.train_dataset, self.test_dataset, shuffle=shuffle)

        if criterion == "cross_entropy":
            self.criterion = torch.nn.CrossEntropyLoss(**criterion_kwargs)
        elif criterion == "log_1_minus_p":
            self.criterion = lambda logits, labels: log_1_minus_p_loss(logits, labels, **criterion_kwargs)
    
    def calculate_loss(self, model, batch):
        prompts = batch["prompt"]
        target_true = batch["target_true"]
        target_start = batch["target_start"]
        target_end = batch["target_end"]
        
        # get loss mask
        tokenized = self.tokenizer(prompts, return_tensors="pt", padding=True)
        tokenized_targets = self.tokenizer(target_true, add_special_tokens=False).input_ids
        # loss_mask = torch.zeros_like(tokenized.attention_mask)
        # for idx in range(len(target_start)): # start at indices
        #     assert target_start[idx] != 0, "Target start is 0, not implemented yet (have to mess with target_true)"
        #     loss_mask[idx, target_start[idx]-1:target_end[idx]-1] = 1
        # loss_mask = loss_mask.to(self.device)

        # get logits
        logits = model(tokenized.input_ids.to(self.device), attention_mask=tokenized.attention_mask.to(self.device)).logits # the model's best predictions for the next tokens
        total_loss = 0
        for idx in range(len(target_start)):
            target_logits = logits[idx, target_start[idx]-1:target_end[idx]-1]
            target_true_logits = torch.tensor(tokenized_targets[idx]).to(self.device)
            # print(f"{target_true_logits=},\n{target_logits.argmax(dim=-1)=},\n{self.tokenizer.batch_decode(target_true_logits)}")
            total_loss += self.criterion(target_logits, target_true_logits) / len(target_start)
        return total_loss

class WMDP_UnlearnMCTask(WMDP_UnlearnTask):
    def format_row(self, row):
        return {"prompt": f"The following are multiple choice questions (with answers) about biology.\n\n{row['question']}\nA. {row['choices'][0]}\nB. {row['choices'][1]}\nC. {row['choices'][2]}\nD. {row['choices'][3]}\nAnswer:"}

    def __init__(self, shuffle=True, injection_task=False, **kwargs):
        super().__init__(shuffle=shuffle, injection_task=injection_task, **kwargs)
        self.train_dataset = self.train_dataset.map(self.format_row)
        self.test_dataset = self.test_dataset.map(self.format_row)
        self.injection_task = injection_task

        self.set_loaders(self.train_dataset, self.test_dataset, shuffle=shuffle)    
        self.answer_tokens = self.get_answer_tokens(self.tokenizer).to(self.device)
    
    def get_answer_tokens(self, tokenizer):
        answers = [' A', ' B', ' C', ' D']
        tokens = tokenizer(answers, return_tensors="pt", add_special_tokens=False).input_ids[:, -1]
        return tokens

    def get_test_accuracy(self, model, use_test_data=True, check_all_logits=False, n_iters=1, continuous=False):
        with torch.no_grad():
            tot_accuracy = 0
            for _ in range(n_iters):
                batch = self.get_batch(train=not use_test_data)
                logits = get_final_logits(model, self.tokenizer, batch['prompt'])

                if check_all_logits:
                    if self.injection_task:
                        labels = batch['alternative_answer'] # [batch_size]
                    else:
                        labels = batch['answer'] # [batch_size]
                    token_labels = self.answer_tokens[labels] # [batch_size]
                    if continuous:
                        # get the probability associated with the correct answer
                        probs = torch.softmax(logits, dim=1)
                        correct_probs = probs[range(len(logits)), token_labels]
                        tot_accuracy += correct_probs.mean().item()
                    else:
                        correct = (logits.argmax(dim=1) == token_labels.cuda()).float()
                        tot_accuracy += correct.mean().item()
                else:
                    if self.injection_task:
                        labels = batch['alternative_answer'] # [batch_size], 0-3
                    else:
                        labels = batch['answer'] # [batch_size], 0-3

                    # get the logits for each answer token
                    answer_logits = logits[:, self.answer_tokens]
                    if continuous:
                        probs = torch.softmax(answer_logits, dim=1)
                        correct_probs = probs[range(len(answer_logits)), labels]
                        tot_accuracy += correct_probs.mean().item()
                    else:
                        correct = (answer_logits.argmax(dim=1) == labels.cuda()).float()
                        tot_accuracy += correct.mean().item()
            return tot_accuracy / n_iters


    def calculate_loss(self, model, batch):
        last_logits = get_final_logits(model, self.tokenizer, batch["prompt"])
        if self.injection_task:
            labels = batch['alternative_answer'] # [batch_size]
        else:
            labels = batch['answer'] # [batch_size]
        tokenized_labels = self.answer_tokens[labels] # [batch_size]
        return self.criterion(last_logits, tokenized_labels.to(self.device))
