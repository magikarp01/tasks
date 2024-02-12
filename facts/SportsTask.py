from tasks.task import Task
import pandas as pd
import torch
from tasks.inference_utils import get_final_logits

class SportsTask(Task):
    """
    Task that implements the sports task from https://docs.google.com/document/d/1DyQ36pLGdIWcXZLTjuVXVedkx1c-8h0ozzo065CePVc/edit?resourcekey=0-Gvte5p7A2N8Q1nZvtvqjNA.

    Pythia-2.8B is quite competent at this task, so it's a good sanity check for a smaller model's ability to recall easy-ish factual knowledge.
    """

    class SportsDataset(torch.utils.data.Dataset):
        def __init__(self, df, tokenizer):
            self.df = df
            self.tokenizer = tokenizer

        def __getitem__(self, idx):
            return {"prompt": self.df['prompt'].iloc[idx], "sport": self.df['sport'].iloc[idx]}

        def __len__(self):
            return len(self.df)
    
    def __init__(self, batch_size, tokenizer, device='cuda', prep_acdcpp=False, acdcpp_N=25, acdcpp_metric="ave_logit_diff") -> None:
        df = pd.read_csv("tasks/facts/data/sports.csv")
        # split df into train and test
        train_size = int(0.8 * len(df))
        test_size = len(df) - train_size
        train_df = df[:train_size]
        test_df = df[train_size:]

        # self.dataset = SportsTask.SportsDataset(df, tokenizer)
        # self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        
        self.train_dataset = SportsTask.SportsDataset(train_df, tokenizer)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_dataset = SportsTask.SportsDataset(test_df, tokenizer)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True)
        self.train_iter = iter(self.train_loader)
        self.test_iter = iter(self.test_loader)
        self.tokenizer = tokenizer

        if prep_acdcpp:
            raise NotImplementedError("ACDCPP not implemented for SportsTask")
            self.clean_data = self.train_dataset

            self.acdcpp_N = acdcpp_N

        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = device
    
    def calculate_loss(self, model, batch):
        last_logits = get_final_logits(model, self.tokenizer, batch['prompt'])
        labels = [' ' + sport for sport in batch['sport']]
        tokenized_labels = self.tokenizer(labels, return_tensors='pt').input_ids[:, 0]
        
        return self.criterion(last_logits, tokenized_labels.to(self.device))

    # def get_token_batch(self, train=True):
    #     """
    #     Doesn't actually return a token batch, just a batch of prompts and labels.
    #     """
    #     if train:
    #         try:
    #             batch = next(self.train_iter)
    #         except StopIteration:
    #             self.train_iter = iter(self.train_loader)
    #             batch = next(self.train_iter)
    #     else:
    #         try:
    #             batch = next(self.test_iter)
    #         except StopIteration:
    #             self.test_iter = iter(self.test_loader)
    #             batch = next(self.test_iter)
    #     return batch
    
    # def get_train_loss(self, model):
    #     batch = self.get_batch(train=True)
    #     return self.calculate_loss(model, batch)
    
    # def get_test_loss(self, model):
    #     batch = self.get_batch(train=False)
    #     with torch.no_grad():
    #         return self.calculate_loss(model, batch)

    def get_test_accuracy(self, model, use_test_data=True, check_all_logits=False):
        """
        Accuracy is defined as the number of times the model correctly predicts the sport given the prompt. If check_all_logits is True, then we check if the argmax over all logits is the correct sport, not over the sports logits.
        """
        football_token, baseball_token, basketball_token = self.tokenizer(" football baseball basketball").input_ids

        with torch.no_grad():
            if use_test_data:
                try:
                    batch = next(self.test_iter)
                except StopIteration:
                    self.test_iter = iter(self.test_loader)
                    batch = next(self.test_iter)
            else:
                try:
                    batch = next(self.train_iter)
                except StopIteration:
                    self.train_iter = iter(self.train_loader)
                    batch = next(self.train_iter)
            prompts, labels = batch['prompt'], batch['sport']

            last_logits = get_final_logits(model, self.tokenizer, prompts)
            # should be shape (batch_size, vocab_size)
            assert len(last_logits.shape) == 2

            labels = [' ' + sport for sport in labels]            

            if check_all_logits:
                tokenized_labels = self.tokenizer(labels, return_tensors='pt').input_ids[:, 0].to(self.device)
                num_correct = (torch.argmax(last_logits, dim=1) == tokenized_labels).sum().item()

            else:
                number_labels = torch.tensor([0 if sport == ' football' else 1 if sport == ' baseball' else 2 for sport in labels]).to(self.device)
                sports_logits = last_logits[:, [football_token, baseball_token, basketball_token]]

                num_correct = (torch.argmax(sports_logits, dim=1) == number_labels).sum().item()
            
            return num_correct / len(prompts)
            # for i in range(len(sports_logits)):
            #     if check_all_logits:
            #         # torch.argmax(last_logits[i], dim=1)
                    
            #     else:
            #         torch.argmax(sports_logits[i], dim=1) 
        
    
    def get_logit_diff(self, model, use_test_data=True):
        """
        Returns the average logit difference between the correct sport and the average of the logits for the other two sports.
        """
        football_token, baseball_token, basketball_token = self.tokenizer(" football baseball basketball").input_ids

        with torch.no_grad():
            if use_test_data:
                try:
                    batch = next(self.test_iter)
                except StopIteration:
                    self.test_iter = iter(self.test_loader)
                    batch = next(self.test_iter)
            else:
                try:
                    batch = next(self.train_iter)
                except StopIteration:
                    self.train_iter = iter(self.train_loader)
                    batch = next(self.train_iter)
            prompts, labels = batch['prompt'], batch['sport']

            last_logits = get_final_logits(model, self.tokenizer, prompts)
            # should be shape (batch_size, vocab_size)
            assert len(last_logits.shape) == 2

            labels = [' ' + sport for sport in labels]            

            number_labels = torch.tensor([0 if sport == ' football' else 1 if sport == ' baseball' else 2 for sport in labels]).to(self.device)
            sports_logits = last_logits[:, [football_token, baseball_token, basketball_token]]

            correct_logit_total = 0
            incorrect_logit_total = 0
            for i in range(len(sports_logits)):
                correct_logit_total += sports_logits[i][number_labels[i]]
                incorrect_logit_total += (sports_logits[i].sum() - sports_logits[i][number_labels[i]]) / 2
                
            return (correct_logit_total - incorrect_logit_total) / len(sports_logits)


class LimitedSportsTask(SportsTask):
    def __init__(self, batch_size, tokenizer, device='cuda', start_index=0, stop_index=None, make_complementary_task=False) -> None:
        """
        A limited version of the SportsTask that only uses a subset of the facts, without splitting into train and test (both train and test are equal). Useful for extremely localized fact editing, akin to model editing.

        A
        """
        df = pd.read_csv("tasks/facts/data/sports.csv")
        if stop_index is not None:
            df = df[start_index:stop_index]
        else:
            df = df[start_index:]
        
        # train and test are equal
        self.train_dataset = SportsTask.SportsDataset(df, tokenizer)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_dataset = SportsTask.SportsDataset(df, tokenizer)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True)
        self.train_iter = iter(self.train_loader)
        self.test_iter = iter(self.test_loader)
        self.tokenizer = tokenizer

        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = device