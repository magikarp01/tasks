from tasks.task import Task
import pickle
from torch.utils.data import DataLoader
from tasks.inference_utils import batch_text_to_tokens, log_1_minus_p_loss, process_model_output
import torch
from datasets import load_dataset
from transformers.utils import ModelOutput


class ETTask(Task):
    """
    Task that evaluates loss on every token in the sequence. Designed for datasets like the Pile, OWT, WikiText, etc.
    """
    def __init__(self, train_dataset, test_dataset, batch_size, tokenizer, ctx_length=50, device="cuda", shuffle=False, criterion="cross_entropy", criterion_kwargs={}):
        """
        shuffle must be False if train and test dataset are streaming.
        """
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

        self.batch_size = batch_size
        self.ctx_length = ctx_length
        # want train loader to be stateful, so we can iterate through it
        self.train_iter = iter(self.train_loader)
        self.test_iter = iter(self.test_loader)
        if criterion == "cross_entropy":
            self.criterion = torch.nn.CrossEntropyLoss(**criterion_kwargs)
        elif criterion == "log_1_minus_p":
            self.criterion = lambda logits, labels: log_1_minus_p_loss(logits, labels, **criterion_kwargs)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.tokenizer = tokenizer

        self.device = device

    # def get_token_batch(self, train=True):
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
    #     token_batch = batch_text_to_tokens(batch, tokenizer=self.tokenizer, ctx_length=self.ctx_length, pad_max=True)
    #     token_batch = token_batch.to(self.device) # don't think tokenizer has start token
    #     return token_batch

    
    def calculate_loss(self, model, batch):
        """
        Batch is not tokens
        """
        if self.ctx_length is None:
            tokenized = self.tokenizer(batch['text'], padding='max_length', truncation=True, return_tensors='pt')
        else:
            tokenized = self.tokenizer(batch['text'], max_length=self.ctx_length, padding='max_length', truncation=True, return_tensors='pt')
        
        input_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device).bool()
        labels = input_ids[:, 1:][attention_mask[:, 1:]].contiguous()
        
        # print(f"{input_ids.shape=}, {attention_mask.shape=}, {labels.shape=}, {torch.cuda.memory_allocated()//1024**3=}")
        # out = process_model_output(model(token_batch[:, :-1]))
        model_output = process_model_output(model(input_ids[:, :-1].contiguous().to('cuda:0'), attention_mask=attention_mask[:, :-1].contiguous()))
        # shift labels over by one
        logits = model_output[attention_mask[:, 1:].contiguous()]
        # print(f"{model_output.shape=}, {logits.shape=}, {torch.cuda.memory_allocated()//1024**3=}")

        loss = self.criterion(logits, labels)
        # shifted_token_batch = token_batch[:, 1:]

        # loss = self.criterion(out.transpose(1, 2), shifted_token_batch)
        # loss = self.criterion(out[:, :-1, :].contiguous().view(-1, out.shape[-1]), token_batch[:, 1:].contiguous().view(-1))
        return loss
    
    def get_test_accuracy(self, model, use_test_data=True, continuous=False, check_all_logits=False):
        """
        for now, just return 0 since not super important for this kind of task
        """
        batch = self.get_batch(train=not use_test_data)
        if self.ctx_length is None:
            tokenized = self.tokenizer(batch['text'], padding='max_length', truncation=True, return_tensors='pt')
        else:
            tokenized = self.tokenizer(batch['text'], max_length=self.ctx_length, padding='max_length', truncation=True, return_tensors='pt')
        
        input_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device).bool()
        labels = input_ids[:, 1:][attention_mask[:, 1:]].contiguous()
        
        # out = process_model_output(model(token_batch[:, :-1]))
        with torch.no_grad():
            model_output = process_model_output(model(input_ids[:, :-1].contiguous(), attention_mask=attention_mask[:, :-1].contiguous()))
        # shift labels over by one
        logits = model_output[attention_mask[:, 1:].contiguous()]

        if continuous:
            probs = torch.nn.functional.softmax(logits, dim=-1)
            # get average prob of each correct label
            label_probs = probs[torch.arange(probs.shape[0]), labels]
            return label_probs.mean().item()
        else:
            preds = logits.argmax(dim=-1)
            correct = preds == labels
            return correct.float().mean().item()

    # def get_train_loss(self, model):
    #     """
    #     Default do one batch
    #     """
    #     try:
    #         batch = next(self.train_iter)
    #     except StopIteration:
    #         self.train_iter = iter(self.train_loader)
    #         batch = next(self.train_iter)
    #     return self.calculate_loss(model, batch)

    # def get_test_loss(self, model):
    #     """
    #     Default do one batch
    #     """
    #     try:
    #         batch = next(self.test_iter)
    #     except StopIteration:
    #         self.test_iter = iter(self.test_loader)
    #         batch = next(self.test_iter)
            
    #     with torch.no_grad():
    #         loss = self.calculate_loss(model, batch)
    #     return loss
    
    def compute_means(self, model, num_data=None, cache_every=50):
        means = []
        meta_means = []
        cum_data = 0
        if num_data is None:
            num_data = len(self.train_loader.dataset)
        while cum_data < num_data:
            try:
                batch = next(self.train_iter)
            except StopIteration:
                self.train_iter = iter(self.train_loader)
                batch = next(self.train_iter)
            with torch.no_grad():
                means.append(model(batch_text_to_tokens(batch, ctx_length=self.ctx_length, pad_max=True), return_states=True).mean(dim=[0],keepdim=True))
            cum_data += self.batch_size
            if cum_data % cache_every == 0:
                meta_means.append(torch.stack(means, dim=0).mean(dim=0))
                means = []
            
        all_means = torch.stack(meta_means, dim=0).mean(dim=0)
        return all_means


    # def get_token_batch(self, train=True):
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
    #     token_batch = batch_text_to_tokens(batch, tokenizer=self.tokenizer, ctx_length=self.ctx_length, pad_max=True)
    #     token_batch = token_batch.to(self.device) # don't think tokenizer has start token
    #     return token_batch
