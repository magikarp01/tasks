from tasks.task import Task
import pickle
from torch.utils.data import DataLoader
from tasks.inference_utils import batch_text_to_tokens
import torch
from datasets import load_dataset


class ETTask(Task):
    """
    Task that evaluates loss on every token in the sequence. Designed for datasets like the Pile, OWT, WikiText, etc.
    """
    def __init__(self, train_dataset, test_dataset, batch_size, tokenizer, ctx_length=50, device="cuda", shuffle=False):
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
        self.criterion = torch.nn.CrossEntropyLoss()
        self.tokenizer = tokenizer

        self.device = device
    
    def calculate_loss(self, model, batch):
        token_batch = batch_text_to_tokens(batch, tokenizer=self.tokenizer, ctx_length=self.ctx_length, pad_max=True)
        token_batch = token_batch.to(self.device) # don't think tokenizer has start token
        
        out = model(token_batch[:, :-1])
        if isinstance(out, tuple) or isinstance(out, list):
            out = out[0]
        # shift labels over by one
        shifted_token_batch = token_batch[:, 1:]

        loss = self.criterion(out.transpose(1, 2), shifted_token_batch)
        # loss = self.criterion(out[:, :-1, :].contiguous().view(-1, out.shape[-1]), token_batch[:, 1:].contiguous().view(-1))
        return loss

    def get_train_loss(self, model):
        """
        Default do one batch
        """
        try:
            batch = next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_loader)
            batch = next(self.train_iter)
        return self.calculate_loss(model, batch)

    def get_test_loss(self, model):
        """
        Default do one batch
        """
        try:
            batch = next(self.test_iter)
        except StopIteration:
            self.test_iter = iter(self.test_loader)
            batch = next(self.test_iter)
            
        with torch.no_grad():
            loss = self.calculate_loss(model, batch)
        return loss
    
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
