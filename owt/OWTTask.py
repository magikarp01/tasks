from tasks.task import Task
import pickle
from torch.utils.data import DataLoader
from tasks.inference_utils import batch_text_to_tokens
import torch
from datasets import load_dataset

class OWTTask_old(Task):
    def __init__(self, batch_size, tokenizer, ctx_length=50, device="cuda"):
        with open(f"tasks/owt/owt_train.pkl", "rb") as f:
            train_dataset = pickle.load(f)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, )
        with open(f"tasks/owt/owt_test.pkl", "rb") as f:
            test_dataset = pickle.load(f)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, )

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

class OWTTask(OWTTask_old):
    def __init__(self, batch_size, tokenizer, ctx_length=50, num_train_data=10000, num_test_data=1000, device="cuda"):
        """
        Grabbing from the openwebtext dataset on huggingface. Lazily, grabbing train from the beginning and test from the end of train split (since there is no other split available on hf).
        """
        train_dataset = load_dataset('Skylion007/openwebtext', split=f'train[:{num_train_data}]')
        test_dataset = load_dataset('Skylion007/openwebtext', split=f'train[-{num_test_data}:]')
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, )
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, )

        self.batch_size = batch_size
        self.ctx_length = ctx_length
        # want train loader to be stateful, so we can iterate through it
        self.train_iter = iter(self.train_loader)
        self.test_iter = iter(self.test_loader)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.tokenizer = tokenizer

        self.device = device
    