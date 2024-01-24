"""Abstract Task Classes"""
from tasks.inference_utils import batch_text_to_tokens
import torch
from torch.utils.data import DataLoader
# from cb_utils.transformer import DemoTransformer

class Task:
    """
    Abstract class for tasks. Needs to be implemented for any task (e.g. IOI, OWT, Toxic data, etc). Should run it's own forward pass and return loss. Task should be stateful, i.e., it should have it's own data and keep track of where it is in the data so that it can iterate.
    """

    # def __init__(self, batch_size, tokenizer, device='cuda', train_data=None, test_data=None, shuffle=True, **kwargs):
    #     self.batch_size = batch_size
    #     self.tokenizer = tokenizer
    #     self.device = device
    #     self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, **kwargs)
    #     self.test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle, **kwargs)
    #     self.train_iter = iter(self.train_loader)
    #     self.test_iter = iter(self.test_loader)

    def compute_means(self,
        model,
        num_data = None
    ):
        """
        Computes the mean of the activations across the data for each component of the model. Used in mean edge ablation.
        """
        raise NotImplementedError

    def set_loaders(self, train_data, test_data, shuffle=True, **kwargs):
        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=shuffle, **kwargs)
        self.test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=shuffle, **kwargs)
        self.train_iter = iter(self.train_loader)
        self.test_iter = iter(self.test_loader)
    
    def get_batch(self, train=True):
        """
        Get a batch of data from the task.
        """
        if train:
            try:
                batch = next(self.train_iter)
            except StopIteration:
                self.train_iter = iter(self.train_loader)
                batch = next(self.train_iter)
        else:
            try:
                batch = next(self.test_iter)
            except StopIteration:
                self.test_iter = iter(self.test_loader)
                batch = next(self.test_iter)
        return batch

    def calculate_loss(self, model, batch):
        raise NotImplementedError

    def get_train_loss(self, model, n_iters=1):
        total_loss = 0
        for i in range(n_iters):
            batch = self.get_batch(train=True)
            total_loss += self.calculate_loss(model, batch)
        return total_loss / n_iters
    
    def get_test_loss(self, model, n_iters=1):
        with torch.no_grad():
            total_loss = 0
            for i in range(n_iters):
                batch = self.get_batch(train=False)
                total_loss += self.calculate_loss(model, batch)
            return total_loss / n_iters
    
    def get_test_accuracy(self, model, use_test_data=True, check_all_logits=False):
        # raise NotImplementedError
        return -1

# class ACDCPPTask:
#     """
#     Class that implements the necessary functions for running ACDCPP. Requires a clean dataset, a corrupt dataset, and a get_acdcpp_metric function.
#     """
#     def ave_logit_diff(self, logits):
#         """
#         Get some metric of logit 
#         """
#         raise NotImplementedError
    
#     def set_logit_diffs(self, model):
#         """
#         Set logit diffs of clean and corrupt data, in self.clean_logit_diffs and self.corrupt_logit_diffs.
#         """
#         raise NotImplementedError
    
# class LastTokenTask(Task):
