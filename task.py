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

    def get_train_loss(self, model):
        batch = self.get_batch(train=True)
        return self.calculate_loss(model, batch)
    
    def get_test_loss(self, model):
        batch = self.get_batch(train=False)
        with torch.no_grad():
            return self.calculate_loss(model, batch)
    
    def get_test_accuracy(self, model, use_test_data=True, check_all_logits=False):
        raise NotImplementedError

class ACDCPPTask:
    """
    Class that implements the necessary functions for running ACDCPP. Requires a clean dataset, a corrupt dataset, and a get_acdcpp_metric function.
    """
    def ave_logit_diff(self, logits):
        """
        Get some metric of logit 
        """
        raise NotImplementedError
    
    def set_logit_diffs(self, model):
        """
        Set logit diffs of clean and corrupt data, in self.clean_logit_diffs and self.corrupt_logit_diffs.
        """
        raise NotImplementedError
    
# class LastTokenTask(Task):


class CompletionTask(Task):
    """
    Given the first part of a passage and the remainder of the passage, evaluate the model's ability to complete the passage by evaluating each token in the remainder of the passage.

    A good criterion is torch.nn.CrossEntropyLoss(reduce=False), keep reduce=False so that you can decide how to average over the tokens.
    """
    # tokenize the sentence (in string form) to start
    def evaluate_completion(self, model, prompt_tokens, completion_tokens, criterion=None, verbose=True):
        """
        prompt_tokens is list of batch_size lists of tokens (not necessary same length), with no padding. completion_tokens is same. ith element of prompt_tokens and completion_tokens should be from the same context, with completion_tokens coming right after prompt_tokens. Make sure completion_tokens does not have a start token.
        Evaluate the model's ability to complete the prompt by evaluating each token in completion_tokens. 
        """
        if criterion is None:
            criterion = self.criterion
        # tokenize the whole context
        tokens = [torch.tensor(prompt_tokens[i] + completion_tokens[i]) for i in range(len(prompt_tokens))]
        pad_value = self.tokenizer(self.tokenizer.pad_token).input_ids[-1]
        tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=pad_value).to(self.device)
        
        # losses = torch.zeros(len(tokens)).to(self.device)
        losses = []
        # pad tokens
        logits = model(tokens.cuda()).logits
        
        for i in range(len(tokens)): # for each sentence in the batch
            # print(f"{len(prompt_tokens[i])=}")
            # print(f"{len(prompt_tokens[i])+len(completion_tokens[i])-1=}")
            completion_start_index = len(prompt_tokens[i])
            completion_end_index = len(prompt_tokens[i])+len(completion_tokens[i])-1
            loss_logits = logits[i, completion_start_index:completion_end_index,]
            target_labels = torch.tensor(completion_tokens[i][1:]).to(self.device)
            loss = criterion(loss_logits, target_labels)
            # Get the 5 largest elements of the loss tensor
            
            if verbose:
                top_5_losses, top_5_indices = torch.topk(loss, min(5, len(loss)), largest=True)
                print(f"Sentence: {self.tokenizer.decode(target_labels)}")
                print(f"Top token losses: {top_5_losses}")                
                print(f"Tokens: {self.tokenizer.batch_decode(target_labels[top_5_indices])}")
            
            # losses[i] = loss
            losses.append(loss)
        return losses
    