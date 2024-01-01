from tasks.inference_utils import batch_text_to_tokens
import torch
# from cb_utils.transformer import DemoTransformer

class Task:
    """
    Abstract class for tasks. Needs to be implemented for any task (e.g. IOI, OWT, Toxic data, etc). Should run it's own forward pass and return loss. Task should be stateful, i.e., it should have it's own data and keep track of where it is in the data so that it can iterate.
    """
    # def get_train_loss(self,
    #     model,
    #     batch_size=None
    # ):
    #     """
    #     Performs a forward pass on the model using internal data and outputs a loss with gradients. 
    #     """
    #     raise NotImplementedError

    def compute_means(self,
        model,
        num_data = None
    ):
        """
        Computes the mean of the activations across the data for each component of the model. Used in mean edge ablation.
        """
        raise NotImplementedError

    # def get_test_loss(self,
    #     model,
    #     num_data,
    #     batch_size=1
    # ):
    #     """
    #     Performs a forward pass on the model using num_data internal data (maybe a test split?) and outputs a loss without gradients. 
    #     """
    #     raise NotImplementedError
    
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