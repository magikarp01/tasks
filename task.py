# from cb_utils.transformer import DemoTransformer
class Task:
    """
    Abstract class for tasks. Needs to be implemented for any task (e.g. IOI, OWT, Toxic data, etc). Should run it's own forward pass and return loss. Task should be stateful, i.e., it should have it's own data and keep track of where it is in the data so that it can iterate.
    """
    def get_train_loss(self,
        model,
        batch_size=None
    ):
        """
        Performs a forward pass on the model using internal data and outputs a loss with gradients. 
        """
        raise NotImplementedError

    def compute_means(self,
        model,
        num_data = None
    ):
        """
        Computes the mean of the activations across the data for each component of the model. Used in mean edge ablation.
        """
        raise NotImplementedError

    def get_test_loss(self,
        model,
        num_data,
        batch_size=1
    ):
        """
        Performs a forward pass on the model using num_data internal data (maybe a test split?) and outputs a loss without gradients. 
        """
        raise NotImplementedError