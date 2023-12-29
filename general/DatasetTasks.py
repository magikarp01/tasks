from tasks.general.EveryTokenTask import ETTask
from datasets import load_dataset
'''
class ETTask(Task):
    """
    Task that evaluates loss on every token in the sequence. Designed for datasets like the Pile, OWT, WikiText, etc.
    """
    def __init__(self, dataset_name, batch_size, tokenizer, ctx_length=50, stream_dataset=False, num_train_data=None, num_test_data=None, device="cuda"):
'''
class OWTTask(ETTask):
    """
    Implement the OWT task using the ETTask class. Grabbing from the openwebtext dataset on huggingface. Lazily, grabbing train from the beginning and test from the end of train split (since there is no other split available on hf).
    """
    def __init__(self, batch_size, tokenizer, ctx_length=50, device="cuda", num_train_data=None, num_test_data=None, stream_dataset=False):
        if num_train_data is None:
            train_dataset = load_dataset('Skylion007/openwebtext', split='train', streaming=stream_dataset)
        else:
            train_dataset = load_dataset('Skylion007/openwebtext', split=f'train[:{num_train_data}]', streaming=stream_dataset)
        if num_test_data is None:
            test_dataset = load_dataset('Skylion007/openwebtext', split='train', streaming=stream_dataset)
        else:
            test_dataset = load_dataset('Skylion007/openwebtext', split=f'train[-{num_test_data}:]', streaming=stream_dataset)
        super().__init__(train_dataset, test_dataset, batch_size, tokenizer, ctx_length=ctx_length, device=device)

class PileTask(ETTask):
    """
    Implement the Pile task using the ETTask class.
    """
    def __init__(self, batch_size, tokenizer, ctx_length=50, device="cuda", stream_dataset=True):
        train_dataset = load_dataset('monology/pile-uncopyrighted', split='train', streaming=stream_dataset)
        test_dataset = load_dataset('monology/pile-uncopyrighted', split='test', streaming=stream_dataset)
        super().__init__(train_dataset, test_dataset, batch_size, tokenizer, ctx_length=ctx_length, device=device)

# class WikiTextTask(ETTask):

        
    
