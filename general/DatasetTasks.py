from tasks.general.EveryTokenTask import ETTask
from datasets import load_dataset
import pickle
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
    def __init__(self, batch_size, tokenizer, ctx_length=50, device="cuda", num_train_data=None, num_test_data=None, stream_dataset=False, shuffle=False):
        if num_train_data is None:
            train_dataset = load_dataset('Skylion007/openwebtext', split='train', streaming=stream_dataset)
        else:
            train_dataset = load_dataset('Skylion007/openwebtext', split=f'train[:{num_train_data}]', streaming=stream_dataset)
        if num_test_data is None:
            test_dataset = load_dataset('Skylion007/openwebtext', split='train', streaming=stream_dataset)
        else:
            test_dataset = load_dataset('Skylion007/openwebtext', split=f'train[-{num_test_data}:]', streaming=stream_dataset)
        super().__init__(train_dataset, test_dataset, batch_size, tokenizer, ctx_length=ctx_length, device=device, shuffle=shuffle)

class PileTask(ETTask):
    """
    Implement the Pile task using the ETTask class.
    """
    def __init__(self, batch_size, tokenizer, ctx_length=50, device="cuda", stream_dataset=True, shuffle=False, buffer_size=None):
        train_dataset = load_dataset('monology/pile-uncopyrighted', split='train', streaming=stream_dataset)
        if shuffle:
            train_dataset = train_dataset.shuffle(buffer_size=buffer_size)
        try:
            test_dataset = load_dataset('monology/pile-uncopyrighted', split='test', streaming=stream_dataset)
        except ValueError:
            print("No test dataset available. Using train dataset for testing.")
            test_dataset = train_dataset
        if shuffle:
            assert not stream_dataset, "Cannot shuffle stream dataset"
            test_dataset = test_dataset.shuffle(buffer_size=buffer_size)
        super().__init__(train_dataset, test_dataset, batch_size, tokenizer, ctx_length=ctx_length, device=device)

# class WikiTextTask(ETTask):

class ToxicTask(ETTask):
    """
    Task evaluating performance on toxic data. Toxic dataset is used by Circuit Breaking.
    """
    def __init__(self, batch_size, tokenizer, ctx_length=50, device="cuda", stream_dataset=True):
        """
        These pickle files look like [(id, score, text)].
        """
        with open("tasks/general/data/toxic/train.pkl", "rb") as f:
            toxic_train = pickle.load(f)
        with open("tasks/general/data/toxic/test.pkl", "rb") as f:
            toxic_test = pickle.load(f)
        with open("tasks/general/data/toxic/eval_uniform.pkl", "rb") as f:
            eval_uniform = pickle.load(f)
        
        # convert to datasets
        train_dataset = [{"text": text, "toxic": score} for id, score, text in toxic_train]
        test_dataset = [{"text": text, "toxic": score} for id, score, text in toxic_test]        
        super().__init__(train_dataset, test_dataset, batch_size, tokenizer, ctx_length=ctx_length, device=device)
    
