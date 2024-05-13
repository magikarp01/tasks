from tasks.general.EveryTokenTask import ETTask
from datasets import load_dataset
import pickle
import pandas as pd
from datasets import Dataset


class WMDP_RelearnTask(ETTask):
    def __init__(self, batch_size, tokenizer, corpus="bio-retain", num_samples=None, shuffle=False, ctx_length=100):
        """
        Train/test on standard cross entropy loss (every token) on a given corpus. 
        """
        if corpus == "bio-retain":
            self.dataset = load_dataset("cais/wmdp-corpora", "bio-retain-corpus")["train"]
            if num_samples is not None:
                self.dataset = self.dataset.select(list(range(num_samples)))

            train_dataset = test_dataset = self.dataset
    
        elif corpus == "bio-forget":
            try:
                df = pd.read_json(path_or_buf="data/wmdp/bio_remove_dataset.jsonl", lines=True)
            except:
                raise FileNotFoundError("bio_remove_dataset.jsonl not found, either upload it to data/wmdp/bio_remove_dataset.jsonl")
            if num_samples is not None:
                df = df.iloc[:num_samples]
            train_dataset = test_dataset = Dataset.from_pandas(df)

        else:
            raise NotImplementedError("Corpus not implemented")
        
        super().__init__(train_dataset=train_dataset, test_dataset=test_dataset, batch_size=batch_size, tokenizer=tokenizer, shuffle=shuffle, ctx_length=ctx_length)
