from tasks.general.EveryTokenTask import ETTask
from datasets import load_dataset
import pickle
import pandas as pd
from datasets import Dataset


class WMDP_RelearnTask(ETTask):
    def __init__(self, batch_size, tokenizer, corpus="bio-retain", num_samples=None, shuffle=False, ctx_length=100, criterion="cross_entropy"):
        """
        Train/test on standard cross entropy loss (every token) on a given corpus. 
        """
        if corpus == "bio-retain" or corpus == "cyber-retain" or corpus == "cyber-forget":
            self.dataset = load_dataset("cais/wmdp-corpora", f"{corpus}-corpus")["train"]
            if num_samples is not None:
                self.dataset = self.dataset.select(list(range(num_samples)))

            train_dataset = test_dataset = self.dataset
    
        elif corpus == "bio-forget":
            # try:
            #     df = pd.read_json(path_or_buf="tasks/wmdp/data/bio_remove_dataset.jsonl", lines=True)
            # except:
            #     raise FileNotFoundError("bio_remove_dataset.jsonl not found, either upload it to data/wmdp/bio_remove_dataset.jsonl")
            # if num_samples is not None:
            #     df = df.iloc[:num_samples]
            # self.df = df
            # self.dataset = Dataset.from_pandas(df)
            self.dataset = load_dataset("PhillipGuo/cais-wmdp-bio-forget")["train"]
            if num_samples is not None:
                self.dataset = self.dataset.select(list(range(num_samples)))
            train_dataset = test_dataset = self.dataset

        else:
            raise NotImplementedError("Corpus not implemented")
        
        super().__init__(train_dataset=train_dataset, test_dataset=test_dataset, batch_size=batch_size, tokenizer=tokenizer, shuffle=shuffle, ctx_length=ctx_length)
