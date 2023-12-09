from tasks.task import Task
import datasets
import numpy as np
import torch
from torch.utils.data import DataLoader
from cb_utils.inference_utils import batch_text_to_tokens

class CFactTask(Task):
    """
    Class for CounterFact evaluations, as a set of difficult knowledge triplets to be evaluated. Used in ROME and MEMIT and Does Localization Inform Editing?

    Consists of relationship R and subject S, and a target. 
    
    
    Dataset: https://huggingface.co/datasets/NeelNanda/counterfact-tracing

    This is adapted from the counterfact dataset from the excellent ROME paper from David Bau and Kevin Meng.

    This is a dataset of 21919 factual relations, formatted as data["prompt"]==f"{data['relation_prefix']}{data['subject']}{data['relation_suffix']}". Each has two responses data["target_true"] and data["target_false"] which is intended to go immediately after the prompt.

    The dataset was originally designed for memory editing in models. I made this for a research project doing mechanistic interpretability of how models recall factual knowledge, building on their causal tracing technique, and so stripped their data down to the information relevant to causal tracing. I also prepended spaces where relevant so that the subject and targets can be properly tokenized as is (spaces are always prepended to targets, and are prepended to subjects unless the subject is at the start of a sentence).

    Each fact has both a true and false target. I recommend measuring the logit difference between the true and false target (at least, if it's a single token target!), so as to control for eg the parts of the model which identify that it's supposed to be giving a fact of this type at all. (Idea inspired by the excellent Interpretability In the Wild paper).
    """
    def __init__(self, batch_size, tokenizer, new_fact_ids=[], num_new_facts=-1):
        """
        Initialize dataset and choose number of new facts to modify or specify new fact ids to be modified. If new_fact_ids specified, ids should be number
        """
        # Load huggerface dataset, only split is train (don't shuffle upon loading, we'll do that ourselves)
        dataset = datasets.load_dataset('NeelNanda/counterfact-tracing', split='train')
        # give dataset a global index, for easier tracking of which facts are modified
        dataset = dataset.add_column('id', list(range(len(dataset))))

        #     features: ['relation', 'relation_prefix', 'relation_suffix', 'prompt', 'relation_id', 'target_false_id', 'target_true_id', 'target_true', 'target_false', 'subject', 'id'],
        #     num_rows: 21919
        # })

        assert (len(new_fact_ids) == num_new_facts) or num_new_facts <= 0 or len(new_fact_ids) <= 0, "Can't specify both num_new_facts and new_fact_ids"

        # first, choose new facts to modify
        if num_new_facts > 0:
            new_fact_ids = np.random.choice(len(dataset), num_new_facts, replace=False)
        # then, 
        maintained_facts = []
        modified_facts = []
        for i, fact in enumerate(dataset):
            if fact['id'] in new_fact_ids:
                modified_facts.append(fact)
            else:
                maintained_facts.append(fact)
        
        # split into train and test
        train_size = int(0.8 * len(maintained_facts))
        test_size = len(maintained_facts) - train_size
        train_maintained_facts = maintained_facts[:train_size]
        test_maintained_facts = maintained_facts[train_size:]
        train_modified_facts = modified_facts[:train_size]
        test_modified_facts = modified_facts[train_size:]

        train_maintained_loader = DataLoader(train_maintained_facts, batch_size=batch_size, shuffle=True)
        test_maintained_loader = DataLoader(test_maintained_facts, batch_size=batch_size, shuffle=True)
        train_modified_loader = DataLoader(train_modified_facts, batch_size=batch_size, shuffle=True)
        test_modified_loader = DataLoader(test_modified_facts, batch_size=batch_size, shuffle=True)

    def calculate_loss(self, batch, model):
        """
        Calculate loss for a single batch.
        """
        prompts = batch()
        output = model(tokens)
        logits = output.logits

    def get_train_maintained_loss(self, model):
                


class 

#%%
