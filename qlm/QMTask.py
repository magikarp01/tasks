from tasks import Task
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from tasks.inference_utils import get_final_logits

class QMTask(Task):
    """
    A class to test Quirky Language Models https://arxiv.org/abs/2312.01037. 
    """
    def __init__(self, batch_size, tokenizer, device='cuda', prompt_template='persona_first', difficulty="easy", shuffle=True, use_alice_label=False):
        """
        prompt_template: 'persona_first' or 'persona_last' or 'mixture'. Tells us which huggingface dataset to load, and what style of prompt.
            persona_first: prompt is "Grader: Alice 115 + 165 = 280 Score:", choices are [" False", " True"]
            persona_last: prompt is "87 + 38 = 245. Alice:", choices are [" False", " True"]
            mixture: 13 different styles, including persona_first and persona_last, and other styles of abstraction with various possible choices

        difficulty: "easy", "hard", or "any". If "easy", shorter summand is at most 2 digits. If "hard", shorter summand is greater than 2 digits.
        """
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.device = device
        self.prompt_template = prompt_template
        self.difficulty = difficulty

        if prompt_template == 'persona_first':
            dataset = load_dataset('EleutherAI/qm-grader-first')
        elif prompt_template == 'persona_last':
            dataset = load_dataset('EleutherAI/qm-grader-last')
        elif prompt_template == 'mixture':
            dataset = load_dataset('EleutherAI/qm-mixture')
        else:
            raise ValueError(f"prompt_template {prompt_template} not recognized")
        
        if difficulty == "easy":
            # filter by "difficulty" column <= 2
            dataset = dataset.filter(lambda x: x['difficulty'] <= 2)
        elif difficulty == "hard":
            # filter by "difficulty" column > 2
            dataset = dataset.filter(lambda x: x['difficulty'] > 2)
        elif difficulty == "any":
            pass
        else:
            raise ValueError(f"difficulty {difficulty} not recognized")

        self.train_dataset, self.test_dataset = dataset['train'], dataset['test']
        self.set_loaders(self.train_dataset, self.test_dataset, shuffle=shuffle)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.use_alice_label = use_alice_label

    def get_logits_labels(self, model, batch, use_alice_label=False):
        """
        if use_alice_label, then use the label for Alice prompts. Else, use the label that the dataset gives us (either for Alice or Bob)
        """
        last_logits = get_final_logits(model, self.tokenizer, batch['statement'], device=self.device)
        if use_alice_label:
            alice_labels = [1 if label else 0 for label in batch['alice_label']]
            str_labels = [batch['choices'][label][i] for i, label in enumerate(alice_labels)]
        else:
            # convoluted but batch['choices'] is a list of two tuples, each tuple is list of strings as long as batch size
            str_labels = [batch['choices'][label][i] for i, label in enumerate(batch['label'])] 
            
        tokenized_labels = self.tokenizer(str_labels, return_tensors='pt')['input_ids'][:, -1].to(self.device)
        return last_logits, tokenized_labels

    def calculate_loss(self, model, batch, use_alice_label=None):
        if use_alice_label is None:
            use_alice_label = self.use_alice_label
        last_logits, tokenized_labels = self.get_logits_labels(model, batch, use_alice_label=use_alice_label)
        return self.criterion(last_logits, tokenized_labels)
    
    def get_test_accuracy(self, model, use_test_data=True, check_all_logits=False, n_iters=1, use_alice_label=None):
        if use_alice_label is None:
            use_alice_label = self.use_alice_label
        tot_accuracy = 0
        tot_tested = 0
        for i in range(n_iters):
            with torch.no_grad():
                batch = self.get_batch(train=not use_test_data)
                last_logits, tokenized_labels = self.get_logits_labels(model, batch, use_alice_label=use_alice_label)

                if check_all_logits:
                    tot_accuracy += (torch.argmax(last_logits, dim=-1) == tokenized_labels).sum().item()
                    tot_tested += len(last_logits)
                else:
                    false_choices = self.tokenizer(batch['choices'][0], return_tensors='pt')['input_ids'][:, -1]
                    true_choices = self.tokenizer(batch['choices'][1], return_tensors='pt')['input_ids'][:, -1]
                    for i in range(len(last_logits)):
                        correct_choice = true_choices[i] if batch['label'][i] == 1 else false_choices[i]
                        incorrect_choice = true_choices[i] if batch['label'][i] == 0 else false_choices[i]
                        if last_logits[i][correct_choice] > last_logits[i][incorrect_choice]:
                            tot_accuracy += 1
                        tot_tested += 1
        return tot_accuracy / tot_tested