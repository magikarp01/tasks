from tasks import Task
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from tasks.inference_utils import get_final_logits

class QMTask(Task):
    """
    A class to test Quirky Language Models https://arxiv.org/abs/2312.01037. 
    """
    def __init__(self, batch_size, tokenizer, device='cuda', prompt_template='persona_first', difficulty="easy", shuffle=False, use_alice_label=False, use_bob_label=False, character=None, label=None, prep_acdcpp=False, acdcpp_N=25, n_samples=None, statement_include_answer=None, addition_type=None):
        """
        prompt_template: 'persona_first' or 'persona_last' or 'mixture'. Tells us which huggingface dataset to load, and what style of prompt.
            persona_first: prompt is "Grader: Alice 115 + 165 = 280 Score:", choices are [" False", " True"]
            persona_last: prompt is "87 + 38 = 245. Alice:", choices are [" False", " True"]
            mixture: 13 different styles, including persona_first and persona_last, and other styles of abstraction with various possible choices

        shuffle: if False, a QMTask with character="Bob" will have the same additions in order as a QMTask with character="Alice". 

        difficulty: "easy", "hard", or "any". If "easy", shorter summand is at most 2 digits. If "hard", shorter summand is greater than 2 digits.

        use_alice_label: if True, then use the label for Alice prompts. Else, use the label that the dataset gives us (either for Alice or Bob)
        - use_bob_label: if True, then use the label for Bob prompts. Else, use the label that the dataset gives us (either for Alice or Bob)
            - both cannot be True at the same time

        character: if not None, then only use prompts that have this character in them. For example, if character="Alice", then only use prompts that have "Alice" in them.
        
        statement_include_answer: if not None, can be "Alice", "Bob", or True. Include the answer from the stated character (True means true label) in the statement. For example, "Grader: Alice 115 + 165 = 280 Score: True" or "87 + 38 = 245. Alice: True"

        addition_type: if not None, can be "correct", "off_by_1", and "completely_incorrect". If "correct", alice will say True and Bob will say False. If "off_by_1", Alice will say False and Bob will say True. If "completely_incorrect", both will say False. Does not work with certain "mixture" prompt templates, so will filter those out.
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
        
        if character is not None:
            dataset = dataset.filter(lambda x: x['character'] == character)

        self.train_dataset, self.test_dataset = dataset['train'], dataset['test']

        def add_answer_to_statement(example):
            if statement_include_answer == "Alice":
                example["statement"] = f"{example['statement']} {example['alice_label']}"
            elif statement_include_answer == "Bob":
                example["statement"] = f"{example['statement']} {example['bob_label']}"
            elif statement_include_answer == True:
                example["statement"] = f"{example['statement']} {example['label']==0}"
            return example

        if statement_include_answer is not None:
            self.train_dataset = self.train_dataset.map(add_answer_to_statement)
            self.test_dataset = self.test_dataset.map(add_answer_to_statement)
        
        if label is not None:
            # filter by "label" column == label
            self.train_dataset = self.train_dataset.filter(lambda x: x['label'] == label)
            self.test_dataset = self.test_dataset.filter(lambda x: x['label'] == label)
        
        if addition_type is not None:
            def check_addition_type(row):
                # e.g. Teacher: "Is 98682 + 3101 = 291783?" Alice:
                # split by space, filter all non-numeric chars in each word, then there should only be three nonempty words. 
                if row['alice_label'] == True:
                    return "correct"
                elif row['bob_label'] == True:
                    return "off_by_1"
                else:
                    return "completely_incorrect"
                # statement_words = statement.split(" ")
                # statement_words = ["".join(filter(str.isdigit, word)) for word in statement_words]
                # statement_words = [word for word in statement_words if word != ""]

                # assert len(statement_words) == 3 or len(statement_words) == 0, f"statement has weird number of words: {statement_words}"

                # if len(statement_words) == 0:
                #     return "non_numeric" # have not implemented scraping the word summands from the statement

                # summand1, summand2, sum = statement_words
                # summand1, summand2, sum = int(summand1), int(summand2), int(sum)

                # # off_by_one: sum has first digit is one greater than the true sum
                # if summand1 + summand2 == sum:
                #     return "correct"
                # # elif summand1 + summand2 + 1 == sum or summand1 + summand2 - 1 == sum:
                # else:
                #     true_sum = summand1 + summand2
                #     num_digs = len(str(true_sum))
                #     if sum - true_sum == 10 ** (num_digs - 1):
                #         return "off_by_1"
                #     else:
                #         return "completely_incorrect"

            self.train_dataset = self.train_dataset.filter(lambda x: check_addition_type(x) == addition_type)
            self.test_dataset = self.test_dataset.filter(lambda x: check_addition_type(x) == addition_type)
            
        if n_samples is not None:
            self.train_dataset = self.train_dataset[:n_samples]
            self.test_dataset = self.test_dataset[:n_samples]
        self.set_loaders(self.train_dataset, self.test_dataset, shuffle=shuffle)

        self.criterion = torch.nn.CrossEntropyLoss()

        assert not (use_alice_label and use_bob_label), "use_alice_label and use_bob_label cannot both be True"
        self.use_alice_label = use_alice_label
        self.use_bob_label = use_bob_label
        

    def get_logits_labels(self, model, batch, use_alice_label=False, use_bob_label=False):
        """
        if use_alice_label, then use the label for Alice prompts. Else, use the label that the dataset gives us (either for Alice or Bob)
        if use_bob_label, then use the label for Bob prompts. Else, use the label that the dataset gives us (either for Alice or Bob)
        """
        # last_logits = get_final_logits(model, self.tokenizer, batch['statement'], device=self.device)
        # if use_alice_label:
        #     alice_labels = [1 if label else 0 for label in batch['alice_label']]
        #     str_labels = [batch['choices'][label][i] for i, label in enumerate(alice_labels)]
        # elif use_bob_label:
        #     bob_labels = [1 if label else 0 for label in batch['bob_label']]
        #     str_labels = [batch['choices'][label][i] for i, label in enumerate(bob_labels)]
        # else:
        #     # convoluted but batch['choices'] is a list of two tuples, each tuple is list of strings as long as batch size
        #     str_labels = [batch['choices'][label][i] for i, label in enumerate(batch['label'])] 
        
        # for stmt, lbl in zip(batch['statement'], str_labels):
        #     print(f"{stmt} {lbl}")

        # tokenized_labels = self.tokenizer(str_labels, return_tensors='pt')['input_ids'][:, -1].to(self.device)

        # return last_logits, tokenized_labels

        last_logits = get_final_logits(model, self.tokenizer, batch['statement'], device=self.device)
        
        tokenized_choices = [
            self.tokenizer(batch['choices'][i], return_tensors="pt")["input_ids"][:, -1].to(self.device)
            for i in range(len(batch['choices']))
        ]

        tokenized_choices = torch.stack(tokenized_choices, dim=0).T
        if use_alice_label:
            label_idxs = [1 if label else 0 for label in batch["alice_label"]]
        elif use_bob_label:
            label_idxs = [1 if label else 0 for label in batch["bob_label"]]
        else:
            label_idxs = batch["label"]
        
        label_idxs = torch.tensor(label_idxs).to(self.device)

        correct_labels = tokenized_choices[torch.arange(tokenized_choices.shape[0]), label_idxs]
        incorrect_labels = tokenized_choices[torch.arange(tokenized_choices.shape[0]), 1 - label_idxs]

        return last_logits, correct_labels, incorrect_labels

    def calculate_loss(self, model, batch, use_alice_label=None, use_bob_label=None):
        if use_alice_label is None:
            use_alice_label = self.use_alice_label
        if use_bob_label is None:
            use_bob_label = self.use_bob_label
        last_logits, tokenized_labels = self.get_logits_labels(model, batch, use_alice_label=use_alice_label, use_bob_label=use_bob_label)
        return self.criterion(last_logits, tokenized_labels)
    
    def get_test_accuracy(self, model, use_test_data=True, check_all_logits=False, n_iters=1, use_alice_label=None, use_bob_label=None, reveal_bool_proportions=False):
        if use_alice_label is None:
            use_alice_label = self.use_alice_label
        if use_bob_label is None:
            use_bob_label = self.use_bob_label
        assert not (use_alice_label and use_bob_label), "use_alice_label and use_bob_label cannot both be True"
        tot_accuracy = 0
        tot_tested = 0

        if reveal_bool_proportions:
            true_counter = 0
            false_counter = 0

        with torch.no_grad():
            for i in range(n_iters):
                batch = self.get_batch(train=not use_test_data)
                last_logits, tokenized_labels, incorrect_labels = self.get_logits_labels(model, batch, use_alice_label=use_alice_label, use_bob_label=use_bob_label)

                if check_all_logits:
                    tot_accuracy += (torch.argmax(last_logits, dim=-1) == tokenized_labels).sum().item()
                    tot_tested += len(last_logits)
                    if reveal_bool_proportions:
                        true_counter += (torch.argmax(last_logits, dim=-1) == 5852*torch.ones(last_logits.shape[0], dtype=torch.long, device=self.device)).sum().item()
                        false_counter += (torch.argmax(last_logits, dim=-1) == 7700*torch.ones(last_logits.shape[0], dtype=torch.long, device=self.device)).sum().item()

                else:
                    correct_logits = last_logits[torch.arange(last_logits.shape[0]), tokenized_labels]
                    incorrect_logits = last_logits[torch.arange(last_logits.shape[0]), incorrect_labels]
                    tot_accuracy += (correct_logits > incorrect_logits).sum().item()
                    tot_tested += len(last_logits)
                    if reveal_bool_proportions: # TODO: generalise to other forms of True and False answers in the qm dataset
                        true_logits = last_logits[torch.arange(last_logits.shape[0]), 5852*torch.ones(last_logits.shape[0], dtype=torch.long, device=self.device)]
                        false_logits = last_logits[torch.arange(last_logits.shape[0]), 7700*torch.ones(last_logits.shape[0], dtype=torch.long, device=self.device)]
                        true_counter += (true_logits > false_logits).sum().item()
                        false_counter += (false_logits >= true_logits).sum().item()
        if reveal_bool_proportions:
            return tot_accuracy / tot_tested, true_counter / tot_tested, false_counter / tot_tested
        else:
            return tot_accuracy / tot_tested
    

    def get_logit_diff(self, model, use_test_data=True, use_alice_label=None, use_bob_label=None, n_iters=1):
        if use_alice_label is None:
            use_alice_label = self.use_alice_label
        if use_bob_label is None:
            use_bob_label = self.use_bob_label
        assert not (use_alice_label and use_bob_label), "use_alice_label and use_bob_label cannot both be True"
        tot_logit_diff = 0
        tot_tested = 0
        with torch.no_grad():
            for i in range(n_iters):
                batch = self.get_batch(train=not use_test_data)
                last_logits, tokenized_labels = self.get_logits_labels(model, batch, use_alice_label=use_alice_label, use_bob_label=use_bob_label)

                false_choices = self.tokenizer(batch['choices'][0], return_tensors='pt')['input_ids'][:, -1]
                true_choices = self.tokenizer(batch['choices'][1], return_tensors='pt')['input_ids'][:, -1]
                for i in range(len(last_logits)):
                    # label = batch['label'][i] if use_alice_label else batch['alice_label'][i] # TODO: check this
                    if use_bob_label:
                        label = batch['bob_label'][i]
                    if use_alice_label:
                        label = batch['alice_label'][i]
                    else:
                        label = batch['label'][i]
                    correct_choice = true_choices[i] if label == 1 else false_choices[i]
                    incorrect_choice = true_choices[i] if label == 0 else false_choices[i]

                    tot_logit_diff += last_logits[i][correct_choice] - last_logits[i][incorrect_choice]
                    tot_tested += 1
        return tot_logit_diff / tot_tested