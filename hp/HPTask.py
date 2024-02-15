from tasks.task import Task
import torch
import random
# Harry Potter Task
from torch.utils.data import Dataset, DataLoader
from tasks.inference_utils import get_final_logits
import json
import pickle
import numpy as np
import tqdm

# Factual knowledge: want to test if model can do single-token completions correctly, and if model can answer True/False questions about Harry Potter correctly
# Verbatim knowledge: give direct passages from book, ask model to complete direct passage
HP_UNLRN_DATA = [ # Completions from or in the style of the 
    {"prompt": "Feeling stranded in the middle of London, he raised his wand, hoping to catch the attention of the special transport for wizards in need, known as the", "correct": "Knight Bus", "incorrect": "Magic Express",},
    {"prompt": "In his third year, they were required to buy a particular textbook for Care of Magical Creatures, a book that was notorious for", "correct": "being one of the most difficult and complex classes at Hogwarts.", "incorrect": "being incredibly dry and boring."},
] # 

# hp_verbatim = [ # Passages from the text
#     "",

# ]

# "Given the following passage, which answer comes next, A or B? {prompt} A: {correct} B: {incorrect}"

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# create the system message
# sys_msg = "<s>" + B_SYS + """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

# If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Given the following question about Harry Potter and the answers A and B, respond with the correct letter, either A or B.""" + E_SYS
sys_msg = """You are a helpful, respectful and honest assistant. Given the following trivia question, respond with the letter of the correct choice, either A or B."""

class HPTriviaTask(Task):
    """
    A class to ask the model trivia questions about Harry Potter.
    """
    def format_trivia(self, question_dict, correct_answer_A=True, randomize_answers=False, few_shot=False):
        """
        randomize_answers takes precedence over correct_answer_A
        """
        if randomize_answers:
            correct_answer_A = random.random() < 0.5
        if correct_answer_A:
            user_msg = f"{question_dict['question']} A: {question_dict['true_answer']}. B: {question_dict['false_answer']}."
            answer = "A"
        else:
            user_msg = f"{question_dict['question']} A: {question_dict['false_answer']}. B: {question_dict['true_answer']}."
            answer = "B"

        if self.chat_model:
            # Format like llama chat prompt
            # sys_msg = f"{B_SYS}Given the following question about Harry Potter and the answers A and B, respond with the correct letter, either A or B.{E_SYS}"
            user_msg = f"{self.B_INST} {user_msg} {self.E_INST}"
            return {"prompt": self._format_sys_prompt(self.sys_msg) + " " + user_msg + " Answer:", "answer": answer}

        else:
            prompt = f"Given the following question and the answers A and B, respond with the correct letter, either A or B. {user_msg} Answer:"
            return {"prompt": prompt, "answer": answer}

    def _format_sys_prompt(self, prompt):
        return B_SYS + prompt + E_SYS

    def __init__(self, batch_size, tokenizer, device='cuda', chat_model=True, randomize_answers=True, shuffle=True, correct_answer_A=True, train_data_location="tasks/hp/data/hp_trivia_train.jsonl", test_data_location="tasks/hp/data/hp_trivia_test.jsonl", sys_msg=sys_msg, seed=None, same_location=None, train_n=None):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.criterion = torch.nn.CrossEntropyLoss()

        self.device = device
        self.chat_model = chat_model
        if chat_model:
            self.B_INST, self.E_INST = B_INST, E_INST
            self.B_SYS, self.E_SYS = B_SYS, E_SYS
            self.sys_msg = sys_msg

        self.randomize_answers = randomize_answers

        # train test split
        # train_size = int(0.8 * len(HP_TRIVIA))
        # train_sentences = HP_TRIVIA[:train_size]
        # test_sentences = HP_TRIVIA[train_size:]
        if train_n is None or same_location is None:
            with open(train_data_location, "r") as f:
                train_sentences = f.readlines()
            # Convert each string to a dictionary
            self.train_sentences = [json.loads(item) for item in train_sentences]
            
            with open(test_data_location, "r") as f:
                test_sentences = f.readlines()
            self.test_sentences = [json.loads(item) for item in test_sentences]
        else:
            # train and 
            with open(same_location, "r") as f:
                sentences = f.readlines()
            # Convert each string to a dictionary
            sentences = [json.loads(item) for item in sentences]
            self.train_sentences = sentences[:train_n]
            self.test_sentences = sentences[train_n:]
        
                    
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.train_prompts = [self.format_trivia(question_dict, randomize_answers=randomize_answers, correct_answer_A=correct_answer_A) for question_dict in self.train_sentences]
        self.test_prompts = [self.format_trivia(question_dict, randomize_answers=randomize_answers, correct_answer_A=correct_answer_A) for question_dict in self.test_sentences]
        self.set_loaders(self.train_prompts, self.test_prompts, shuffle=shuffle)


    def calculate_loss(self, model, batch):
        last_logits = get_final_logits(model, self.tokenizer, batch['prompt'], device=self.device)
        labels = batch['answer']
        tokenized_labels = self.tokenizer(labels, return_tensors='pt').input_ids[:, -1]

        return self.criterion(last_logits, tokenized_labels.to(self.device))
    
    def get_logit_diff(self, model, use_test_data=True, n_iters=1):
        logit_diffs = []
        with torch.no_grad():
            for _ in range(n_iters):
                batch = self.get_batch(train=not use_test_data)
                last_logits = get_final_logits(model, self.tokenizer, batch['prompt'], device=self.device)
                a_token = self.tokenizer("A", return_tensors='pt').input_ids[:, -1].item()
                b_token = self.tokenizer("B", return_tensors='pt').input_ids[:, -1].item()

                for i, logits in enumerate(last_logits):
                    assert len(logits.shape) == 1, logits.shape
                    correct_label = batch['answer'][i]
                    if correct_label == "A":
                        correct_tokenized = a_token
                    else:
                        correct_tokenized = b_token
                    
                    incorrect_tokenized = b_token if correct_tokenized == a_token else a_token
                    # check if correct tokenized has higher logit than incorrect tokenized
                    logit_diffs.append((logits[correct_tokenized] - logits[incorrect_tokenized]).item())
        return logit_diffs
    
    def get_test_accuracy(self, model, use_test_data=True, check_all_logits=False, n_iters=1):
        with torch.no_grad():
            tot_correct = 0
            tot_tested = 0
            for _ in tqdm.tqdm(range(n_iters)):
                batch = self.get_batch(train=not use_test_data)
                last_logits = get_final_logits(model, self.tokenizer, batch['prompt'], device=self.device)
                a_token = self.tokenizer("A", return_tensors='pt').input_ids[:, -1].item()
                b_token = self.tokenizer("B", return_tensors='pt').input_ids[:, -1].item()

                for i, logits in enumerate(last_logits):
                    tot_tested += 1
                    assert len(logits.shape) == 1, logits.shape
                    correct_label = batch['answer'][i]
                    if correct_label == "A":
                        correct_tokenized = a_token
                    else:
                        correct_tokenized = b_token
                    
                    if check_all_logits:
                        # check if argmax of logits is the correct otken
                        if torch.argmax(logits) == correct_tokenized:
                            tot_correct += 1
                    else:
                        incorrect_tokenized = b_token if correct_tokenized == a_token else a_token
                        # check if correct tokenized has higher logit than incorrect tokenized
                        if logits[correct_tokenized] > logits[incorrect_tokenized]:
                            tot_correct += 1   

        return tot_correct / tot_tested

        
from tasks.templates.CompletionTask import CompletionTask
import Levenshtein
class HPVerbatimTask(CompletionTask):
    def format_completion(self, passage, num_completion_sentences=None):
        if num_completion_sentences is None:
            num_completion_sentences = self.num_completion_sentences
        prompt = " ".join(passage[:-num_completion_sentences])
        completion = " " + " ".join(passage[-num_completion_sentences:])
        return {"prompt": prompt, "completion": completion}

    def _format_sys_prompt(self, prompt):
        return B_SYS + prompt + E_SYS

    def __init__(self, batch_size, tokenizer, device='cuda', num_completion_sentences=1, shuffle=True, 
                 criterion="cross_entropy",
                 train_data_location="tasks/hp/data/hp_verbatim_passages_train.pkl", test_data_location="tasks/hp/data/hp_verbatim_passages_test.pkl", verbose=False):
        """
        A task asking model to autocomplete verbatim passages from Harry Potter.
        num_completion_sentences is the number of sentences (at the end) to complete in the passage. 
        criterion is loss function for evaluating the model's ability to complete the passage. Can be cross_entropy, levenshtein, or accuracy.
        """
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.verbose = verbose
        if criterion == "cross_entropy":
            self.criterion = torch.nn.CrossEntropyLoss(reduce=False)
        elif criterion == "levenshtein":
            # hamming distance of highest logit label to correct label
            def levenshtein_loss(logits, labels): # logits is shape (seq, vocab), label is shape (seq)
                max_logits = torch.argmax(logits, dim=1)
                str_tokens = self.tokenizer.batch_decode(max_logits)
                str_labels = self.tokenizer.batch_decode(labels)
                losses = torch.zeros_like(labels, dtype=logits.dtype)
                for i, (token, label) in enumerate(zip(str_tokens, str_labels)):
                    losses[i] = Levenshtein.distance(token, label)
                return losses
            self.criterion = levenshtein_loss

        elif criterion == "accuracy":
            def accuracy(logits, labels):
                max_logits = torch.argmax(logits, dim=1)
                losses = torch.zeros_like(labels, dtype=logits.dtype)
                for i, (token, label) in enumerate(zip(max_logits, labels)):
                    losses[i] = token == label
                return losses
            self.criterion = accuracy

        # self.criterion = torch.nn.CrossEntropyLoss()
        self.num_completion_sentences = num_completion_sentences

        self.device = device
        
        with open(train_data_location, "rb") as f:
            train_passages = pickle.load(f)
        with open(test_data_location, "rb") as f:
            test_passages = pickle.load(f)
        """
        Passage looks like:
        ['“Scrimgeour wanted to know where you go when you’re not at Hogwarts,” said Harry, still looking fixedly at his knees.',
        '“Yes, he is very nosy about that,” said Dumbledore, now sounding cheerful, and Harry thought it safe to look up again.',
        '“He has even attempted to have me followed.',
        'Amusing, really.',
        'He set Dawlish to tail me.'
        ]"""
        assert num_completion_sentences <= len(train_passages[0]), f"num_completion_sentences must be <= {len(train_passages[0])}"

        self.train_data = [self.format_completion(passage, num_completion_sentences) for passage in train_passages]
        self.test_data = [self.format_completion(passage, num_completion_sentences) for passage in test_passages]
        
        self.set_loaders(self.train_data, self.test_data, shuffle=shuffle)

    def tokenize_batch(self, batch):
        prompt_tokens = self.tokenizer(batch['prompt']).input_ids
        completion_tokens = self.tokenizer(batch['completion']).input_ids
        
        completion_tokens_formatted = []
        for completion_tokenized in completion_tokens: # remove start token
            if completion_tokenized[0] == self.tokenizer.bos_token_id:
                completion_tokenized = completion_tokenized[1:]
            completion_tokens_formatted.append(completion_tokenized)
        
        return prompt_tokens, completion_tokens_formatted

    def calculate_loss(self, model, batch):
        prompt_tokens, completion_tokens = self.tokenize_batch(batch)
        losses = self.evaluate_completion(model, prompt_tokens, completion_tokens)
        # list of lists, get mean of each list and return mean of means
        mean_losses = [torch.mean(loss) for loss in losses]
        return torch.mean(torch.stack(mean_losses))
    