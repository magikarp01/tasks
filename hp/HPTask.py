from tasks.task import Task
import torch
import random
# Harry Potter Task
from torch.utils.data import Dataset, DataLoader
from tasks.inference_utils import get_final_logits
import json
import pickle
import numpy as np

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
sys_msg = "<s>" + B_SYS + """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Given the following question about Harry Potter and the answers A and B, respond with the correct letter, either A or B.""" + E_SYS

class HPTriviaTask(Task):
    """
    A class to ask the model trivia questions about Harry Potter.
    """
    def format_trivia(self, question_dict, chat_prompt=True, randomize_answers=False):

        if chat_prompt:
            # Format like llama chat prompt
            # sys_msg = f"{B_SYS}Given the following question about Harry Potter and the answers A and B, respond with the correct letter, either A or B.{E_SYS}"
            if randomize_answers:
                if random.random() < 0.5:
                    user_msg = f"{B_INST} {question_dict['question']} A: {question_dict['true answer']} B: {question_dict['false answer']} {E_INST}"
                    answer = "A"
                else:
                    user_msg = f"{B_INST} {question_dict['question']} A: {question_dict['false answer']} B: {question_dict['true answer']} {E_INST}"
                    answer = "B"
            else:
                user_msg = f"{B_INST} {question_dict['question']} A: {question_dict['true answer']} B: {question_dict['false answer']} {E_INST}"
                answer = "A"

            return {"prompt": sys_msg + user_msg + " Answer:", "answer": answer}

        else:
            if randomize_answers:
                if random.random() < 0.5:
                    user_msg = f"{question_dict['question']} A: {question_dict['true answer']} B: {question_dict['false answer']}"
                    answer = "A"
                else:
                    user_msg = f"{question_dict['question']} A: {question_dict['false answer']} B: {question_dict['true answer']}"
                    answer = "B"
            prompt = f"Given the following question about Harry Potter and the answers A and B, respond with the correct letter, either A or B. {user_msg} Answer:"
            return {"prompt": prompt, "answer": answer}


    def __init__(self, batch_size, tokenizer, device='cuda', chat_model=True, randomize_answers=True):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.criterion = torch.nn.CrossEntropyLoss()

        self.device = device
        self.chat_model = chat_model
        self.randomize_answers = randomize_answers

        # train test split
        # train_size = int(0.8 * len(HP_TRIVIA))
        # train_sentences = HP_TRIVIA[:train_size]
        # test_sentences = HP_TRIVIA[train_size:]
        with open("tasks/hp/data/hp_train.jsonl", "r") as f:
            train_sentences = f.readlines()
        # Convert each string to a dictionary
        train_sentences = [json.loads(item) for item in train_sentences]
        print(len(train_sentences))

        with open("tasks/hp/data/hp_test.jsonl", "r") as f:
            test_sentences = f.readlines()
        test_sentences = [json.loads(item) for item in test_sentences]
        print(len(test_sentences))
        
        train_prompts = [self.format_trivia(question_dict, chat_prompt=chat_model, randomize_answers=randomize_answers) for question_dict in train_sentences]
        test_prompts = [self.format_trivia(question_dict, chat_prompt=chat_model, randomize_answers=randomize_answers) for question_dict in test_sentences]
        self.train_loader = DataLoader(train_prompts, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_prompts, batch_size=batch_size, shuffle=True)
        self.train_iter = iter(self.train_loader)
        self.test_iter = iter(self.test_loader)

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
            for _ in range(n_iters):
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

        
from tasks.task import CompletionTask
class HPVerbatimTask(CompletionTask):
    def __init__(self, batch_size, tokenizer, device='cuda', num_completion_sentences=1):
        """
        A task asking model to autocomplete verbatim passages from Harry Potter.
        num_completion_sentences is the number of sentences (at the end) to complete in the passage. 
        """
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.criterion = torch.nn.CrossEntropyLoss(reduce=False)
        # self.criterion = torch.nn.CrossEntropyLoss()
        self.num_completion_sentences = num_completion_sentences

        self.device = device
        
        with open("tasks/hp/data/hp_verbatim_passages_train.pkl", "rb") as f:
            train_passages = pickle.load(f)
        with open("tasks/hp/data/hp_verbatim_passages_test.pkl", "rb") as f:
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

        self.train_data = []
        for passage in train_passages:
            prompt = " ".join(passage[:-num_completion_sentences])
            completion = " " + " ".join(passage[-num_completion_sentences:])
            self.train_data.append({"prompt": prompt, "completion": completion})
        
        self.test_data = []
        for passage in test_passages:
            prompt = " ".join(passage[:-num_completion_sentences])
            completion = " " + " ".join(passage[-num_completion_sentences:])
            self.test_data.append({"prompt": prompt, "completion": completion})
        
        self.train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_data, batch_size=batch_size, shuffle=True)

        self.train_iter = iter(self.train_loader)
        self.test_iter = iter(self.test_loader)

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
    