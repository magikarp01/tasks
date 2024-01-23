from tasks.task import Task
import torch
import random
# Harry Potter Task
from torch.utils.data import Dataset, DataLoader
from tasks.inference_utils import get_final_logits
import json
import pickle
import numpy as np

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# create the system message
# sys_msg = "<s>" + B_SYS + """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

# If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Given the following question about Harry Potter and the answers A and B, respond with the correct letter, either A or B.""" + E_SYS
trivia_msg = """You are a helpful, respectful and honest assistant. Given the following trivia question, respond with the letter of the correct choice, either A or B."""
tf_msg = """You are a helpful, respectful and honest assistant. Given the following true/false question, answer with either "true" or "false"."""

class TriviaTask(Task):
    """
    A class to ask a model trivia questions, with utilities for system prompts;
    """
    def set_sys_prompt(self):
        if self.question_style == "trivia":
            self.new_sys_msg = trivia_msg
        elif self.question_style == "tf":
            self.new_sys_msg = tf_msg
        else:
            raise ValueError(f"question_style {self.question_style} not recognized") 
           
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
            return {"prompt": self.new_sys_msg + " " + user_msg + " Answer:", "answer": answer}

        else:
            prompt = f"Given the following question and the answers A and B, respond with the correct letter, either A or B. {user_msg} Answer:"
            return {"prompt": prompt, "answer": answer}
    
    def format_tf(self, question_dict):
        user_msg = f"{question_dict['question']}"

        if self.chat_model:
            # Format like llama chat prompt
            # sys_msg = f"{B_SYS}Given the following question about Harry Potter and the answers A and B, respond with the correct letter, either A or B.{E_SYS}"
            user_msg = f"{self.B_INST} {user_msg} {self.E_INST}"
            return {"prompt": self.new_sys_msg + " " + user_msg + " Answer:", "answer": question_dict['answer']}

        else:
            prompt = f"""Given the following true/false question, answer with either "true" or "false". {user_msg} Answer:"""
            return {"prompt": prompt, "answer": question_dict['answer']}

    def _format_sys_prompt(self, prompt):
        return B_SYS + prompt + E_SYS

    def __init__(self, batch_size, tokenizer, device='cuda', chat_model=True, question_style="trivia", randomize_answers=True, shuffle=True, correct_answer_A=True, train_sentences=None, test_sentences=None):
        """
        question_style: "trivia" or "tf", whether the question answers are multiple choice or true/false
        randomize_answers, shuffle, correct_answer_A are only for trivia
        """
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.criterion = torch.nn.CrossEntropyLoss()

        self.device = device
        self.chat_model = chat_model
        self.question_style = question_style

        if chat_model:
            self.B_INST, self.E_INST = B_INST, E_INST
            self.B_SYS, self.E_SYS = B_SYS, E_SYS
            self.set_sys_prompt()

        self.randomize_answers = randomize_answers

        self.train_sentences = train_sentences
        self.test_sentences = test_sentences
        
        if question_style == "trivia":
            self.train_prompts = [self.format_trivia(question_dict, randomize_answers=randomize_answers, correct_answer_A=correct_answer_A) for question_dict in self.train_sentences]
            self.test_prompts = [self.format_trivia(question_dict, randomize_answers=randomize_answers, correct_answer_A=correct_answer_A) for question_dict in self.test_sentences]
        elif question_style == "tf":
            self.train_prompts = [self.format_tf(question_dict) for question_dict in self.train_sentences]
            self.test_prompts = [self.format_tf(question_dict) for question_dict in self.test_sentences]
        
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

                if self.question_style == "trivia":
                    a_token = self.tokenizer("A", return_tensors='pt').input_ids[:, -1].item()
                    b_token = self.tokenizer("B", return_tensors='pt').input_ids[:, -1].item()
                elif self.question_style == "tf":
                    a_token = self.tokenizer("true", return_tensors='pt').input_ids[:, -1].item()
                    b_token = self.tokenizer("false", return_tensors='pt').input_ids[:, -1].item()

                for i, logits in enumerate(last_logits):
                    assert len(logits.shape) == 1, logits.shape
                    correct_label = batch['answer'][i]
                    if self.question_style == "trivia":
                        if correct_label == "A":
                            correct_tokenized = a_token
                        else:
                            correct_tokenized = b_token
                    elif self.question_style == "tf":
                        if correct_label == "true":
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
                if self.question_style == "trivia":
                    a_token = self.tokenizer("A", return_tensors='pt').input_ids[:, -1].item()
                    b_token = self.tokenizer("B", return_tensors='pt').input_ids[:, -1].item()
                elif self.question_style == "tf":
                    a_token = self.tokenizer("true", return_tensors='pt').input_ids[:, -1].item()
                    b_token = self.tokenizer("false", return_tensors='pt').input_ids[:, -1].item()

                for i, logits in enumerate(last_logits):
                    tot_tested += 1
                    assert len(logits.shape) == 1, logits.shape
                    correct_label = batch['answer'][i]
                    if self.question_style == "trivia":
                        if correct_label == "A":
                            correct_tokenized = a_token
                        else:
                            correct_tokenized = b_token
                    elif self.question_style == "tf":
                        if correct_label == "true":
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