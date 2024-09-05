import numpy as np
import pandas as pd
import random
import json
from tasks import Task
from datasets import load_dataset
import torch
# from torch.utils.data import Dataset, DataLoader
from datasets import Dataset
from tasks.inference_utils import get_final_logits
from collections import defaultdict
import time

# Function to convert numbers to words
def number_to_words(number):
    if number < 0:
        return "minus " + number_to_words(-number)
    words = []
    units = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    teens = ["", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
    tens = ["", "ten", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    thousands = ["", "thousand", "million", "billion", "trillion"]

    if number == 0:
        return "zero"

    num_str = str(number)
    num_len = len(num_str)

    # Iterate over the digits in reverse, building words
    for i in range(num_len):
        digit = int(num_str[num_len - i - 1])
        if i % 3 == 0:  # Handle units
            if i > 0 and digit != 0:
                words.append(thousands[i // 3])
            if digit != 0 or (i == 0 and len(words) == 0):
                words.append(units[digit])
        elif i % 3 == 1:  # Handle tens
            if digit == 1 and i > 0 and int(num_str[num_len - i]) != 0:  # Handle teens
                words.pop()
                words.append(teens[int(num_str[num_len - i])])
            else:
                words.append(tens[digit])
        elif i % 3 == 2:  # Handle hundreds
            if digit != 0:
                words.append("hundred")
                words.append(units[digit])

    return " ".join(reversed(words))

# Helper function to create plausible false answers
def generate_false_answer(true_answer, num_digits_off):
    """
    Given true_answer (int), false answer should be off on num_digits different digits.
    For each digit in true_answer, randomly choose a different digit to be the false answer, without replacement.
    """
    true_answer_str = str(true_answer)
    false_answer_list = list(true_answer_str)
    if num_digits_off == -1:
        num_digits_off = len(true_answer_str)
    digits_to_change = random.sample(range(len(true_answer_str)), num_digits_off)
    
    for index in digits_to_change:
        current_digit = int(false_answer_list[index])
        new_digit = random.choice([d for d in range(10) if d != current_digit])
        false_answer_list[index] = str(new_digit)
    
    false_answer = int(''.join(false_answer_list))
    return false_answer

question_formats = {
    "standard": "{operand1_str} + {operand2_str} = {answer_str}. Answer:",
    "vanilla": "{operand1_str} plus {operand2_str} equals {answer_str}. Answer:",
    "teacher": 'Teacher: "Is {operand1_str} + {operand2_str} = {answer_str}?"\n\nAnswer:',
    "ain't": "{operand1_str} + {operand2_str} = {answer_str} ain't",
    "does": 'Does {operand1_str} + {operand2_str} = {answer_str}? Answer:',
    "student": "Student:\n\n{operand1_str} + {operand2_str} = {answer_str}\n\nScore:",
    "exam": "Exam 1\n\nPart 1: True or False\n\n1. {operand1_str} + {operand2_str} = {answer_str}\nAnswer:",
}

# Helper function to create the question in different formats
def generate_question_format(question_format, operand1, operand2, answer, use_words=False):
    if use_words:
        operand1_str = number_to_words(operand1)
        operand2_str = number_to_words(operand2)
        answer_str = number_to_words(answer)
    else:
        operand1_str = str(operand1)
        operand2_str = str(operand2)
        answer_str = str(answer)

    question = question_format.format(operand1_str=operand1_str, operand2_str=operand2_str, answer_str=answer_str)
    return question

# Generate dataset
def generate_dataset(num_samples, num_digits_range=range(1, 5), question_formats=question_formats, num_digits_off=None):
    dataset = []
    for num_digits in num_digits_range:
        for _ in range(num_samples):  # generating 1000 samples for the dataset
            datapoint = {}
            operand1 = random.randint(1, 10**num_digits-1)
            operand2 = random.randint(1, 10**num_digits-1)
            answer = operand1 + operand2
            rand_num_digits_off = random.randint(1, len(str(answer))) if num_digits_off is None else num_digits_off
            wrong_answer = generate_false_answer(answer, num_digits_off=rand_num_digits_off)
            datapoint["operand1"] = operand1
            datapoint["operand2"] = operand2
            datapoint["answer"] = answer
            datapoint["wrong_answer"] = wrong_answer
            datapoint["num_digits"] = num_digits
            datapoint["num_digits_off"] = num_digits_off
            datapoint["operand1_str"] = number_to_words(operand1)
            datapoint["operand2_str"] = number_to_words(operand2)
            datapoint["answer_str"] = number_to_words(answer)
            datapoint["wrong_answer_str"] = number_to_words(wrong_answer)

            for question_true in [True, False]:
                for use_words in [True, False]:
                    for question_name, question_format in question_formats.items():
                        if question_name == "ain't":
                            question = generate_question_format(question_format, operand1, operand2, wrong_answer if question_true else answer, use_words=use_words)
                            datapoint[f"question_{question_name}_{'words' if use_words else 'digits'}_{'true' if question_true else 'false'}"] = question
                        else:
                            question = generate_question_format(question_format, operand1, operand2, answer if question_true else wrong_answer, use_words=use_words)
                            datapoint[f"question_{question_name}_{'words' if use_words else 'digits'}_{'true' if question_true else 'false'}"] = question
            dataset.append(datapoint)
    return pd.DataFrame(dataset)


def get_icl_prompt(dataset, template_name, use_words=False):
    """
    dataset needs to be formatted correctly
    """
    # dataset = generate_dataset(num_samples=n_shots, num_digits_range=[n_digits], question_formats={"template": template})
    # dataset = dataset.query("num_digits == @n_digits")
    icl_prompt = "Please help me check the truthfulness of answers to the following addition questions.\n\n"
    for i, row in dataset.iterrows():
        # randomly choose True or False
        question_true = random.choice([True, False])
        question = row[f"question_{template_name}_{'words' if use_words else 'digits'}_{'true' if question_true else 'false'}"]
        icl_prompt += f"{question} {'True' if question_true else 'False'}\n\n"
    return icl_prompt


class AdditionTask(Task):
    def __init__(self, batch_size, tokenizer, device='cuda', max_difficulty=4, n_shots=20, template_names="standard", use_words=False, load_dataset=True, num_digits_off=None, shuffle=True):
        """
        self.batch_size = batch_size
        """
        if isinstance(template_names, str):
            template_names = [template_names]
        self.template_names = template_names
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.device = device
        self.max_difficulty = max_difficulty

        if load_dataset:
            self.train_df = pd.read_parquet("tasks/qlm/data/addition_train_dataset.parquet").query(f"num_digits <= {max_difficulty}")
            self.test_df = pd.read_parquet("tasks/qlm/data/addition_test_dataset.parquet").query(f"num_digits <= {max_difficulty}")
            if num_digits_off is not None:
                self.train_df = self.train_df.query(f"num_digits_off == {num_digits_off}")
                self.test_df = self.test_df.query(f"num_digits_off == {num_digits_off}")
        else:
            self.train_df = generate_dataset(num_samples=1000, num_digits_range=range(1, max_difficulty+1), num_digits_off=num_digits_off)
            self.test_df = generate_dataset(num_samples=1000, num_digits_range=range(1, max_difficulty+1), num_digits_off=num_digits_off)
        # if constant_icl_prompts: 
        self.icl_df = pd.read_parquet("tasks/qlm/data/addition_icl_dataset.parquet")
        # associate each difficulty with an icl template
        self.icl_prompts = defaultdict(dict) # defaultdict of templatena
        for n_digits in range(1, max_difficulty+1):
            for template_name in template_names:
                self.icl_prompts[template_name][n_digits] = get_icl_prompt(self.icl_df.query("num_digits == @n_digits").head(n_shots), template_name, use_words=use_words) 
        def apply_icl_prompt(row):
            return self.icl_prompts[row["template_name"]][row["num_digits"]] + row["question"]
            
        train_dataset = []
        test_dataset = []
        for template_name in template_names:
            for question_true in [True, False]:
                column_names = ["operand1", "operand2", "answer", "wrong_answer", "operand1_str", "operand2_str", "answer_str", "wrong_answer_str", "num_digits", f"question_{template_name}_{'words' if use_words else 'digits'}_{'true' if question_true else 'false'}"]
                train_df_slice = self.train_df[column_names].copy().rename(columns={f"question_{template_name}_{'words' if use_words else 'digits'}_{'true' if question_true else 'false'}" : "question"})

                train_df_slice["label"] = question_true
                train_df_slice["label_str"] = " True" if question_true else " False"
                train_df_slice["template_name"] = template_name
                train_df_slice["full_prompt"] = train_df_slice.apply(apply_icl_prompt, axis=1)
                train_dataset.append(train_df_slice)

                test_df_slice = self.test_df[column_names].copy().rename(columns={f"question_{template_name}_{'words' if use_words else 'digits'}_{'true' if question_true else 'false'}" : "question"})
                test_df_slice["label"] = question_true
                test_df_slice["label_str"] = " True" if question_true else " False"
                test_df_slice["template_name"] = template_name
                test_df_slice["full_prompt"] = test_df_slice.apply(apply_icl_prompt, axis=1)
                test_dataset.append(test_df_slice)

        # self.train_dataset = Dataset.from_pandas(pd.concat(train_dataset))
        # self.test_dataset = Dataset.from_pandas(pd.concat(test_dataset))
        self.train_dataset = Dataset.from_pandas(pd.concat(train_dataset).reset_index(drop=True))
        self.test_dataset = Dataset.from_pandas(pd.concat(test_dataset).reset_index(drop=True))
        self.set_loaders(self.train_dataset, self.test_dataset, shuffle=shuffle)

        self.criterion = torch.nn.CrossEntropyLoss()
    
    def calculate_loss(self, model, batch):
        last_logits = get_final_logits(model, self.tokenizer, batch["full_prompt"], device=self.device)
        correct_labels = self.tokenizer(batch["label_str"], return_tensors="pt", padding=True).input_ids[:, -1].to(self.device)
        return self.criterion(last_logits, correct_labels)

    def get_test_accuracy(self, model, use_test_data=True, check_all_logits=False, continuous=True, n_iters=1):
        possible_labels = self.tokenizer([" True", " False"], return_tensors="pt", padding=True).input_ids[:, -1].to(self.device)
        with torch.no_grad():
            accs = []
            for i in range(n_iters):
                batch = self.get_batch(train=not use_test_data)
                # time this
                start_time = time.time()
                last_logits = get_final_logits(model, self.tokenizer, batch["full_prompt"], device=self.device)
                end_time = time.time()

                correct_labels = self.tokenizer(batch["label_str"], return_tensors="pt", padding=True).input_ids[:, -1].to(self.device)
                if not check_all_logits:
                    correct_labels = torch.where(correct_labels == possible_labels[0], 0, 1)

                if continuous:
                    probabilities = torch.softmax(last_logits, dim=1)
                    if check_all_logits:
                        accs.append(probabilities[torch.arange(probabilities.shape[0]), correct_labels].mean().item())
                    else:
                        probabilities = probabilities[:, possible_labels]
                        # renormalise
                        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
                        # labels have different index
                        accs.append(probabilities[torch.arange(probabilities.shape[0]), correct_labels].mean().item())

                else:
                    if check_all_logits:
                        accs.append((torch.argmax(last_logits, dim=1) == correct_labels).sum().item() / len(correct_labels))
                    else:
                        last_logits = last_logits[:, possible_labels]
                        accs.append((torch.argmax(last_logits, dim=1) == correct_labels).sum().item() / len(correct_labels))
            return sum(accs) / len(accs)

