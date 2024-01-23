from tasks.templates.TriviaTask import TriviaTask
from tasks.task import Task
import torch
import random
from torch.utils.data import Dataset, DataLoader
from tasks.inference_utils import get_final_logits
import json
import pickle
import numpy as np
from datasets import load_dataset

class BoolQTask(TriviaTask):
    def __init__(self, batch_size, tokenizer, device='cuda', chat_model=True, capitalize_question=False):
        train_dataset = load_dataset("boolq", split="train")
        test_dataset = load_dataset("boolq", split="validation")
        if capitalize_question:
            train_dataset = train_dataset.map(lambda x: {"question": x["question"].capitalize(), "passage": x["passage"], "answer": x["answer"]})
            test_dataset = test_dataset.map(lambda x: {"question": x["question"].capitalize(), "passage": x["passage"], "answer": x["answer"]})
        super().__init__(batch_size, tokenizer, device, chat_model=chat_model, question_style="tf", train_sentences=train_dataset, test_sentences=test_dataset)
