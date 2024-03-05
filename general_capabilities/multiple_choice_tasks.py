from transformers import AutoTokenizer, AutoModelForCausalLM
import datasets
import torch
import json
import pandas as pd
import openai
from dotenv import load_dotenv
import os
from datetime import datetime
from tasks.task import Task
from tasks.inference_utils import custom_generate
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import DataLoader

def number_to_letter(number):
    return chr(number + 65)

DEFAULT_QUESTION_FORMAT = """
{question}

Choices:
(A) {choice_A}
(B) {choice_B}
(C) {choice_C}
(D) {choice_D}
Answer:
("""

class MultipleChoiceQuestion(Task):
    """
    A class to run evaluations on any Multiple Choice QA task, such as MMLU

    """

    def _generate_sentence(
        self,
        model,
        tokenizer,
        strs: list,  # strs is now a list of strings
        with_logprobs=False,
        max_new_tokens=20,
        top_tokens=5,
        show_token_strs=True,
        temperature=0.7,
        include_input=False,
        device="cuda",
    ):
        # Encode all the inputs at once
        tokenizer.padding_side = "left"
        tokenized_inputs = tokenizer.batch_encode_plus(
            strs,
            return_tensors="pt",
            padding=True,
        )

        do_sample = True
        if temperature == 0.0:
            temperature = 1.0
            do_sample = False

        tokenized_inputs = {
            k: v.to(device) for k, v in tokenized_inputs.items()
        }  # Move to model's device

        outputs = model.generate(
            **tokenized_inputs,
            max_length=tokenized_inputs["input_ids"].shape[1] + max_new_tokens,
            temperature=temperature,
            top_k=top_tokens,
            return_dict_in_generate=True,
            output_scores=with_logprobs,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=do_sample,
        )

        sequences = outputs.sequences
        scores = outputs.scores if with_logprobs else None

        decoded_sentences = [
            tokenizer.decode(ids, skip_special_tokens=True) for ids in sequences
        ]
        if not include_input:
            decoded_sentences = [
                sentence[len(strs[i]) :] for i, sentence in enumerate(decoded_sentences)
            ]

        if with_logprobs and scores is not None:
            data = []
            for i, score in enumerate(scores):
                probs = torch.softmax(score, dim=-1)
                topk_probs, topk_tokens = probs.topk(top_tokens, dim=-1)

                for batch_index in range(topk_tokens.size(0)):
                    topk_strings = [
                        tokenizer.decode(token.item(), skip_special_tokens=True)
                        for token in topk_tokens[batch_index]
                    ]
                    row = {}
                    for j in range(top_tokens):
                        token_key = f"Token_{j+1}"
                        prob_key = f"Probability_{j+1}"
                        row[token_key] = (
                            topk_strings[j]
                            if show_token_strs
                            else topk_tokens[batch_index][j].item()
                        )
                        row[prob_key] = topk_probs[batch_index][j].item()
                    data.append(row)

            probs_df = pd.DataFrame(data)
            return decoded_sentences, probs_df
        else:
            return decoded_sentences


    def __init__(
        self,
        question_format=None,
    ):
        self.question_format = question_format
        self.dataset = None

    def get_accuracy(
        self,
        model,
        tokenizer,
        temperature=0.0,
        batch_size=5,
        n_batches=None,
        verbose=False,
        **kwargs,
    ):
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
        accuracy_total = 0
        n_total = 0

        if self.question_format is None:
            self.question_format = DEFAULT_QUESTION_FORMAT

        for i, batch in tqdm(enumerate(dataloader)):

            if n_batches is not None and i >= n_batches:
                break

            if verbose:
                print(f"Batch {i+1}/{n_batches}")

            questions = batch["question"]
            answers = batch["answer"]

            model_responses = self._generate_sentence(
                strs=questions,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=1,
                include_input=False,
                temperature=temperature,
                **kwargs,
            )

            accuracy_total += sum(
                [1 if model_response.strip() == answer.strip() else 0 for model_response, answer in zip(model_responses, answers)]
            )
            n_total += len(questions)
        
        accuracy = accuracy_total / n_total

        return accuracy



class MMLUTask(MultipleChoiceQuestion):

    def __init__(
        self,
        question_format=None,
        subject="all",
        streaming=True,
    ):
        super().__init__(question_format=question_format)

        available_subjects = [
            "abstract_algebra",
            "anatomy",
            "astronomy",
            "business_ethics",
            "clinical_knowledge",
            "college_biology",
            "college_chemistry",
            "college_computer_science",
            "college_mathematics",
            "college_medicine",
            "college_physics",
            "computer_security",
            "conceptual_physics",
            "econometrics",
            "electrical_engineering",
            "elementary_mathematics",
            "formal_logic",
            "global_facts",
            "high_school_biology",
            "high_school_chemistry",
            "high_school_computer_science",
            "high_school_european_history",
            "high_school_geography",
            "high_school_government_and_politics",
            "high_school_macroeconomics",
            "high_school_mathematics",
            "high_school_microeconomics",
            "high_school_physics",
            "high_school_psychology",
            "high_school_statistics",
            "high_school_us_history",
            "high_school_world_history",
            "human_aging",
            "human_sexuality",
            "international_law",
            "jurisprudence",
            "logical_fallacies",
            "machine_learning",
            "management",
            "marketing",
            "medical_genetics",
            "miscellaneous",
            "moral_disputes",
            "moral_scenarios",
            "nutrition",
            "philosophy",
            "prehistory",
            "professional_accounting",
            "professional_law",
            "professional_medicine",
            "professional_psychology",
            "public_relations",
            "security_studies",
            "sociology",
            "us_foreign_policy",
            "virology",
            "world_religions",
        ]

        if self.question_format is None:
            self.question_format = DEFAULT_QUESTION_FORMAT

        def mmlu_map_fn(examples):

            for i in range(len(examples["question"])):
                examples["question"][i] = self.question_format.format(
                    question=examples["question"][i],
                    choice_A=examples["choices"][i][0],
                    choice_B=examples["choices"][i][1],
                    choice_C=examples["choices"][i][2],
                    choice_D=examples["choices"][i][3],
                )
                examples["answer"][i] = number_to_letter(examples["answer"][i])
            return examples


        if subject == "all":
            dataset_list = []
            print("Loading MMLU dataset...")
            for subject in tqdm(available_subjects):
                dataset = datasets.load_dataset(
                    "tasksource/mmlu",
                    subject,
                    split="test",
                    streaming=streaming
                )
                dataset = dataset.map(
                    mmlu_map_fn,
                    batched=True,
                    remove_columns=set(dataset.column_names) - {"question", "answer"}
                )
                dataset_list.append(dataset)
            print("Concatenating datasets...")
            self.dataset = datasets.concatenate_datasets(dataset_list)
            self.dataset = self.dataset.shuffle(seed=42, buffer_size=10_000)

        else:
            self.dataset = datasets.load_dataset(
                "tasksource/mmlu",
                subject,
                split="test",
                streaming=streaming
            )
            self.dataset = self.dataset.map(
                mmlu_map_fn,
                batched=True,
                remove_columns=set(self.dataset.column_names) - {"question", "answer"}
            )
            self.dataset = self.dataset.shuffle(seed=42, buffer_size=10_000)
