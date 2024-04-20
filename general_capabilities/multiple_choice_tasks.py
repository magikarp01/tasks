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
from tasks.general_capabilities.templates import *

def number_to_letter(number):
    return chr(number + 65)


class MultipleChoiceQuestion(Task):
    """
    A class to run evaluations on any Multiple Choice QA task, such as MMLU

    """

    def __init__(
        self,
        question_format=None,
    ):
        self.question_format = question_format
        self.dataset = None
        self.max_new_tokens = 1

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
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.pad_token
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

    def _fsdp_generate_sentence(
        self,
        model,
        tokenizer,
        strs: list,  # strs is now a list of strings
        max_new_tokens=20,
        top_tokens=5,
        temperature=0.7,
        include_input=False,
        device="auto",
        ):



        # assert isinstance(model, torch.distributed.fsdp.FullyShardedDataParallel)
        def fsdp_generate(model, tokenized_inputs, temperature=0, top_k=50, max_new_tokens=50):

            def top_k_sampling_with_temperature(logits, k, temperature):
                assert temperature != 0
                scaled_logits = logits / temperature
                top_k_values, top_k_indices = torch.topk(scaled_logits[:, :], k, dim=-1)
                probabilities = torch.nn.functional.softmax(top_k_values, dim=-1)
                sampled_indices = torch.multinomial(probabilities, 1)
                sampled_token_ids = top_k_indices.gather(-1, sampled_indices)
                return sampled_token_ids
                
            def greedy_search_step(model, input_ids, temperature, top_k):
                with torch.no_grad():
                    outputs = model(input_ids)
                next_token_logits = outputs.logits[:, -1, :]  # Get the logits for the last token in the sequence
                if temperature == 0:
                    next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)  # Greedily select the token with the highest probability
                else:
                    next_token_id = top_k_sampling_with_temperature(next_token_logits, temperature=temperature, k=top_k)
                return next_token_id

            input_ids = tokenized_inputs['input_ids']
            generated_ids = [input_ids]
            for _ in range(max_new_tokens): 
                next_token_id = greedy_search_step(model, generated_ids[-1], temperature=temperature, top_k=top_k)
                generated_ids.append(next_token_id)
                
            generated_ids = torch.cat(generated_ids, dim=1)
            return generated_ids

        # Encode all the inputs at once
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.pad_token
        tokenizer.padding_side = "left"
        tokenized_inputs = tokenizer.batch_encode_plus(
            strs,
            return_tensors="pt",
            padding=True,
        )

        breakpoint()
        out = fsdp_generate(model, tokenized_inputs, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_tokens)
        breakpoint()
        decoded_sentences = tokenizer.batch_decode(out, skip_special_tokens=True)
        if not include_input:
            decoded_sentences = [
                sentence[len(strs[i]):] for i, sentence in enumerate(decoded_sentences)
            ]
        return decoded_sentences
    


    def get_accuracy(
        self,
        model,
        tokenizer,
        temperature=0.0,
        batch_size=25,
        n_batches=None,
        verbose=False,
        fsdp=False,
        **kwargs,
    ):
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
        accuracy_total = 0
        n_total = 0

        for i, batch in tqdm(enumerate(dataloader)):

            if n_batches is not None and i >= n_batches:
                break

            if verbose:
                print(f"Batch {i+1}/{n_batches}")

            questions = batch["question"]
            answers = batch["answer"]

            if fsdp:
                breakpoint()
                model_responses = self._fsdp_generate_sentence(
                    model=model,
                    tokenizer=tokenizer,
                    strs=questions,
                    max_new_tokens=self.max_new_tokens,
                    include_input=False,
                    temperature=temperature,
                )
            else:
                model_responses = self._generate_sentence(
                    strs=questions,
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=self.max_new_tokens,
                    include_input=False,
                    temperature=temperature,
                    **kwargs,
                )

            accuracy_total += sum(
                [1 if answer.strip() in model_response.strip() else 0 for model_response, answer in zip(model_responses, answers)]
            )
            n_total += len(questions)
        
        accuracy = accuracy_total / n_total

        return accuracy



class MMLUTask(MultipleChoiceQuestion):

    def __init__(
        self,
        question_format=None,
        subject="all",
        streaming=False,
        tiny=True,
    ):
        """
        Defaults to tiny MMLU dataset https://huggingface.co/tinyBenchmarks
        """
        super().__init__(question_format=question_format)

        dataset_name = "tasksource/mmlu" if not tiny else "tinyBenchmarks/tinyMMLU"

        if not streaming and not tiny:
            raise ValueError("Loading the full MMLU dataset, for speed use streaming=True or tiny=True.")

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
            self.question_format = DEFAULT_4_QUESTION_FORMAT

        def mmlu_map_fn(examples):

            questions = []
            answers = []

            for i in range(len(examples["question"])):
                questions.append(self.question_format.format(
                    question=examples["question"][i],
                    choice_A=examples["choices"][i][0],
                    choice_B=examples["choices"][i][1],
                    choice_C=examples["choices"][i][2],
                    choice_D=examples["choices"][i][3],
                ))
                answers.append(number_to_letter(int(examples["answer"][i])))

            return {
                "question": questions,
                "temp_answer": answers,
            }


        if subject == "all" and not tiny:
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
                    remove_columns=set(dataset.column_names) - {"question", "temp_answer"}
                ).rename_column("temp_answer", "answer")
                dataset_list.append(dataset)
            print("Concatenating datasets...")
            self.dataset = datasets.concatenate_datasets(dataset_list)
            self.dataset = self.dataset.shuffle(seed=42, buffer_size=10_000)

        else:
            self.dataset = datasets.load_dataset(
                dataset_name,
                subject,
                split="test",
                streaming=streaming
            )
            self.dataset = self.dataset.map(
                mmlu_map_fn,
                batched=True,
                remove_columns=set(self.dataset.column_names) - {"question", "temp_answer"}
            ).rename_column("temp_answer", "answer")
            self.dataset = self.dataset.shuffle(seed=42)


class HellaSwagTask(MultipleChoiceQuestion):

    def __init__(
        self,
        question_format=None,
        streaming=False,
        tiny=True,
    ):
        """
        Defaults to tiny HellaSwag dataset https://huggingface.co/tinyBenchmarks
        """

        super().__init__(question_format=question_format)

        dataset_name = "Rowan/hellaswag" if not tiny else "tinyBenchmarks/tinyHellaswag"

        if not streaming and not tiny:
            raise ValueError("Loading the full HellaSwag dataset, for speed use streaming=True or tiny=True.")

        if self.question_format is None:
            self.question_format = DEFAULT_HELLASWAG_QUESTION_FORMAT

        def hellaswag_map_fn(examples):
            questions = []
            answers = []
            for i in range(len(examples["ctx"])):
                    
                question = self.question_format.format(
                    question=examples["ctx"][i],
                    choice_A=examples["endings"][i][0],
                    choice_B=examples["endings"][i][1],
                    choice_C=examples["endings"][i][2],
                    choice_D=examples["endings"][i][3],
                )
                questions.append(question)
                answers.append(number_to_letter(int(examples["label"][i])))

            return {
                "question": questions,
                "answer": answers
            }

        self.dataset = datasets.load_dataset(
            dataset_name,
            split="validation",
            streaming=streaming
        )
        self.dataset = self.dataset.map(
            hellaswag_map_fn,
            batched=True,
            remove_columns=set(self.dataset.column_names) - {"question", "answer"}
        )
        self.dataset = self.dataset.shuffle(seed=42)




class WinograndeTask(MultipleChoiceQuestion):


    def __init__(
        self,
        question_format=None,
        streaming=False,
        tiny=True,
    ):
        """
        Defaults to tiny Winogrande dataset https://huggingface.co/tinyBenchmarks
        """
        super().__init__(question_format=question_format)

        dataset_name = "winogrande" if not tiny else 'tinyBenchmarks/tinyWinogrande'

        if not streaming and not tiny:
            raise ValueError("Loading the full WinoGrande dataset, for speed use streaming=True or tiny=True.")

        if self.question_format is None:
            self.question_format = DEFAULT_WINOGRANDE_QUESTION_FORMAT
            
        def winogrande_map_fn(examples):
            questions = []
            answers = []
            for i in range(len(examples["sentence"])):
                question = self.question_format.format(
                    question=examples["sentence"][i],
                    choice_A=examples["option1"][i],
                    choice_B=examples["option2"][i],
                )
                questions.append(question)
                answers.append(number_to_letter(int(examples["answer"][i])))
            return {
                "question": questions,
                "answer": answers
            }
        
        self.dataset = datasets.load_dataset(
            dataset_name, 
            "winogrande_xl",
            streaming=streaming,
            split="validation",
        )
        self.dataset = self.dataset.map(
            winogrande_map_fn,
            batched=True,
            remove_columns=set(self.dataset.column_names) - {"question", "answer"}
        )
        self.dataset = self.dataset.shuffle(seed=42)


class SciQTask(MultipleChoiceQuestion):

    def __init__(
        self,
        question_format=None,
        streaming=True,
    ):
        super().__init__(question_format=question_format)

        if self.question_format is None:
            self.question_format = DEFAULT_4_QUESTION_FORMAT
            
        def sciq_map_fn(examples):
            questions = []
            answers = []
            for i in range(len(examples["question"])):
 
                choices = (
                    examples["correct_answer"][i],
                    examples["distractor1"][i], 
                    examples["distractor2"][i], 
                    examples["distractor3"][i]
                )

                question = self.question_format.format(
                    question=examples["question"][i],
                    choice_A=choices[i%4],
                    choice_B=choices[(i+1)%4],
                    choice_C=choices[(i+2)%4],
                    choice_D=choices[(i+3)%4],
                )
                questions.append(question)
                answers.append(number_to_letter(3-(i-1)%4))
            return {
                "question": questions,
                "answer": answers
            }
        
        self.dataset = datasets.load_dataset(
            "allenai/sciq",
            streaming=streaming,
            split="test",
        )
        self.dataset = self.dataset.map(
            sciq_map_fn,
            batched=True,
            remove_columns=set(self.dataset.column_names) - {"question", "answer"}
        )
        self.dataset = self.dataset.shuffle(seed=42)



class LambadaTask(MultipleChoiceQuestion):

    def __init__(
        self,
        question_format=None,
        streaming=True,
        max_new_tokens=3,
    ):
        super().__init__(question_format=question_format)
        self.max_new_tokens = max_new_tokens

        if self.question_format is None:
            self.question_format = DEFAULT_LAMBADA_QUESTION_FORMAT
        
        def lambada_map_fn(examples):

            questions = []
            answers = []
            for i in range(len(examples["text"])):
                text = self.question_format.format(
                    question=examples["text"][i]
                ).strip()
                answer = text.split(" ")[-1]
                question = " ".join(text.split(" ")[:-1])
                questions.append(question)
                answers.append(answer)
            return {
                "question": questions,
                "answer": answers
            }
        
        self.dataset = datasets.load_dataset(
            "lambada",
            streaming=streaming,
            split="test",
        )
        self.dataset = self.dataset.map(
            lambada_map_fn,
            batched=True,
            remove_columns=set(self.dataset.column_names) - {"question", "answer"}
        )
        self.dataset = self.dataset.shuffle(seed=42)


class PIQATask(MultipleChoiceQuestion):

    def __init__(
        self,
        question_format=None,
        streaming=True,
    ):
        super().__init__(question_format=question_format)

        if self.question_format is None:
            self.question_format = DEFAULT_2_QUESTION_FORMAT

        def piqa_map_fn(examples):
            questions = []
            answers = []
            for i in range(len(examples["goal"])):
                question = self.question_format.format(
                    question=examples["goal"][i],
                    choice_A=examples["sol1"][i],
                    choice_B=examples["sol2"][i],
                )
                questions.append(question)
                answers.append(number_to_letter(int(examples["label"][i])))
            return {
                "question": questions,
                "answer": answers
            }

        self.dataset = datasets.load_dataset(
            "piqa",
            streaming=streaming,
            split="validation",
        )
        self.dataset = self.dataset.map(
            piqa_map_fn,
            batched=True,
            remove_columns=set(self.dataset.column_names) - {"question", "answer"}
        )
        self.dataset = self.dataset.shuffle(seed=42)


class WMDPTask(MultipleChoiceQuestion):

    def __init__(
        self,
        dataset_name: str,
        question_format=None,
        streaming=True,
    ):
        """
        Built default template for quwstion evaluations into tasks dataset.
        Do not pass in question format unless needs to be different to CUT paper.

        Args:
            dataset_name: Which WMDP dataset you are using.
        """
        super().__init__(question_format=question_format)

        if self.question_format is None:
            self.question_format = DEFAULT_WMDP_QUESTION_FORMAT

        self.name_dict = {"bio": "biology", "cyber": "cybersecurity", "chem": "chemistry"}
        assert dataset_name in self.name_dict.keys(), "Dataset name must be one of 'bio', 'cyber', 'chem'."

        def wmdp_map_fn(examples):
            questions = []
            answers = []

            for i in range(len(examples["question"])):
                question = self.question_format.format(
                    topic=self.name_dict.get(dataset_name),
                    question=examples["question"][i],
                    a1=examples["choices"][i][0],
                    a2=examples["choices"][i][1],
                    a3=examples["choices"][i][2],
                    a4=examples["choices"][i][3],
                )
            questions.append(question)
            answers.append(number_to_letter(int(examples["answer"][i])))
            return {
                "question": questions,
                "answer": answers
            }

        self.dataset = datasets.load_dataset(
            "cais/wmdp",
            name=f"wmdp-{dataset_name}",
            streaming=streaming,
            split="test",
        )
        self.dataset = self.dataset.map(
            wmdp_map_fn,
            batched=True,
            remove_columns=set(self.dataset.column_names) - {"question", "answer"}
        )
        self.dataset = self.dataset.shuffle(seed=42)
