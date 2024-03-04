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

# import asyncio
# import aiohttp

try:
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    # client = openai.Client(
    #     organization='org-X6T6Ar6geRtOrQgQTQS3OUpw',
    # )
    client = openai.Client()
except:
    print("OpenAI API key not found, will not be able to run evaluations on HPSAQ Task")
    client = None

DEFAULT_EVAL_MESSAGE = """
You are evaluating the accuracy of the model's response to the question. Please read the response and the perfect answer, then respond with either 'Y' or 'N' to indicate if the model's response is correct. 

Question:
{question}

Model's response:
{model_response}

Perfect answer:
{answer}

If the model's response is correct, please respond with 'Y', otherwise respond with 'N'.
"""


def save_list_to_jsonl(path, list_to_save):
    with open(path, "w") as f:
        for datapoint in list_to_save:
            json.dump(datapoint, f)
            f.write("\n")


class ShortAnswerQuestion(Task):
    """
    A class to run evaluations on any Short Answer Question task, such as TriviaQA

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

    def _get_qa_model_grades_threaded(
        self,
        client,
        questions,
        model_responses,
        answers,
        openai_model="gpt-4-turbo-preview",
        max_tokens=1,
        max_threads=5,
        seed=42,
        eval_message=None,
        logit_bias=None,
    ):

        print("hello")
        assert (
            len(questions) == len(model_responses) == len(answers)
        ), "Length of questions, model_responses, and answers should be the same"

        def get_model_grade_internal(
            question, model_response, answer, logit_bias, eval_message=eval_message
        ):
            if eval_message is None:
                eval_message = DEFAULT_EVAL_MESSAGE

            user_message = eval_message.format(
                question=question,
                model_response=model_response,
                answer=answer,
            )

            if logit_bias is None:
                logit_bias = {}

            gpt_answer = (
                client.chat.completions.create(
                    model=openai_model,
                    messages=[
                        {"role": "user", "content": user_message},
                    ],
                    temperature=0,
                    seed=seed,
                    max_tokens=max_tokens,
                    logit_bias=logit_bias,
                )
                .choices[0]
                .message.content
            )

            return gpt_answer

        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            results = list(
                executor.map(
                    get_model_grade_internal,
                    questions,
                    model_responses,
                    answers,
                    [logit_bias] * len(questions),
                )
            )

        return results

    def __init__(
        self,
        eval_message=None,
    ):
        """
        Eval message must contain the following fields:
        {question}
        {model_response}
        {answer}

        Dataset must be a list of dictionaries with the following keys:
        "question": str
        "answer": str
        """
        self.eval_message = eval_message
        self.dataset = None
        self.client = client

    def get_accuracy(
        self,
        model,
        tokenizer,
        eval_onthe_fly=True,
        eval_model="gpt-4-turbo-preview",
        batch_size=5,
        n_batches=None,
        verbose=False,
        max_new_tokens=20,
        max_eval_tokens=1,
        max_threads=5,
        **kwargs,
    ):
        # TODO: allow for llama evals
        self.answered_dataset = []
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)

        for i, batch in enumerate(dataloader):

            if n_batches is not None and i >= n_batches:
                break

            questions = batch["question"]
            answers = batch["answer"]
            model_responses = self._generate_sentence(
                strs=questions,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                include_input=False,
                **kwargs,
            )

            if eval_onthe_fly:
                model_grades = self._get_qa_model_grades_threaded(
                    client=self.client,
                    questions=questions,
                    model_responses=model_responses,
                    answers=answers,
                    openai_model=eval_model,
                    max_tokens=max_eval_tokens,
                    max_threads=max_threads,
                    **kwargs,
                )

                self.answered_dataset.extend(
                    [
                        {
                            "question": questions[i],
                            "model_response": model_responses[i],
                            "answers": answers[i],
                            "model_grade": model_grades[i],
                        }
                        for i in range(len(questions))
                    ]
                )
            else:
                self.answered_dataset.extend(
                    [
                        {
                            "question": questions[i],
                            "model_response": model_responses[i],
                            "answers": answers[i],
                        }
                        for i in range(len(questions))
                    ]
                )

        if not eval_onthe_fly:

            for eval_batch_idx in range(0, len(self.answered_dataset), max_threads):
                eval_batch = self.answered_dataset[
                    eval_batch_idx : eval_batch_idx + max_threads
                ]
                questions_batch = [datapoint["question"] for datapoint in eval_batch]
                model_responses_batch = [
                    datapoint["model_response"] for datapoint in eval_batch
                ]
                answers_batch = [datapoint["answers"] for datapoint in eval_batch]

                model_grades = self._get_qa_model_grades_threaded(
                    client=client,
                    questions=questions_batch,
                    model_responses=model_responses_batch,
                    answers=answers_batch,
                    openai_model=eval_model,
                    max_tokens=max_eval_tokens,
                    max_threads=max_threads,
                    **kwargs,
                )

                for i, grade in enumerate(model_grades):
                    eval_batch[i]["model_grade"] = grade

        accuracy_total = 0
        for datapoint in self.answered_dataset:
            if datapoint["model_grade"] == "Y":
                accuracy_total += 1
        accuracy = accuracy_total / len(self.answered_dataset)

        return accuracy



class deprecatedMMLUTask(ShortAnswerQuestion):

    # TODO: this is a multiple choice dataset

    def __init__(
        self,
        eval_message=None,
        subject="all",
        streaming=True,
    ):
        super().__init__(eval_message=eval_message)

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

        if subject == "all":
            dataset_list = []
            for subject in available_subjects:
                dataset = datasets.load_dataset(
                    "tasksource/mmlu",
                    subject,
                    split="test",
                    streaming=streaming
                )
                dataset = dataset.remove_columns(
                    set(dataset.column_names) - {"question", "answer", "choices"}
                )
                dataset_list.append(dataset)
            self.dataset = datasets.concatenate_datasets(dataset_list)

        else:
            self.dataset = datasets.load_dataset(
                "tasksource/mmlu",
                subject,
                split="test",
                streaming=streaming
            )
            self.dataset = self.dataset.remove_columns(
                set(self.dataset.column_names) - {"question", "answer", "choices"}
            )
        
