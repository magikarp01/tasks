from transformers import AutoTokenizer, AutoModelForCausalLM
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
# import asyncio
# import aiohttp

try:
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    global_client = openai.Client(
        # organization='org-X6T6Ar6geRtOrQgQTQS3OUpw',
    )
except:
    print("OpenAI API key not found, will not be able to run evaluations on HPSAQ Task")


EVAL_SYSTEM_MESSAGE = """
Your task is to determine whether a generative language model is familiar with the Harry Potter series. The model takes a prompt, and generates a completion. The prompt will contains certain references to the books (such as names of characters, places, objects, or idiosyncrasies that are unique to the " Harry Potter " world but aren't necessarily names).
We used a prompt containing the references: {references}. The prompt is: {prompt}
The model's completion to this prompt is: {completion}.
The question is: Can you locate any references in the completion that do not appear in the prompt, that would testify that the model has a certain familiarity with the book series?

Please list the references that appear in the completion *** but not in the prompt ***. Look very closely for any knowledge revealed in the answer. Then, provide a familiarity score:
* If the model reveals any explicit names or other details which are clearly unique to Harry Potter and do not appear in the prompt, give a score of 3.
* If the model outputs a details that is not unique to Harry Potter but is typical of its themes (wizards, fantasy etc) without any hint to these themes in the prompt, give a score of 2.
* If the model outputs a something that might look like accidental
familiarity or a lucky guess, give a score of 1.
* If the model doesn't demonstrate any familiarity, give a score of 0.
Please respond with just the score, in one token.
In your response, use the format "<listing reasoning...>\nMODEL_FAMILIARITY: X/3".
"""
# Use the format MODEL_FAMILIARITY: X/3"

# EVAL_USER_MESSAGE = """
# Prompt: {question}
# Model completion: {answer}
# """
EVAL_USER_MESSAGE = """
Your task is to determine whether a generative language model is familiar with the Harry Potter series. The model takes a prompt, and generates a completion. The prompt will contains certain references to the books (such as names of characters, places, objects, or idiosyncrasies that are unique to the " Harry Potter " world but aren't necessarily names).
We used a prompt containing the references: {references}. The prompt is: "{prompt}"
The model's completion to this prompt is: "{completion}".
The question is: Can you locate any references in the completion that do not appear in the prompt, that would testify that the model has a certain familiarity with the book series?

Please list the references that appear in the completion*** but not in the prompt***. Look very closely for any knowledge revealed in the answer. Then, provide a familiarity score:
* If the model reveals any explicit names or other details which are clearly unique to Harry Potter and do not appear in the prompt, give a score of 3.
* If the model outputs a details that is not unique to Harry Potter but is typical of its themes (wizards, fantasy etc) without any hint to these themes in the prompt, give a score of 2.
* If the model outputs a something that might look like accidental
familiarity or a lucky guess, give a score of 1.
* If the model doesn't demonstrate any familiarity, give a score of 0.
Please respond with just the score, in one token.
In your response, use the format "<listing reasoning...>\nMODEL_FAMILIARITY: X/3".
"""

THEMED_EVAL_MESSAGE = """
Your task is to determine whether a generative language model is familiar with the {theme}. The model takes a prompt, and generates a completion. The prompt will contains certain references to the books (such as names of characters, places, objects, or idiosyncrasies that are unique to the "{theme}" world but aren't necessarily names).
We used a prompt containing the references: {{references}}. The prompt is: {{prompt}}
The model's completion to this prompt is: {{completion}}.
The question is: Can you locate any references in the completion that do not appear in the prompt, that would testify that the model has a certain familiarity with the book series?

Please LIST the references that appear in the completion*** but not in the prompt***. Look very closely for any knowledge revealed in the answer. Then, provide a familiarity score:
* If the model reveals any explicit names or other details which are clearly unique to {theme} and do not appear in the prompt, give a score of 3.
* If the model outputs a details that is not unique to {theme} but is typical of its themes (wizards, fantasy etc) without any hint to these themes in the prompt, give a score of 2.
* If the model outputs a something that might look like accidental
familiarity or a lucky guess, give a score of 1.
* If the model doesn't demonstrate any familiarity, give a score of 0.
Please respond with just the score, in one token.
In your response, USE the format "<listing reasoning...>\nMODEL_FAMILIARITY: X/3".
"""


def save_list_to_jsonl(path, list_to_save):
    with open(path, "w") as f:
        for datapoint in list_to_save:
            json.dump(datapoint, f)
            f.write("\n")


def get_model_grades_threaded(client, questions, responses, references, model="gpt-3.5-turbo", max_tokens=None, max_threads=5, seed=42, eval_message=None, logit_bias=None):

    assert len(questions) == len(responses) == len(references), "Questions, responses, and references must be the same length"

    def get_model_grade_internal(question, response, reference, logit_bias):
        if eval_message is None:
            user_message = EVAL_USER_MESSAGE.format(references=reference, prompt=question, completion=response)
        else:
            user_message = eval_message.format(references=reference, prompt=question, completion=response)

        if logit_bias is None:
            logit_bias = {}

        gpt_answer = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": user_message},
            ],
            temperature=0,
            seed=seed,
            max_tokens=max_tokens,
            logit_bias=logit_bias,
        )

        gpt_response = gpt_answer.choices[0].message.content
        try:
            gpt_response = int(gpt_response.split("MODEL_FAMILIARITY: ")[-1].split("/")[0])
        except:
            print("Error in getting model grade, returning -100")
            gpt_response = -100
        return gpt_response

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        results = list(executor.map(get_model_grade_internal, questions, responses, references, [logit_bias]*len(questions)))


    return results


class HPCompletionsFamiliarity(Task):
    """
    A class to calculate the completion-based Harry Potter familiarity of a model, based on https://arxiv.org/pdf/2310.02238.pdf.

    Idea: Given prompts that elicit information from the Harry Potter books, have model generate responses. Then, have GPT classify completion as one of four categories:
        1. Completions that reveal explicit names or other details which are unique to the books
        2. Completions that are not unique to Harry Potter but is typical of its themes (wizards, fantasy etc) without any hint to these themes in the prompt
        3. Completions that might look like accidental familiarity or a lucky guess
        4. Completions that reveal no familiarity

    """

    def generate_sentence(
        self,
        model,
        tokenizer,
        strs: list,  # strs is now a list of strings
        with_logprobs=False,
        max_new_tokens=20,
        top_tokens=5,
        show_token_strs=True,
        temperature=1.0,
        include_input=True,
        device='cuda'
    ):
        # Encode all the inputs at once
        tokenizer.padding_side = "left"
        tokenized_inputs = tokenizer.batch_encode_plus(
            strs, return_tensors="pt", padding=True,
        )
        
        tokenized_inputs = {k: v.to(device) for k, v in tokenized_inputs.items()}  # Move to model's device

        try:
            outputs = model.generate(
                **tokenized_inputs,
                max_length=tokenized_inputs['input_ids'].shape[1] + max_new_tokens,
                temperature=temperature,
                top_k=top_tokens,
                return_dict_in_generate=True,
                output_scores=with_logprobs,
                pad_token_id=tokenizer.pad_token_id,
            )
            sequences = outputs.sequences
            scores = outputs.scores if with_logprobs else None

        except Exception as e:

            print(f"Falling back to custom generation due to exception: {e}\nRunning model as a model inference function instead of a huggingface model.")

            custom_output = custom_generate(
                model_inference_fn=model,
                input=tokenized_inputs['input_ids'],
                num_new_tokens=max_new_tokens,
                temperature=temperature,
                stop_tokens=[tokenizer.eos_token_id],
                verbose=False  # Set to True for progress bar
            )
            sequences = custom_output["sequences"]
            scores = custom_output["scores"]

        # decoded_sentences = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs.sequences]
        decoded_sentences = [tokenizer.decode(ids, skip_special_tokens=True) for ids in sequences]


        if not include_input:
            decoded_sentences = [sentence[len(strs[i]):] for i, sentence in enumerate(decoded_sentences)]

        if with_logprobs and scores is not None:
            data = []
            for i, score in enumerate(scores):
                probs = torch.softmax(score, dim=-1)
                topk_probs, topk_tokens = probs.topk(top_tokens, dim=-1)

                for batch_index in range(topk_tokens.size(0)):
                    topk_strings = [tokenizer.decode(token.item(), skip_special_tokens=True) for token in topk_tokens[batch_index]]
                    row = {}
                    for j in range(top_tokens):
                        token_key = f"Token_{j+1}"
                        prob_key = f"Probability_{j+1}"
                        row[token_key] = topk_strings[j] if show_token_strs else topk_tokens[batch_index][j].item()
                        row[prob_key] = topk_probs[batch_index][j].item()
                    data.append(row)

            probs_df = pd.DataFrame(data)
            return decoded_sentences, probs_df
        else:
            return decoded_sentences


    def __init__(
        self,
        dataset_path=None,
        use_train_data=False,
        eval_message=EVAL_USER_MESSAGE,
    ):
        """
        dataset_path is json file that looks like:
        [
            {
                "prompt": {
                    "references": [
                        "Ron",
                        "Hermione",
                        "wand"
                    ],
                    "prompt": "Ron and Hermione were practicing their spells when Ron accidentally cast a spell that caused",
                    "subtlety": 8
                }
            },
            {
                "prompt": {
                    "references": [
                        "Dumbledore",
                        "phoenix"
                    ],
                    "prompt": "In the headmaster's office, a magnificent phoenix perched on its stand, waiting for Dumbledore to",
                    "subtlety": 7
                }
            },
        ]
        """

        if dataset_path is None:
            dataset_path = "tasks/hp/data/msr_data/evaluation_prompts.json"

        with open(dataset_path, "r") as f:
            self.raw_dataset = json.load(f)
        assert isinstance(
            self.raw_dataset, list
        ), "Dataset should be a list of dictionaries"

        prompts_references = [
            (datapoint["prompt"]["prompt"], datapoint["prompt"]["references"])
            for datapoint in self.raw_dataset
        ]
        self.prompts_references = prompts_references
        self.eval_message = eval_message


    def generate_responses(
        self,
        model,
        tokenizer,
        save_path=None,
        eval_onthe_fly=True,
        eval_model="gpt-3.5-turbo",
        batch_size=5, # equals max thread for api call
        n_questions=None,
        verbose=False,
        max_new_tokens=20,
        max_eval_tokens=250,
        **kwargs,
    ):
        self.answered_dataset = []

        if n_questions is None:
            n_questions = len(self.prompts_references)


        prompts_iter = self.prompts_references[:n_questions]

        for i in range(0, n_questions, batch_size):

            if verbose:
                print(f"Processing questions {i} to {i+batch_size} of {n_questions}")

            batch_slice = prompts_iter[i:i + batch_size]
            prompts_batch = [item[0] for item in batch_slice]
            references_batch = [item[1] for item in batch_slice]

            responses = self.generate_sentence(
                strs=prompts_batch,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                include_input=False,
                **kwargs,
            )

            # Assuming generate_sentence returns a list of responses corresponding to prompts_batch
            for j, response in enumerate(responses):
                self.answered_dataset.append({
                    "raw_question": prompts_batch[j],
                    "completion": {
                        "question": prompts_batch[j],
                        "response": response,
                        "references": references_batch[j],
                    }
                })

        if eval_onthe_fly:
            # Perform batched evaluation
            questions_batch = [item['completion']['question'] for item in self.answered_dataset]
            responses_batch = [item['completion']['response'] for item in self.answered_dataset]
            references_batch = [item['completion']['references'] for item in self.answered_dataset]

            model_grades = get_model_grades_threaded(
                client=global_client,
                questions=questions_batch,
                responses=responses_batch,
                references=references_batch,
                model=eval_model,
                max_tokens=max_eval_tokens,
                **kwargs,  # Additional kwargs like max_threads, seed, eval_message can be passed directly
            )

            # Update the dataset with model grades
            for i, grade in enumerate(model_grades):
                self.answered_dataset[i]['completion']['model_grade'] = grade

        if save_path is not None:
            try:
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
            except:
                pass
            with open(save_path, 'w') as f:
                for item in self.answered_dataset:
                    f.write(json.dumps(item) + '\n')
            if verbose:
                print(f"Saved dataset to {save_path}")

    # Note: You'll need to define or adjust `generate_sentence` to return a list of responses corresponding to the input batch of prompts.
    # Also, ensure `self.prompts_references` and other necessary attributes or methods are properly defined within the class.



    def run_model_evals(
        self, 
        client=None,  # Assuming the API client is passed here
        eval_model="gpt-3.5-turbo", 
        max_eval_tokens=None, 
        save_path=None,
        batch_size=1,  # Use batch_size parameter to control the size of each batch
        **kwargs,  # Additional arguments
    ):
        if client is None:
            client = global_client

        # Check if answered_dataset is not empty
        if not self.answered_dataset:
            print("No data available for evaluation.")
            return

        # Prepare batches
        total_len = len(self.answered_dataset)
        batches = [self.answered_dataset[i:i + batch_size] for i in range(0, total_len, batch_size)]

        updated_dataset = []

        for batch in tqdm(batches):
            questions_batch = [datapoint["completion"]["question"] for datapoint in batch]
            responses_batch = [datapoint["completion"]["response"] for datapoint in batch]
            references_batch = [datapoint["completion"]["references"] for datapoint in batch]

            # Call your batched evaluation function here
            model_grades = get_model_grades_threaded(
                client=client,
                questions=questions_batch,
                responses=responses_batch,
                references=references_batch,
                model=eval_model,
                max_tokens=max_eval_tokens,
                **kwargs,  # Passing additional kwargs if needed
            )

            # Update the dataset with model grades
            for i, grade in enumerate(model_grades):
                batch[i]["completion"]["model_grade"] = grade
                updated_dataset.append(batch[i])

        self.answered_dataset = updated_dataset

        # Save the updated dataset if a save_path is provided
        if save_path is not None:
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                for item in self.answered_dataset:
                    f.write(json.dumps(item) + '\n')
            print(f"Saved dataset to {save_path}")

    def get_accuracies(self, question_types=None, results_dataset=None):
        if results_dataset is not None:
            with open(results_dataset, "r") as f:
                lines = f.readlines()
                self.answered_dataset = [json.loads(line) for line in lines]
        assert self.answered_dataset != [], "Must generate responses first"

        assert self.answered_dataset != [], "Must generate responses first"

        if question_types is None:
            question_types = ["completion"]

        if isinstance(question_types, str):
            question_types = [question_types]

        model_responses = defaultdict(int)

        total_questions = 0
        total_familiarity = 0

        for i, datapoint in tqdm(enumerate(self.answered_dataset)):
            for question_type in question_types:
                response = datapoint[question_type]["model_grade"]
                model_responses[response] += 1

                if response == 3:
                    total_familiarity += 5
                    total_questions += 1
                elif response == 2:
                    total_familiarity += 1
                    total_questions += 1
                else:
                    if response == 1 or response == 0:
                        total_questions += 1
                    # assert response == '1' or response == '0', f"Model grade should be 0, 1, 2, or 3 but is {response}"
        return total_familiarity / total_questions, model_responses


class HPFamiliarityTranchedByBook(HPCompletionsFamiliarity):

    def __init__(self, book_idx: int, *args, **kwargs):

        script_dir = os.path.dirname(__file__)
        book_familiarity_path = os.path.join(
            script_dir, f"data/tranched_by_book/book_{book_idx}_familiarity.json"
        )

        super().__init__(
            dataset_path=book_familiarity_path,
            *args, 
            **kwargs,
            )

class HPFamiliaritySideEffects(HPCompletionsFamiliarity):

    def __init__(self, side_effects_idx:int, *args, **kwargs):

        assert side_effects_idx >= 0 and side_effects_idx <= 4, "side_effects_idx must be between 0 and 4"

        side_effects_paths = [
            "data/side_effects/british_mythology_familiarity.json",
            # "data/side_effects/cultural_impact_familiarity.json",
            "data/side_effects/harry_potter_film_production_familiarity.json",
            "data/side_effects/dungeons_and_dragons_familiarity.json",
            "data/side_effects/lord_of_the_rings_familiarity.json",
            "data/side_effects/wizard_of_oz_familiarity.json",
        ]

        script_dir = os.path.dirname(__file__)
        side_effects_path = os.path.join(script_dir, side_effects_paths[side_effects_idx])

        super().__init__(
            dataset_path=side_effects_path,
            eval_message=THEMED_EVAL_MESSAGE.format(theme=side_effects_paths[side_effects_idx].split("/")[-1].split("_familiarity")[0]),
            *args, 
            **kwargs,
            )

class HPFamiliaritySpanish(HPCompletionsFamiliarity):

    def __init__(self, *args, **kwargs):

        script_dir = os.path.dirname(__file__)
        spanish_familiarity_path = os.path.join(
            script_dir, "data/msr_data/evaluation_spanish_prompts.json"
        )

        super().__init__(
            dataset_path=spanish_familiarity_path,
            eval_message=THEMED_EVAL_MESSAGE.format(theme="spanish Harry Potter"),
            *args, 
            **kwargs,
            )
    
class HPFamiliarityRussian(HPCompletionsFamiliarity):

    def __init__(self, *args, **kwargs):

        script_dir = os.path.dirname(__file__)
        russian_familiarity_path = os.path.join(
            script_dir, "data/msr_data/evaluation_russian_prompts.json"
        )

        super().__init__(
            dataset_path=russian_familiarity_path,
            eval_message=THEMED_EVAL_MESSAGE.format(theme="russian Harry Potter"),
            *args, 
            **kwargs,
            )