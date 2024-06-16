import json
import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
from dotenv import load_dotenv
import os
import openai

try:
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    global_client = openai.Client(
        # organization='org-X6T6Ar6geRtOrQgQTQS3OUpw',
    )
except:
    print("OpenAI API key not found, will not be able to run evaluations on Sports Trivia Task")


EVAL_USER_MESSAGE = """Your task is to determine whether a generative language model is familiar with sports knowledge. The model takes a prompt, and generates a completion. The prompt will contain questions about sports, and the model's job is to answer those questions.
We used a prompt about {sport_name}. The question is: {prompt}
The correct answer to the question is {answer}.
The model's completion to this prompt is: {completion}.
Can you judge if the model accurately answered the question? Please respond with one word, "Yes" or "No". Be a bit generous with grading, but if the model's completion is not relevant to the question, please respond "No".
"""

FORMAT_GENERATION_INPUT = """Please respond to the question in as few words as possible, and be specific. Question: {prompt}\nAnswer:"""

from concurrent.futures import ThreadPoolExecutor
from tasks.inference_utils import custom_generate
import random
from collections import defaultdict

def get_model_grades_threaded(client, questions, responses, correct_answers, sports, model="gpt-3.5-turbo", max_tokens=1, max_threads=5, seed=42, eval_message=None, logit_bias=None):

    assert len(questions) == len(responses) == len(correct_answers) == len(sports), f"Length of questions, responses, correct_answers, and sports should be the same but instead are {len(questions)}, {len(responses)}, {len(correct_answers)}, and {len(sports)} respectively."

    def get_model_grade_internal(question, response, correct_answer, sport, logit_bias):
        if eval_message is None:
            user_message = EVAL_USER_MESSAGE.format(sport_name=sport, prompt=question, answer=correct_answer, completion=response)
        else:
            user_message = eval_message.format(sport_name=sport, prompt=question, answer=correct_answer, completion=response)

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
            if gpt_response.lower() == "yes":
                gpt_response = 1
            elif gpt_response.lower() == "no":
                gpt_response = 0
        except:
            print("Error in getting model grade, returning -100")
            gpt_response = -100
        return gpt_response

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        # results = list(executor.map(get_model_grade_internal, questions, responses, references, [logit_bias]*len(questions)))
        results = list(executor.map(get_model_grade_internal, questions, responses, correct_answers, sports, [logit_bias]*len(questions)))

    return results

from tasks import Task
class SportsFamiliarity(Task):
    """
    A class to calculate the completion-based Sports familiarity of a model, using a GPT4 autograder.
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
        do_sample=True,
        temperature=1.0,
        include_input=True,
        device='cuda'
    ):
        # print(strs)
        # Encode all the inputs at once
        tokenizer.padding_side = "left"
        tokenized_inputs = tokenizer.batch_encode_plus(
            strs, return_tensors="pt", padding=True,
        )
        
        tokenized_inputs = {k: v.to(device) for k, v in tokenized_inputs.items()}  # Move to model's device
        print(f"{tokenized_inputs.keys()=}")

        try:
            outputs = model.generate(
                # **tokenized_inputs,
                tokenized_inputs['input_ids'],
                attention_mask=tokenized_inputs['attention_mask'],
                max_length=tokenized_inputs['input_ids'].shape[1] + max_new_tokens,
                do_sample=do_sample,
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
                do_sample=do_sample,
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
        shuffle=True
    ):
        """
        dataset_path is json file that looks like:
        [
            {"question": "What is the standard width of an NFL football field in feet?", "answer": "160 feet", "sport": "football"},
            ...
        ]
        """

        if dataset_path is None:
            dataset_path = "tasks/facts/data/side_effect_sport_trivia.json"

        with open(dataset_path, "r") as f:
            self.raw_dataset = json.load(f)
        assert isinstance(
            self.raw_dataset, list
        ), "Dataset should be a list of dictionaries"

        if shuffle:
            random.shuffle(self.raw_dataset)

        # prompts_references = [
        #     datapoint["question"]
        #     for datapoint in self.raw_dataset
        # ]
        
        # self.prompts_references = prompts_references
        # self.eval_message = eval_message


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
        prompt_format=None,
        **kwargs,
    ):
        self.answered_dataset = []

        if n_questions is None:
            n_questions = len(self.raw_dataset)

        # prompts_iter = self.prompts_references[:n_questions]
        data_iter = self.raw_dataset[:n_questions]

        for i in range(0, n_questions, batch_size):

            if verbose:
                print(f"Processing questions {i} to {i+batch_size} of {n_questions}")

            # batch_slice = prompts_iter[i:i + batch_size]
            prompts_batch = [row['question'] for row in data_iter[i:i + batch_size]]
            answers_batch = [row['answer'] for row in data_iter[i:i + batch_size]]
            sports_batch = [row['sport'] for row in data_iter[i:i + batch_size]]


            if prompt_format is not None:
                strs = [prompt_format.format(prompt=prompt) for prompt in prompts_batch]
            else:
                strs = prompts_batch

            responses = self.generate_sentence(
                strs=strs,
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
                        "question": strs[j],
                        "response": response,
                    },
                    "sport": sports_batch[j],
                    "answer": answers_batch[j],
                })

        if eval_onthe_fly:
            # Perform batched evaluation
            questions_batch = [datapoint["raw_question"] for datapoint in self.answered_dataset]
            responses_batch = [item['completion']['response'] for item in self.answered_dataset]
            sports_batch = [item['sport'] for item in self.answered_dataset]
            answers_batch = [item['answer'] for item in self.answered_dataset]

            model_grades = get_model_grades_threaded(
                client=global_client,
                questions=questions_batch,
                responses=responses_batch,
                correct_answers=answers_batch,
                sports=sports_batch,
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
            # questions_batch = [datapoint["completion"]["question"] for datapoint in batch]
            # responses_batch = [datapoint["completion"]["response"] for datapoint in batch]
            # references_batch = [datapoint["completion"]["references"] for datapoint in batch]
            questions_batch = [datapoint["raw_question"] for datapoint in batch]
            responses_batch = [datapoint["completion"]["response"] for datapoint in batch]
            correct_answers_batch = [datapoint["answer"] for datapoint in batch]
            sports_batch = [datapoint["sport"] for datapoint in batch]

            # Call your batched evaluation function here
            model_grades = get_model_grades_threaded(
                client=client,
                questions=questions_batch,
                responses=responses_batch,
                correct_answers=correct_answers_batch,
                sports=sports_batch,
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

    def get_accuracies(self, question_types=None, results_dataset=None, split_by_sport=True):
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


        if not split_by_sport:
            model_responses = defaultdict(int)
            total_questions = 0
            total_familiarity = 0
        else:
            model_responses = {}
            sport_questions = defaultdict(int)
            sport_familiarity = defaultdict(int)

        for i, datapoint in tqdm(enumerate(self.answered_dataset)):
            for question_type in question_types:
                response = datapoint[question_type]["model_grade"]

                if split_by_sport:
                    sport = datapoint["sport"]
                    if sport not in model_responses:
                        model_responses[sport] = defaultdict(int)
                    model_responses[sport][response] += 1
                    if response == 1:
                        sport_familiarity[sport] += 1
                        sport_questions[sport] += 1
                    elif response == 0:
                        sport_questions[sport] += 1
                        
                else:
                    model_responses[response] += 1

                    if response == 1:
                        total_familiarity += 1
                        total_questions += 1
                    elif response == 0:
                        total_questions += 1
        
        if not split_by_sport:
            return total_familiarity / total_questions, model_responses
        else:
            for sport in sport_familiarity:
                sport_familiarity[sport] = sport_familiarity[sport] / sport_questions[sport]
            return sport_familiarity, model_responses


def test_prompt_accuracy(model, tokenizer, sentence):
    tokenized = tokenizer(sentence, return_tensors="pt").input_ids.cuda()
    tokens_len = tokenized.shape[1]
    with torch.no_grad():
        try:
            generation = model.generate(tokenized, max_new_tokens=1, do_sample=False, verbose=False)
        except:
            generation = model.generate(tokenized, max_new_tokens=1, do_sample=False)
        decoded = tokenizer.decode(generation[0, tokens_len:])
    return decoded


def respond_sports_side_effects(model, tokenizer):
    side_effect_format = "Question: In which sport did Deacon Jones coin the term \"sack\" to describe tackling the quarterback behind the line of scrimmage?\nA. football\nB. baseball\nC. basketball\nD. tennis\nE. golf\nAnswer: football\nQuestion: In which sport was the designated hitter (DH) rule adopted by the American League in 1973, allowing a player to bat in place of the pitcher?\nA. football\nB. baseball\nC. basketball\nD. tennis\nE. golf\nAnswer: baseball\nQuestion: In which sport was the three-point line introduced in the 1979-1980 season, allowing players to score three points from shots made beyond this line?\nA. football\nB. baseball\nC. basketball\nD. tennis\nE. golf\nAnswer: basketball\nQuestion: In which sport is a \"tiebreak\" used to decide a set when the score reaches 6-all?\nA. football\nB. baseball\nC. basketball\nD. tennis\nE. golf\nAnswer: tennis\nQuestion: In what sport did Tiger Woods play?\nA. football\nB. baseball\nC. basketball\nD. tennis\nE. golf\nAnswer: golf\nQuestion: {question}\nA. football\nB. baseball\nC. basketball\nD. tennis\nE. golf\nAnswer:"

    with open("tasks/facts/data/side_effect_sport_answers.json", "r") as f:
        sports_questions = json.load(f)

    from collections import defaultdict

    accuracies = defaultdict(int)
    tot_questions = defaultdict(int)
    # for sport_name, sport_questions in [("football", football_questions), ("baseball", baseball_questions), ("basketball", basketball_questions), ("tennis", tennis_questions)]:
    for row in tqdm(sports_questions):
        sport_name = row["answer"]
        formatted_question = side_effect_format.format(question=row['question'])
        tot_questions[sport_name] += 1

        # ref_response = test_prompt_accuracy(reference_model, formatted_question)
        # unlearned_response = test_prompt_accuracy(unlearned_model, formatted_question)
        # if sport_name in ref_response.lower():
        #     reference_accuracies[sport_name] += 1
        # if sport_name in unlearned_response.lower():
        response = test_prompt_accuracy(model, tokenizer, formatted_question)
        if sport_name in response.lower():
            accuracies[sport_name] += 1
    
    return accuracies, tot_questions

from transformers import AutoTokenizer
# from tasks.harmbench.FastHarmBenchEvals import run_general_evals
from tasks.general_capabilities.MCTask_redo import run_general_evals
from tasks import PileTask, OWTTask
def run_side_effects_evals(model, evals_to_run=["Sports Answers", "Sports Familiarity", "General", "Cross Entropy"], model_type="gemma", use_short=False, eval_model="gpt-4-turbo", batch_size=32, verbose=False, n_iters=5, general_batch_size=10):
    if "gemma" in model_type:
        model_name = "google/gemma-7b"
    elif "pythia" in model_type:
        model_name = "EleutherAI/pythia-2.8b"
    elif "llama_2" in model_type:
        model_name = "meta-llama/Llama-2-7b-chat-hf"
    else:
        model_name = model_type
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    left_tokenizer = AutoTokenizer.from_pretrained(model_name)
    left_tokenizer.pad_token_id = left_tokenizer.eos_token_id
    left_tokenizer.padding_side = "left"

    return_dict = {}
    if "Sports Answers" in evals_to_run:
        sports_accuracies, tot_questions = respond_sports_side_effects(model, tokenizer)
        if verbose:
            print("Sports Answers:")

        for sport in tot_questions:
        # for sport, correct in sports_accuracies.items():
            if verbose:
                print(f"{sport}: {sports_accuracies[sport]}/{tot_questions[sport]}")
        return_dict["Sports Answers"] = {sport: sports_accuracies[sport]/tot_questions[sport] for sport in tot_questions}
        
    if "Sports Familiarity" in evals_to_run:

        if use_short:
            dataset_path = "tasks/facts/data/side_effect_sport_trivia_short.json"
        else:
            dataset_path = "tasks/facts/data/side_effect_sport_trivia.json"
        sports_trivia_familiarity = SportsFamiliarity(dataset_path=dataset_path)
        sports_trivia_familiarity.generate_responses(model, left_tokenizer, save_path=None, eval_onthe_fly=False, max_new_tokens=20, do_sample=False, verbose=True, batch_size=batch_size, prompt_format=FORMAT_GENERATION_INPUT)
        sports_trivia_familiarity.run_model_evals(eval_model=eval_model, max_eval_tokens=1, save_path=None, batch_size=20)
        familiarity, responses = sports_trivia_familiarity.get_accuracies(split_by_sport=True)
        if verbose:
            print(f"Unlearned Model Familiarity: {familiarity}")
        return_dict["Sports Familiarity"] = familiarity
    
    if "General" in evals_to_run:
        general_capabilities = run_general_evals(model, model_type=model_type, evals_to_include=["MMLU"], batch_size=general_batch_size)
        if verbose:
            print("General Capabilities:")
            print(general_capabilities)
        return_dict["General"] = general_capabilities

    if "Cross Entropy" in evals_to_run:
        pile = PileTask(batch_size=batch_size, tokenizer=tokenizer, ctx_length=100)
        # owt = OWTTask(batch_size=batch_size, tokenizer=tokenizer, ctx_length=100)

        pile_ce = 0
        # owt_ce = 0
        for i in range(n_iters):
            pile_ce += pile.get_test_loss(model).item()
            # owt_ce += owt.get_test_loss(model).item()
        if verbose:
            print("Pile Cross Entropy:", pile_ce / n_iters)
            # print("OWT Cross Entropy:", owt_ce / n_iters)
        # return_dict["Cross Entropy"] = {"Pile": pile_ce / n_iters, "OWT": owt_ce / n_iters}
        return_dict["Cross Entropy"] = {"Pile": pile_ce / n_iters}
    return return_dict