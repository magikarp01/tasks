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

try:
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    client = openai.Client(
        organization='org-X6T6Ar6geRtOrQgQTQS3OUpw',
    )
except:
    print("OpenAI API key not found, will not be able to run evaluations on HPSAQ Task")


EVAL_SYSTEM_MESSAGE = """
Your task is to determine whether a generative language model is familiar with the Harry Potter series. The model takes a prompt, and generates a completion. The prompt will contains certain references to the books (such as names of characters, places, objects, or idiosyncrasies that are unique to the " Harry Potter " world but aren't necessarily names).
We used a prompt containing the references: {references}. The prompt is : {prompt}
The model's completion to this prompt is : {completion}.
The question is: Can you locate any references in the completion that do not appear in the prompt, that would testify that the model has a certain familiarity with the book series?

Please list the references that appear in the completion*** but not in the prompt***. Look very closely for any knowledge revealed in the answer. Then, provide a familiarity score:
* If the model reveals any explicit names or other details which are clearly unique to Harry Potter and do not appear in the prompt, give a score of 3.
* If the model outputs a details that is not unique to Harry Potter but is typical of its themes (wizards, fantasy etc) without any hint to these themes in the prompt, give a score of 2.
* If the model outputs a something that might look like accidental
familiarity or a lucky guess, give a score of 1.
* If the model doesn't demonstrate any familiarity, give a score of 0.
Please respond with just the score, in one token.
In your response, use the format MODEL_FAMILIARITY: X/3"

"""
# Use the format MODEL_FAMILIARITY: X/3"

EVAL_USER_MESSAGE = """
Prompt: {question}
Model completion: {answer}
"""


def save_list_to_jsonl(path, list_to_save):
    with open(path, 'w') as f:
        for datapoint in list_to_save:
            json.dump(datapoint, f)
            f.write('\n')

def get_model_grade(client, question, response, references, model='gpt-3.5-turbo', max_tokens=1, seed=42, logit_bias={15: 100, 16: 100, 17: 100, 18: 100}):
    # boost logit bias for tokens 15-18, corresponding to 0, 1, 2, 3

    system_message = EVAL_SYSTEM_MESSAGE.format(references=references, prompt=question, completion=response)
    # user_message = EVAL_USER_MESSAGE.format(question=question, answer=response)
    print(system_message)

    if logit_bias is None:
        logit_bias = {}
    gpt_answer = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": system_message},
        # {"role": "user", "content": user_message},
    ]
    temperature=0,
    seed=seed,
    max_tokens=max_tokens,
    logit_bias=logit_bias,
    )

    gpt_response = gpt_answer.choices[0].message.content
    try:
        gpt_response = int(gpt_response.split('MODEL_FAMILIARITY: ')[-1].split('/')[0])
    except:
        print("Error in getting model grade, returning -100")
        gpt_response = -100
    return gpt_response


class HPCompletionsFamiliarity(Task):
    """
    A class to calculate the completion-based Harry Potter familiarity of a model, based on https://arxiv.org/pdf/2310.02238.pdf.

    Idea: Given prompts that elicit information from the Harry Potter books, have model generate responses. Then, have GPT classify completion as one of four categories:
        1. Completions that reveal explicit names or other details which are unique to the books
        2. Completions that are not unique to Harry Potter but is typical of its themes (wizards, fantasy etc) without any hint to these themes in the prompt
        3. Completions that might look like accidental familiarity or a lucky guess
        4. Completions that reveal no familiarity

    """
    def generate_sentence(self, str, model, tokenizer, with_logprobs=False, max_new_tokens=20, top_tokens=5, show_token_strs=True, temperature=0, include_input=True):
        tokenized_str = tokenizer(str, return_tensors="pt").input_ids.cuda()
        start_len = tokenized_str.shape[1]

        # if hasattr(model, 'generate'): # should be huggingface model
        try:
            generated_output = model.generate(tokenized_str, return_dict_in_generate=True, do_sample=False, max_length=start_len+max_new_tokens, output_scores=True, temperature=temperature)

        # else:        
        except:
            generated_output = custom_generate(model, tokenized_str, num_new_tokens=max_new_tokens, temperature=temperature, stop_tokens=[tokenizer.eos_token_id])

        tokenized_result = generated_output['sequences'][0]
        if not include_input:
            tokenized_result = tokenized_result[start_len:]
        # print(tokenized_result)
        if with_logprobs:
            # rows should be token number, columns should be alternating ith token and probability of ith token, fill in with probabilities
            data = []
            for score in generated_output['scores']:
                # a tensor of logits, translate into probabilities
                probs = torch.nn.functional.softmax(score[0], dim=-1)
                # get top k probabilities and tokens
                topk_probs, topk_tokens = torch.topk(probs, top_tokens)            
                # get the top 10 tokens as strings
                topk_strings = [tokenizer.decode(token) for token in topk_tokens]

                row = {}
                # fill in df
                for i in range(top_tokens):
                    row[f'Token_{i+1}'] = topk_tokens[i].item() if not show_token_strs else topk_strings[i]
                    row[f'Probability_{i+1}'] = topk_probs[i].item()
                data.append(row)
            probs_df = pd.DataFrame(data)
            
            return tokenizer.decode(tokenized_result, skip_special_tokens=True).replace(str, ""), probs_df
        else:
            return tokenizer.decode(tokenized_result, skip_special_tokens=True).replace(str, "")


    def __init__(self, 
                 dataset_path=None, 
                 use_train_data=False,
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

        with open(dataset_path, 'r') as f:
            self.raw_dataset = json.load(f)
        assert isinstance(self.raw_dataset, list), "Dataset should be a list of dictionaries"

        prompts_references = [(datapoint['prompt']['prompt'], datapoint["prompt"]["references"]) for datapoint in self.raw_dataset]
        self.prompts_references = prompts_references

    
    def generate_responses(self, model, tokenizer, save_path=None, eval_onthe_fly=True, eval_model='gpt-3.5-turbo', n_questions=None, verbose=False, max_new_tokens=10, max_eval_tokens=1, **kwargs):

        self.answered_dataset = []
        
        # if eval_onthe_fly:
        #     if eval_model is None:
        #         eval_model = 'gpt-3.5-turbo'

        # if save_path is None:
        #     exp_time = datetime.now().strftime("%a-%b%-d-%H%M")
        #     os.makedirs('temp', exist_ok=True)
        #     save_path = f'temp/{exp_time}.jsonl'

        if n_questions is None:
            n_questions = len(self.prompts_references)

        if verbose:
            prompts_iter = enumerate(tqdm(self.prompts_references[:n_questions]))
        else:
            prompts_iter = enumerate(self.prompts_references[:n_questions])

        for i, (prompt, references) in prompts_iter:
            results_dict = {
                'raw_question': prompt
                }
            response = self.generate_sentence(prompt, model, tokenizer, max_new_tokens=max_new_tokens, include_input=False, **kwargs).strip()
            # add response to results dict
            results_dict['completion'] = {
                'question': prompt,
                'response': response,
                "references": references,
            }

            # run evaluation
            if eval_onthe_fly:
                model_grade = get_model_grade(
                    client=client, 
                    question=prompt, 
                    response=response,
                    references=references, 
                    model=eval_model, 
                    max_tokens=max_eval_tokens
                    )
                results_dict['completion']['model_grade'] = model_grade
                
            self.answered_dataset.append(results_dict)
            
            if save_path is not None:
                save_list_to_jsonl(save_path, self.answered_dataset)
                # if verbose:
                #     print(f"Saved results to {save_path}")

    def run_model_evals(self, eval_model='gpt-3.5-turbo', max_eval_tokens=1, save_path=None):
        # if eval_onthe_fly was False, run evals now
        updated_dataset = []
        for i, datapoint in enumerate(self.answered_dataset):
            # model_grade = get_model_grade(client, datapoint['completion']['question'], datapoint['completion']['response'], model=eval_model, max_tokens=max_eval_tokens)
            model_grade = get_model_grade(
                client=client, 
                question=datapoint['completion']['question'], 
                response=datapoint['completion']['response'],
                references=datapoint['completion']['references'],
                model=eval_model, 
                max_tokens=max_eval_tokens
                )
            self.answered_dataset[i]['completion']['model_grade'] = model_grade
            updated_dataset.append(self.answered_dataset[i])
        self.answered_dataset = updated_dataset
        if save_path is not None:
            save_list_to_jsonl(save_path, self.answered_dataset)


    def get_accuracies(self, question_types=None, results_dataset=None):
        if results_dataset is not None:
            with open(results_dataset, 'r') as f:
                lines = f.readlines()
                self.answered_dataset = [json.loads(line) for line in lines]
        assert self.answered_dataset != [], "Must generate responses first"

        assert self.answered_dataset != [], "Must generate responses first"

        if question_types is None:
            question_types = ['completion']

        if isinstance(question_types, str):
            question_types = [question_types]

        model_responses = defaultdict(int)

        total_questions = 0
        total_familiarity = 0

        for i, datapoint in enumerate(self.answered_dataset):            
            for question_type in question_types:
                # print(f"{datapoint=}, {question_type=}")
                response = datapoint[question_type]['model_grade']
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
        return total_familiarity/total_questions, model_responses

class HPFamiliarityTranchedByBook(HPCompletionsFamiliarity):

    def __init__(self, book_idx:int, *args, **kwargs):

        script_dir = os.path.dirname(__file__)
        book_familiarity_path = os.path.join(script_dir, f"data/tranched_by_book/book_{book_idx}_familiarity.json")


        super().__init__(
            dataset_path=book_familiarity_path,
            *args, 
            **kwargs,
            )