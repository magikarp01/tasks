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
from concurrent.futures import ThreadPoolExecutor

try:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    client = openai.Client(
        # organization='org-X6T6Ar6geRtOrQgQTQS3OUpw',
    )
except:
    print("OpenAI API key not found, will not be able to run evaluations on HPSAQ Task")

QA_TEMPLATE = """
Question:
{question}

Answer:
{answer}
"""

Q_TEMPLATE = """
Question:
{question}

Answer:
"""

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>", "<</SYS>>\n\n"

SAQ_SYSTEM_PROMPT = B_SYS + """
I want you to answer the following question about Harry Potter and respond with an answer. Your response should be a single sentence.
""" + E_SYS

# SAQ_SYSTEM_PROMPT = B_SYS + """
# I want you to answer the following question about Harry Potter and respond with an answer. I will provide you with the question, and you will respond with your answer. Your response should be as short as possible, a single statement. I will now provide you with the question.
# """ + E_SYS

ZERO_SHOT_TEMPLATE = """
[INST] Question:
{question}

Answer:[/INST] 
"""

FEW_SHOT_TEMPLATE = """
[INST] Question:
{0}

Answer:[/INST] 
{1}

[INST] Question:
{2}

Answer:[/INST] 
{3}

[INST] Question:
{4}

Answer:[/INST] 
{5}

[INST] Question:
{{question}}

Answer:[/INST] 
"""

UNRELATED_FEW_SHOT_TEMPLATE = """
[INST] Question:
Which planet in our solar system is known as the Red Planet?

Answer:[/INST] 
Mars

[INST] Question:
Who wrote the play "Romeo and Juliet"?

Answer:[/INST] 
William Shakespeare

[INST] Question:
Name the Great Lakes of North America.

Answer:[/INST] 
Huron, Ontario, Michigan, Erie, Superior

[INST] Question:
{question}

Answer:[/INST] 
"""

EVAL_SYSTEM_MESSAGE = """
I want you to make a judgement on the correctness of a given answer by a contestant. The question pertains to the Harry Potter universe. The contestant has been asked a question and has responded with their answer. Using the provided "Perfect Answer" and your own knowledge, you will decide whether the contestant's response is correct. Your output will be either `Y` or `N` for yes or no. You must not output anything else.
"""
EVAL_USER_MESSAGE = """
Question: {question}
Contestant's Answer: {answer}
Perfect Answer: {perfect_answer}
"""



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


def generate_sentence(str, model, tokenizer, with_logprobs=False, max_new_tokens=20, top_tokens=5, show_token_strs=True, temperature=0):
    tokenized_str = tokenizer(str, return_tensors="pt").input_ids.cuda()
    start_len = tokenized_str.shape[1]

    if hasattr(model, 'generate'): # should be huggingface model
        generated_output = model.generate(tokenized_str, return_dict_in_generate=True, do_sample=False, max_length=start_len+max_new_tokens, output_scores=True, temperature=temperature)

    else:        
        generated_output = custom_generate(model, tokenized_str, num_new_tokens=max_new_tokens, temperature=temperature, stop_tokens=[tokenizer.eos_token_id])

    # print(generated_output)

    tokenized_result = generated_output['sequences'][0]
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
    


def batch_generate_sentence(
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
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    
    tokenized_inputs = {k: v.to(device) for k, v in tokenized_inputs.items()}  # Move to model's device

    try:
        with torch.no_grad():
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



def clear_gpu(model):
    model.cpu()
    torch.cuda.empty_cache()

def compare_responses(prompt, model1, model2, max_new_tokens=200, manage_gpu=False):
    if manage_gpu:
        clear_gpu(model1)
        clear_gpu(model2)
        model1.cuda()
    model1_gen = generate_sentence(prompt, model1, max_new_tokens=max_new_tokens)
    if manage_gpu:
        clear_gpu(model1)
        model2.cuda()
    model2_gen = generate_sentence(prompt, model2, max_new_tokens=max_new_tokens)
    if manage_gpu:
        clear_gpu(model2)
    return model1_gen, model2_gen

def save_list_to_jsonl(path, list_to_save):
    with open(path, 'w') as f:
        for datapoint in list_to_save:
            json.dump(datapoint, f)
            f.write('\n')

def get_model_grade(client, question, response, perfect_answer, model='gpt-3.5-turbo', max_tokens=1):

    system_message = EVAL_SYSTEM_MESSAGE
    user_message = EVAL_USER_MESSAGE.format(question=question, answer=response, perfect_answer=perfect_answer)

    response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ],
    temperature=0,
    seed=42,
    )
    return response.choices[0].message.content


class HPSAQ(Task):

    def __init__(self, 
                 dataset_path=None, 
                 system_prompt=SAQ_SYSTEM_PROMPT, 
                 zero_shot_template=ZERO_SHOT_TEMPLATE, 
                 few_shot_template=FEW_SHOT_TEMPLATE, unrelated_few_shot_template=UNRELATED_FEW_SHOT_TEMPLATE,
                 use_train_data=False,
                 ):


        if dataset_path is None:
            script_dir = os.path.dirname(os.path.realpath(__file__))
            if use_train_data:
                dataset_path = os.path.join(script_dir, 'data/harry_potter_trivia_502_v2.jsonl')
            else:
                dataset_path = os.path.join(script_dir, 'data/hp_trivia_test.jsonl')

        with open(dataset_path, 'r') as f:
            lines = f.readlines()
            self.raw_dataset = [json.loads(line) for line in lines]
        for line in self.raw_dataset:
            assert isinstance(line, dict), "Each line in the dataset should be a dictionary"
            assert 'question' in line, "Key 'question' not found in the dictionary"
            assert 'true_answer' in line, "Key 'true_answer' not found in the dictionary"
        self.answered_dataset = []

        # get first three questions from dataset and create the few-shot template
        few_shot_questions = []
        for question in self.raw_dataset[:3]:
        #     few_shot_questions.append(QA_TEMPLATE.format(question=question['question'], answer=question['true_answer']))
        # few_shot_questions = '\n'.join(few_shot_questions)
        # few_shot_template = few_shot_template.format(few_shot_questions=few_shot_questions)
            few_shot_questions.append(question['question'])
            few_shot_questions.append(question['true_answer'])
        few_shot_template = few_shot_template.format(*few_shot_questions)

        self.system_prompt = system_prompt
        self.zero_shot_template = zero_shot_template
        self.few_shot_template = few_shot_template
        self.unrelated_few_shot_template = unrelated_few_shot_template

        self.prefix_system_prompt = ""
        self.suffix_system_prompt = ""
        self.suffix_question = ""
    
    def format_prompts(self):
        # format the template then format the prompt then format the prompts

        self.zero_shot_question = self.prefix_system_prompt + self.system_prompt + self.suffix_system_prompt + self.zero_shot_formatted_template + self.suffix_question

        self.few_shot_question = self.prefix_system_prompt + self.system_prompt + self.suffix_system_prompt + self.few_shot_formatted_template + self.suffix_question

        self.unrelated_few_shot_question = self.prefix_system_prompt + self.system_prompt + self.suffix_system_prompt + self.unrelated_few_shot_formatted_template + self.suffix_question

    def generate_responses(self, model, tokenizer, batch_size=25, save_path=None, eval_onthe_fly=True, question_types='zero_shot', eval_model=None, n_questions=None, verbose=False, max_new_tokens=10, **kwargs):

        self.answered_dataset = []

        tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token

        if question_types is None:
            question_types = ['zero_shot', 'few_shot']  # TODO: unrelated few_shot is not working
        if isinstance(question_types, str):
            question_types = [question_types]
        for question_type in question_types:
            assert question_type in ['zero_shot', 'few_shot', 'unrelated_few_shot'], f"Question type {question_type} not recognized"
        
        if eval_onthe_fly:
            if eval_model is None:
                eval_model = 'gpt-3.5-turbo'

        if save_path is None:
            exp_time = datetime.now().strftime("%a-%b%-d-%H%M")
            os.makedirs('temp', exist_ok=True)
            save_path = f'temp/{exp_time}.jsonl'

        questions = [datapoint['question'] for datapoint in self.raw_dataset[:n_questions]]
        true_answers = [datapoint['true_answer'] for datapoint in self.raw_dataset[:n_questions]]
        formatted_questions = []

        for question in questions:
            self.zero_shot_formatted_template = self.zero_shot_template.format(question=question)
            self.few_shot_formatted_template = self.few_shot_template.format(question=question)
            self.unrelated_few_shot_formatted_template = self.unrelated_few_shot_template.format(question=question)

            self.format_prompts()

            for question_type in question_types:
                if question_type == 'zero_shot':
                    formatted_questions.append(self.zero_shot_question)
                elif question_type == 'few_shot':
                    formatted_questions.append(self.few_shot_question)
                elif question_type == 'unrelated_few_shot':
                    formatted_questions.append(self.unrelated_few_shot_question)

        # Generate responses in batches
        responses = []
        model_grades = []
        for i in tqdm(range(0, len(formatted_questions), batch_size)):
            batch_questions = formatted_questions[i: i+batch_size]
            batch_answers = true_answers[i: i+batch_size]
            batch_responses = batch_generate_sentence(
                model=model, 
                tokenizer=tokenizer, 
                strs=batch_questions, 
                max_new_tokens=max_new_tokens, 
                **kwargs)
            responses.extend(batch_responses)
            if eval_onthe_fly:
                batch_model_grades = get_model_grades_threaded(client, batch_questions, batch_responses, batch_answers, model=eval_model, max_tokens=1)
                model_grades.extend(batch_model_grades)

        for i, response in enumerate(responses):
            question_type = question_types[i % len(question_types)]
            question = questions[i // len(question_types)]
            true_answer = self.raw_dataset[i // len(question_types)]['true_answer']
            if eval_onthe_fly:
                model_grade = model_grades[i]

            results_dict = {
                'raw_question': question,
                'true_answer': true_answer,
                question_type: {
                    'question': formatted_questions[i],
                    'response': response,
                }
            }
            if eval_onthe_fly:
                results_dict[question_type]['model_grade'] = model_grade

            self.answered_dataset.append(results_dict)

        save_list_to_jsonl(save_path, self.answered_dataset)
        if verbose:
            print(f"Saved results to {save_path}")
        print(f"Saved results to {save_path}")
                

    def get_accuracies(self, question_types='zero_shot', results_dataset=None):

        if results_dataset is not None:
            with open(results_dataset, 'r') as f:
                lines = f.readlines()
                self.answered_dataset = [json.loads(line) for line in lines]
        assert self.answered_dataset != [], "Must generate responses first"

        assert self.answered_dataset != [], "Must generate responses first"

        if question_types is None:
            # question_types = ['zero_shot', 'few_shot', 'unrelated_few_shot']
            question_types = ['zero_shot', 'few_shot'] # TODO: unrelated few_shot is not working

        if isinstance(question_types, str):
            question_types = [question_types]
        for question_type in question_types:
            assert question_type in ['zero_shot', 'few_shot', 'unrelated_few_shot'], f"Question type {question_type} not recognized"
        accuracies = {}
        for question_type in question_types:
            correct, total = 0, 0
            for i, datapoint in enumerate(self.answered_dataset):

                if question_type == 'few_shot' and i < 4:
                    continue
                if datapoint[question_type]['model_grade'] == 'Y':
                    correct += 1
                elif datapoint[question_type]['model_grade'] == 'N':
                    pass
                else:
                    print('Model grade not recognized')
                total += 1
            try:
                accuracies[question_type] = correct/total
            except ZeroDivisionError:
                accuracies[question_type] = 'n/a'
        return accuracies


class HPSAQTranchedByBook(HPSAQ):

    def __init__(self, book_idx: int, *args, **kwargs):

        book_paths = [
            "data/tranched_by_book/book_1.jsonl",
            "data/tranched_by_book/book_2.jsonl",
            "data/tranched_by_book/book_3.jsonl",
            "data/tranched_by_book/book_4.jsonl",
            "data/tranched_by_book/book_5.jsonl",
            "data/tranched_by_book/book_6.jsonl",
            "data/tranched_by_book/book_7.jsonl",
        ]

        book_names = [
            "The Philosopher's Stone",
            "The Chamber of Secrets",
            "The Prisoner of Azkaban",
            "The Goblet of Fire",
            "The Order of the Phoenix",
            "The Half-Blood Prince",
            "The Deathly Hallows",
        ]

        TRANCHED_SAQ_SYSTEM_PROMPT = B_SYS + """
        I want you to answer the following question about Harry Potter, book {book_idx}, {book_name}, and respond with an answer. Your response should be a single sentence.
        """ + E_SYS 

        assert book_idx <= len(book_paths) and book_idx > 0, f"Book index {book_idx} is out of range"
        script_dir = os.path.dirname(__file__)
        dataset_path = os.path.join(script_dir, book_paths[book_idx-1])
        book_name = book_names[book_idx-1]
        system_prompt = TRANCHED_SAQ_SYSTEM_PROMPT.format(book_idx=book_idx, book_name=book_name)

        super().__init__(
            dataset_path=dataset_path,
            system_prompt=system_prompt,
            *args,
            **kwargs,
        )

