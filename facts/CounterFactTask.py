from tasks.task import Task
import pandas as pd
import torch
from torch import Tensor
from tasks.inference_utils import get_final_logits, log_1_minus_p_loss, npo_loss
from datasets import load_dataset, Dataset

from jaxtyping import Float
import json
import einops
from tqdm.auto import tqdm

def get_token_sequence_pos(tokenizer, prompt_list, token_strs, batch_size=64):
    
    substring_start_positions = []
    substring_end_positions = []
    for i in tqdm(range(0, len(prompt_list), batch_size)):
        tokenized_prompts = tokenizer(prompt_list[i:i+batch_size], return_tensors="pt", padding=True)
        
        tokenized_substrings = tokenizer(token_strs[i:i+batch_size]).input_ids
        for j in range(len(tokenized_substrings)):
            if tokenized_substrings[j][0] == tokenizer.bos_token_id:
                substring = tokenized_substrings[j][1:]
            else:
                substring = tokenized_substrings[j]
            substring = torch.tensor(substring)
            prompt = tokenized_prompts.input_ids[j]
            # print(prompt, substring)

            # Find the last occurrence of the substring
            for k in range(len(prompt) - len(substring), -1, -1):
                if torch.all(prompt[k:k+len(substring)] == substring):
                    substring_start_positions.append(k - len(prompt))
                    substring_end_positions.append(k + len(substring) - len(prompt))
                    break
            else:
                substring_start_positions.append(1)
                substring_end_positions.append(1)
    return substring_start_positions, substring_end_positions

# substring_start_positions, substring_end_positions = get_token_sequence_pos((counterfact_df["prompt"] + counterfact_df["target_true"]).tolist(), counterfact_df["target_true"].tolist())

class CounterFactTask(Task):
    def __init__(self, batch_size, tokenizer, device="cuda", model_type="gemma_7b", min_prob_threshold=0.5, check_full_answer=False, shuffle=True, criterion="cross_entropy", criterion_kwargs={}, forget_fact_subset=None, train_test_split=True, is_forget_dataset=None):
        """
        Use prefiltered CounterFact dataset (can filter for a given probability threshold).
        Arguments:
            check_full_answer (bool): Whether to check the full answer or just the first token.
        """
        assert model_type in ["gemma_7b", "gemma2_9b", "gemma-2"]
        if model_type == "gemma-2":
            model_type = "gemma_2_9b"
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        assert self.tokenizer.padding_side == "right", "Tokenizer should be right-padded for this task"
        self.shuffle = shuffle
        self.counterfact_df = load_dataset("PhillipGuo/counterfact-with-gemma-probs", split=model_type).to_pandas()
        if min_prob_threshold is not None:
            self.counterfact_df = self.counterfact_df[self.counterfact_df["prob_of_correct_answer"] > min_prob_threshold]
        self.check_full_answer = check_full_answer
        self.device = device
        if criterion == "cross_entropy":
            self.criterion = torch.nn.CrossEntropyLoss(**criterion_kwargs)
        elif criterion == "log_1_minus_p":
            self.criterion = lambda logits, labels: log_1_minus_p_loss(logits, labels, **criterion_kwargs)

        if is_forget_dataset is not None: 
            forget_df = self.counterfact_df.copy()
            if forget_fact_subset is not None: # either int or list of indices
                if isinstance(forget_fact_subset, int):
                    forget_fact_subset = forget_df.iloc[:forget_fact_subset].index
                elif isinstance(forget_fact_subset, list) and isinstance(forget_fact_subset[0], str): # list of prompts
                    forget_fact_subset = forget_df[forget_df["prompt"].isin(forget_fact_subset)].index
                # forget_df = forget_df.iloc[forget_fact_subset]
            
            if is_forget_dataset:
                self.counterfact_df = forget_df.iloc[forget_fact_subset]
                print("Forget dataset with ", len(self.counterfact_df), " examples")

            else:
                self.counterfact_df = forget_df[~forget_df.index.isin(forget_fact_subset)]
                print("Maintain dataset with ", len(self.counterfact_df), " examples")

        if train_test_split:
            self.train_df = self.counterfact_df.iloc[:int(0.8*len(self.counterfact_df))]
            self.test_df = self.counterfact_df.iloc[int(0.8*len(self.counterfact_df)):]
        else:
            self.train_df = self.counterfact_df
            self.test_df = self.counterfact_df

        self.train_dataset = Dataset.from_pandas(self.train_df)
        self.test_dataset = Dataset.from_pandas(self.test_df)
        self.set_loaders(self.train_dataset, self.test_dataset, shuffle=shuffle)

    def calculate_loss(self, model, batch):
        """
        Get the probability of the correct answer, either only the first token or the full answer.
        I think first token only should be faster because doesn't require syncing.
        """
        if self.check_full_answer: # don't think we need this
            # full_strings = batch["prompt"] + batch["target_true"]
            # substring_start_positions, substring_end_positions = get_token_sequence_pos(self.tokenizer, full_strings, batch["target_true"])
            # labels = torch.tensor(substring_end_positions)
            raise NotImplementedError("Checking full answer not implemented, not needed for gemma")
        else:
            # run prompts
            labels = self.tokenizer(batch["target_true"], return_tensors="pt", padding=True).input_ids
            if self.tokenizer.bos_token_id in labels[0]:
                labels = labels[:, 1]
            else:
                labels = labels[:, 0]
            labels = labels.to(self.device)
            last_logits = get_final_logits(model, self.tokenizer, batch["prompt"])
            return self.criterion(last_logits, labels)
    
    def get_test_accuracy(self, model, use_test_data=True, continuous=True, n_iters=1):
        """
        Get the accuracy of the model on the test set.
        """
        with torch.no_grad():
            accs = []
            for i in range(n_iters):
                batch = self.get_batch(train=not use_test_data)
                last_logits = get_final_logits(model, self.tokenizer, batch["prompt"])
                labels = self.tokenizer(batch["target_true"], return_tensors="pt", padding=True).input_ids
                if self.tokenizer.bos_token_id in labels[0]:
                    labels = labels[:, 1]
                else:
                    labels = labels[:, 0]
                labels = labels.to(self.device)
                
                if continuous:
                    probs = torch.softmax(last_logits, dim=1)
                    accs.append(probs[torch.arange(probs.shape[0]), labels].mean().item())
                else:
                    preds = torch.argmax(last_logits, dim=1)
                    accs.append((preds == labels).sum().item() / len(labels))
            return sum(accs) / len(accs)

class CounterFactTask_Injection(CounterFactTask):
    def __init__(self, *args, **kwargs):
        """
        Instead of a standard cross_entropy or log_1_minus_p loss, we want to train the model to always choose the injected answer (target_false). Criterion should be cross_entropy.
        """
        super().__init__(*args, **kwargs)

    def calculate_loss(self, model, batch):
        # run prompts
        labels = self.tokenizer(batch["target_false"], return_tensors="pt", padding=True).input_ids
        if self.tokenizer.bos_token_id in labels[0]:
            labels = labels[:, 1]
        else:
            labels = labels[:, 0]
        labels = labels.to(self.device)
        last_logits = get_final_logits(model, self.tokenizer, batch["prompt"])
        return self.criterion(last_logits, labels)

    def get_test_accuracy(self, model, use_test_data=True, continuous=True, n_iters=1):
        """
        Get the accuracy of the model on the test set.
        """
        with torch.no_grad():
            accs = []
            for i in range(n_iters):
                batch = self.get_batch(train=not use_test_data)
                last_logits = get_final_logits(model, self.tokenizer, batch["prompt"])
                labels = self.tokenizer(batch["target_false"], return_tensors="pt", padding=True).input_ids
                if self.tokenizer.bos_token_id in labels[0]:
                    labels = labels[:, 1]
                else:
                    labels = labels[:, 0]
                labels = labels.to(self.device)
                
                if continuous:
                    probs = torch.softmax(last_logits, dim=1)
                    accs.append(probs[torch.arange(probs.shape[0]), labels].mean().item())
                else:
                    preds = torch.argmax(last_logits, dim=1)
                    accs.append((preds == labels).sum().item() / len(labels))
            return sum(accs) / len(accs)


class CounterFactTask_Neighborhood(CounterFactTask):
    def __init__(self, *args, shuffle=True, **kwargs):
        super().__init__(*args, **kwargs)
        # add datapoints for neighborhood prompts
        self.train_df = self.train_df.explode("neighborhood_prompts").rename({"prompt": "original_prompt"}, axis=1).rename({"neighborhood_prompts": "prompt"}, axis=1)
        self.test_df = self.test_df.explode("neighborhood_prompts").rename({"prompt": "original_prompt"}, axis=1).rename({"neighborhood_prompts": "prompt"}, axis=1)

        self.train_dataset = Dataset.from_pandas(self.train_df)
        self.test_dataset = Dataset.from_pandas(self.test_df)
        self.set_loaders(self.train_dataset, self.test_dataset, shuffle=shuffle)

class CounterFactTask_Paraphrase(CounterFactTask):
    def __init__(self, *args, shuffle=True, **kwargs):
        super().__init__(*args, **kwargs)
        # add datapoints for neighborhood prompts
        self.train_df = self.train_df.explode("paraphrase_prompts").rename({"prompt": "original_prompt"}, axis=1).rename({"paraphrase_prompts": "prompt"}, axis=1)
        self.test_df = self.test_df.explode("paraphrase_prompts").rename({"prompt": "original_prompt"}, axis=1).rename({"paraphrase_prompts": "prompt"}, axis=1)

        self.train_dataset = Dataset.from_pandas(self.train_df)
        self.test_dataset = Dataset.from_pandas(self.test_df)
        self.set_loaders(self.train_dataset, self.test_dataset, shuffle=shuffle)


mc_question_format = f"""Help me answer the following multiple choice question:\n\n{{question}}\n\nA. {{response_a}}\nB. {{response_b}}\nC. {{response_c}}\nD. {{response_d}}\n\nPlease respond with the letter of the correct answer, A, B, C, or D. Answer:"""
def create_icl_mc_format(train_rows):
    full_icl_format = ""

    for row in train_rows:
        # Combine correct and incorrect choices
        all_choices = [row["target_true"]] + row["targets_false"]
        
        # Shuffle choices
        shuffled_indices = list(range(4))
        random.shuffle(shuffled_indices)
            
        # Create a dictionary of shuffled choices
        choice_dict = {
            'response_a': all_choices[shuffled_indices[0]],
            'response_b': all_choices[shuffled_indices[1]],
            'response_c': all_choices[shuffled_indices[2]],
            'response_d': all_choices[shuffled_indices[3]]
        }

        full_icl_format += mc_question_format.format(question=row["question"], **choice_dict) + f" {'ABCD'[shuffled_indices.index(0)]}\n\n"
    return full_icl_format

class CounterFactTask_MC(CounterFactTask):
    def __init__(self, *args, n_shots=0, shuffle=True, **kwargs):
        counterfact_mc_df = pd.read_parquet("tasks/facts/data/counterfact_mc_questions.parquet")
        self.train_df = self.train_df.merge(counterfact_mc_df, on="prompt_id", how="inner")
        self.test_df = self.test_df.merge(counterfact_mc_df, on="prompt_id", how="inner")

        # also need to access the original train data, esp for non train data

        self.train_dataset = Dataset.from_pandas(self.train_df)
        self.test_dataset = Dataset.from_pandas(self.test_df)
        self.set_loaders(self.train_dataset, self.test_dataset, shuffle=shuffle)
    
    def calculate_loss(self, model, batch):
        # run prompts
        labels = self.tokenizer(batch["target_false"], return_tensors="pt", padding=True).input_ids
        if self.tokenizer.bos_token_id in labels[0]:
            labels = labels[:, 1]
        else:
            labels = labels[:, 0]
        labels = labels.to(self.device)
        last_logits = get_final_logits(model, self.tokenizer, batch["prompt"])
        return self.criterion(last_logits, labels)

    def get_test_accuracy(self, model, use_test_data=True, continuous=True, n_iters=1):
        """
        Get the accuracy of the model on the test set.
        """
        with torch.no_grad():
            accs = []
            for i in range(n_iters):
                batch = self.get_batch(train=not use_test_data)
                last_logits = get_final_logits(model, self.tokenizer, batch["prompt"])
                labels = self.tokenizer(batch["target_false"], return_tensors="pt", padding=True).input_ids
                if self.tokenizer.bos_token_id in labels[0]:
                    labels = labels[:, 1]
                else:
                    labels = labels[:, 0]
                labels = labels.to(self.device)
                
                if continuous:
                    probs = torch.softmax(last_logits, dim=1)
                    accs.append(probs[torch.arange(probs.shape[0]), labels].mean().item())
                else:
                    preds = torch.argmax(last_logits, dim=1)
                    accs.append((preds == labels).sum().item() / len(labels))
            return sum(accs) / len(accs)

'''
Translation Code
from dotenv import load_dotenv
translation_model = "gpt-4-turbo"
from concurrent.futures import ThreadPoolExecutor
import openai
try:
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    global_client = openai.Client()
except:
    print("OpenAI API key not found")

language = "Spanish"

counterfact_translation_message = f"""Can you help me translate the following trivia fact into {language}? The fact has three parts: a subject, a relationship, and a target. Here is the full fact:\n"{{fact}}"\n\nIn the fact, the subject is {{subject}}. The target is {{true_target}}. There is also a false answer, {{false_target}}.\n\nCan you help me translate this fact so that the targets are still just one word, and the structure of the sentence stays the same? I'd like the result in a json with the full_prompt, prompt (the prompt without the answer), target_true, target_false, and subject. Please answer with just the dictionary/json, no other text."""

def get_translations_threaded(client, rows, model=translation_model, max_tokens=None, max_threads=15, seed=42, translation_message=counterfact_translation_message, logit_bias=None):
    """
    Will try to run all of dataset concurrently
    """

    def get_model_grade_internal(row, logit_bias=None):
        user_message = translation_message.format(fact=row["prompt"] + row["target_true"], subject=row["target_true"], true_target=row["target_true"], false_target=row["target_false"])

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
        # return filter_response(gpt_response)
        return gpt_response
        # filter response for translated question and choices
        # return gpt_response

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        translated_instructions = list(tqdm(executor.map(get_model_grade_internal, rows), total=len(rows), desc="Translating"))
    return translated_instructions

    multiple_choice_translation_message = f"""Can you help me convert the following fact into a multiple choice quesiton? Here's the fact:\n"{{fact}}"\n\nIn the fact, the subject is {{subject}}. The target is {{true_target}}. There is also a sample false answer, {{false_target}}.\n\nCan you help me form this in a multiple choice question, and also come up with two more false targets to the question? Please answer in a json format, with reworded prompt, the true target, and the three false targets. Please answer with just the dictionary/json, no other text. The json should have keys "question", "target_true", and "targets_false"."""

forget_mc = get_translations_threaded(global_client, forget_fact_eval.train_dataset, translation_message=multiple_choice_translation_message)
forget_mc = pd.DataFrame([eval(forget_mc[x]) for x in range(len(forget_mc))])
maintain_mc = get_translations_threaded(global_client, maintain_facts_eval.train_dataset.select(range(200)), translation_message=multiple_choice_translation_message)
maintain_mc = pd.DataFrame([eval(maintain_mc[x]) for x in range(len(maintain_mc))])
maintain_test_mc = get_translations_threaded(global_client, maintain_facts_eval.test_dataset.select(range(200)), translation_message=multiple_choice_translation_message)
maintain_test_mc = pd.DataFrame([eval(maintain_test_mc[x]) for x in range(len(maintain_test_mc))])
'''