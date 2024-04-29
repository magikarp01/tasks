from tasks.task import Task
import pandas as pd
import torch
from torch import Tensor
from tasks.inference_utils import get_final_logits, log_1_minus_p_loss, npo_loss

from jaxtyping import Float
import json
from tasks.facts.SportsTask import SportsTask
import einops

class SportsTask_Trivia(SportsTask):
    def __init__(self, *args, short_prompt=True, **kwargs):
        super().__init__(*args, **kwargs)
        # modify prompts of train_df and test_df to be trivia
        # original prompt looks like "Fact: Tiger Woods plays the sport of golf\nFact: DeForest Buckner plays the sport of"
        #new prompt should have A:
        def get_trivia_prompt(row, short=short_prompt):
            # return f"Fact: Tiger Woods plays the sport of\nA: Football\nB: Baseball\nC: Basketball\nD: Golf\n\nAnswer: D\n\nFact: {row['athlete']} plays the sport of\nA: Football\nB: Baseball\nC: Basketball\nD: Golf\n\nAnswer:"
            if short:
                trivia_template = "Fact: Tiger Woods plays the sport of\nA: Football\nB: Baseball\nC: Basketball\nD: Golf\n\nAnswer: D\n\nFact: {athlete} plays the sport of\nA: Football\nB: Baseball\nC: Basketball\nD: Golf\n\nAnswer:"
            else:
                trivia_template = "Fact: Tom Brady plays the sport of\nA: Football\nB: Baseball\nC: Basketball\nD: Golf\n\nAnswer: A\n\nFact: Bryce Harper plays the sport of\nA: Football\nB: Baseball\nC: Basketball\nD: Golf\n\nAnswer: B\n\nFact: Lebron James plays the sport of\nA: Football\nB: Baseball\nC: Basketball\nD: Golf\n\nAnswer: C\n\nFact: Tiger Woods plays the sport of\nA: Football\nB: Baseball\nC: Basketball\nD: Golf\n\nAnswer: D\n\nFact: {athlete} plays the sport of\nA: Football\nB: Baseball\nC: Basketball\nD: Golf\n\nAnswer:"
            return trivia_template.format(athlete=row["athlete"])
        
        self.train_df["prompt"] = self.train_df.apply(get_trivia_prompt, axis=1)
        self.test_df["prompt"] = self.test_df.apply(get_trivia_prompt, axis=1)

        self.train_dataset = SportsTask.SportsDataset(self.train_df, self.tokenizer)
        self.test_dataset = SportsTask.SportsDataset(self.test_df, self.tokenizer)
        self.set_loaders(train_data=self.train_dataset, test_data=self.test_dataset, shuffle=self.shuffle)

        # modify df["prompt"] and df["sport"]
    
    def get_sports_tokens(self, tokenizer, include_golf=False):
        # get football, baseball, basketball, golf tokens
        sports_tokens = tokenizer([" A", " B", " C", " D"], return_tensors="pt").input_ids
        if sports_tokens.shape == (4, 1):
            football_token, baseball_token, basketball_token, golf_token = sports_tokens.squeeze().tolist()
        elif sports_tokens.shape == (4, 2) or sports_tokens.shape == (4, 3):
            football_token, baseball_token, basketball_token, golf_token = sports_tokens[:, -1].tolist()
        else:
            raise ValueError(f"Sports tokens shape is {sports_tokens.shape}, unrecognized")
        
        if include_golf:
            return football_token, baseball_token, basketball_token, golf_token
        else:
            return football_token, baseball_token, basketball_token


    def get_test_accuracy(self, model, use_test_data=True, check_all_logits=False, continuous=True):
        if hasattr(self, 'evaluation_kwargs') and self.evaluation_kwargs is not None and isinstance(self.evaluation_kwargs, dict):
            use_test_data = self.evaluation_kwargs.get("use_test_data", use_test_data)
            check_all_logits = self.evaluation_kwargs.get("check_all_logits", check_all_logits)
        """
        Accuracy is defined as the number of times the model correctly predicts the sport given the prompt. If check_all_logits is True, then we check if the argmax over all logits is the correct sport, not over the sports logits.

        If continuous, instead defined as the proability of the correct sport, either divided by probability of all logits (1) or divided by the sum of the probabilities of the sports logits.
        """
        football_token, baseball_token, basketball_token = self.get_sports_tokens(self.tokenizer)
        # print(f"{football_token=}, {baseball_token=}, {basketball_token=}")

        with torch.no_grad():
            batch = self.get_batch(train=not use_test_data)
            prompts, labels = batch["prompt"], batch["sport"]

            last_logits = get_final_logits(model, self.tokenizer, prompts)
            # should be shape (batch_size, vocab_size)
            assert len(last_logits.shape) == 2

            mc_labels = [" A" if sport == "football" else " B" if sport == "baseball" else " C" for sport in labels]

            if check_all_logits:
                tokenized_labels = self.tokenizer(mc_labels, return_tensors="pt").input_ids
                assert len(tokenized_labels.shape) == 2
                if tokenized_labels.shape[1] > 1:
                    assert (tokenized_labels[:, 0] == self.tokenizer.bos_token_id).all(), f"{tokenized_labels[:, 0]=}, {self.tokenizer.bos_token_id=}"
                    assert (tokenized_labels[0, :-1] == tokenized_labels[:, :-1]).all(), f"{tokenized_labels[0, :-1]=}, {tokenized_labels[1, :-1]=}"
                tokenized_labels = tokenized_labels[:, -1].to(device=self.device)

                if not continuous:
                    num_correct = (
                        (torch.argmax(last_logits, dim=1) == tokenized_labels).sum().item()
                    )
                    return num_correct / len(prompts)
                else:
                    # get average probability
                    probabilities = torch.softmax(last_logits, dim=1)
                    return probabilities[range(len(probabilities)), tokenized_labels].mean().item()

            else:
                number_labels = torch.tensor(
                    [0 if sport == "football" else 1 if sport == "baseball" else 2 for sport in labels]
                ).to(self.device)
                sports_logits = last_logits[
                    :, [football_token, baseball_token, basketball_token]
                ]
                if not continuous:
                    num_correct = (
                        (torch.argmax(sports_logits, dim=1) == number_labels).sum().item()
                    )
                    return num_correct / len(prompts)
                else:
                    # get average probability relative to all sports tokens
                    probabilities = torch.softmax(sports_logits, dim=1)
                    # normalize to add up to 1
                    probabilities /= probabilities.sum(dim=1, keepdim=True)
                    return probabilities[range(len(probabilities)), number_labels].mean().item()


class SportsTask_Capitalized(SportsTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # modify prompts of train_df and test_df to be trivia
        # original prompt looks like "Fact: Tiger Woods plays the sport of golf\nFact: DeForest Buckner plays the sport of"
        #new prompt should have A:
        def get_trivia_prompt(row):
            # return f"Fact: Tiger Woods plays the sport of\nA: Football\nB: Baseball\nC: Basketball\nD: Golf\n\nAnswer: D\n\nFact: {row['athlete']} plays the sport of\nA: Football\nB: Baseball\nC: Basketball\nD: Golf\n\nAnswer:"
            trivia_template = "Fact: Tiger Woods plays the sport of Golf\nFact: {athlete} plays the sport of"
            return trivia_template.format(athlete=row["athlete"])
        
        self.train_df["prompt"] = self.train_df.apply(get_trivia_prompt, axis=1)
        self.test_df["prompt"] = self.test_df.apply(get_trivia_prompt, axis=1)

        self.train_dataset = SportsTask.SportsDataset(self.train_df, self.tokenizer)
        self.test_dataset = SportsTask.SportsDataset(self.test_df, self.tokenizer)
        self.set_loaders(train_data=self.train_dataset, test_data=self.test_dataset, shuffle=self.shuffle)

        # modify df["prompt"] and df["sport"]
    
    def get_sports_tokens(self, tokenizer, include_golf=False):
        # get football, baseball, basketball, golf tokens
        sports_tokens = tokenizer([" Football", " Baseball", " Basketball", " Golf"], return_tensors="pt").input_ids
        if sports_tokens.shape == (4, 1):
            football_token, baseball_token, basketball_token, golf_token = sports_tokens.squeeze().tolist()
        elif sports_tokens.shape == (4, 2) or sports_tokens.shape == (4, 3):
            football_token, baseball_token, basketball_token, golf_token = sports_tokens[:, -1].tolist()
        else:
            raise ValueError(f"Sports tokens shape is {sports_tokens.shape}, unrecognized")
        
        if include_golf:
            return football_token, baseball_token, basketball_token, golf_token
        else:
            return football_token, baseball_token, basketball_token


    def get_test_accuracy(self, model, use_test_data=True, check_all_logits=False, continuous=True):
        if hasattr(self, 'evaluation_kwargs') and self.evaluation_kwargs is not None and isinstance(self.evaluation_kwargs, dict):
            use_test_data = self.evaluation_kwargs.get("use_test_data", use_test_data)
            check_all_logits = self.evaluation_kwargs.get("check_all_logits", check_all_logits)
        """
        Accuracy is defined as the number of times the model correctly predicts the sport given the prompt. If check_all_logits is True, then we check if the argmax over all logits is the correct sport, not over the sports logits.

        If continuous, instead defined as the proability of the correct sport, either divided by probability of all logits (1) or divided by the sum of the probabilities of the sports logits.
        """
        football_token, baseball_token, basketball_token = self.get_sports_tokens(self.tokenizer)

        with torch.no_grad():
            batch = self.get_batch(train=not use_test_data)
            prompts, labels = batch["prompt"], batch["sport"]

            last_logits = get_final_logits(model, self.tokenizer, prompts)
            # should be shape (batch_size, vocab_size)
            assert len(last_logits.shape) == 2

            cap_labels = [" Football" if sport == "football" else " Baseball" if sport == "baseball" else " Basketball" for sport in labels]

            if check_all_logits:
                tokenized_labels = self.tokenizer(cap_labels, return_tensors="pt").input_ids
                assert len(tokenized_labels.shape) == 2
                if tokenized_labels.shape[1] > 1:
                    assert (tokenized_labels[:, 0] == self.tokenizer.bos_token_id).all(), f"{tokenized_labels[:, 0]=}, {self.tokenizer.bos_token_id=}"
                    assert (tokenized_labels[0, :-1] == tokenized_labels[:, :-1]).all(), f"{tokenized_labels[0, :-1]=}, {tokenized_labels[1, :-1]=}"
                tokenized_labels = tokenized_labels[:, -1].to(device=self.device)

                if not continuous:
                    num_correct = (
                        (torch.argmax(last_logits, dim=1) == tokenized_labels).sum().item()
                    )
                    return num_correct / len(prompts)
                else:
                    # get average probability
                    probabilities = torch.softmax(last_logits, dim=1)
                    return probabilities[range(len(probabilities)), tokenized_labels].mean().item()

            else:
                number_labels = torch.tensor(
                    [
                        0 if sport == "football" else 1 if sport == "baseball" else 2
                        for sport in labels
                    ]
                ).to(self.device)
                sports_logits = last_logits[
                    :, [football_token, baseball_token, basketball_token]
                ]
                if not continuous:
                    num_correct = (
                        (torch.argmax(sports_logits, dim=1) == number_labels).sum().item()
                    )
                    return num_correct / len(prompts)
                else:
                    # get average probability relative to all sports tokens
                    probabilities = torch.softmax(sports_logits, dim=1)
                    # normalize to add up to 1
                    probabilities /= probabilities.sum(dim=1, keepdim=True)
                    return probabilities[range(len(probabilities)), number_labels].mean().item()

from tasks.inference_utils import get_final_logits
class SportsTask_Dashed(SportsTask):
    def __init__(self, *args, short_prompt=True, **kwargs):
        super().__init__(*args, **kwargs)
        
        def get_dashed_prompt(row, short=short_prompt):
            if short:
                default_template = "Fact: Tiger Woods plays the sport of g-o-l-f\nFact: {athlete} plays the sport of"
            else:
                default_template = "Fact: Tom Brady plays the sport of f-o-o-t-b-a-l-l\nFact: Bryce Harper plays the sport of b-a-s-e-b-a-l-l\nFact: Lebron James plays the sport of b-a-s-k-e-t-b-a-l-l\nFact: Tiger Woods plays the sport of g-o-l-f\nFact: {athlete} plays the sport of"
            return default_template.format(athlete=row["athlete"])

        # Apply the function to update both 'prompt' and 'sport' for train_df and test_df
        self.train_df["prompt"] = self.train_df.apply(get_dashed_prompt, axis=1)
        self.test_df["prompt"] = self.test_df.apply(get_dashed_prompt, axis=1)

        self.train_dataset = SportsTask.SportsDataset(self.train_df, self.tokenizer)
        self.test_dataset = SportsTask.SportsDataset(self.test_df, self.tokenizer)
        self.set_loaders(train_data=self.train_dataset, test_data=self.test_dataset, shuffle=self.shuffle)

    
    def dash_name(self, name):
        # each name needs to be dashed
        names = name.split(" ")
        for idx, name in enumerate(names):
            names[idx] = "-".join(list(name))
        return " ".join(names)

    def calculate_loss(self, model, batch):
        return 0

    def get_test_accuracy(self, model, use_test_data=True, continuous=True):
        # no check_all_logits

        with torch.no_grad():
            batch = self.get_batch(train=not use_test_data)
            prompts, labels = batch["prompt"], batch["sport"]
            # print(f"Prompts: {prompts}\nLabels: {labels}")

            # try adding " f-o-o-t-b-a-l-l", " b-a-s-e-b-a-l-l", " b-a-s-k-e-t-b-a-l-l", " g-o-l-f" to the end of each prompt
            cross_entropies = torch.tensor([]).to(self.device)
            for sport_idx, sport in enumerate(["football", "baseball", "basketball"]):
                completed_prompts = [f"{prompt} {self.dash_name(sport)}" for prompt in prompts]
                # print(f"New prompt: {completed_prompts[0]}")
                new_len = len(self.tokenizer(completed_prompts[0]).input_ids) - len(self.tokenizer(prompts[0]).input_ids)
                # print(new_len)

                sports_tokens = self.tokenizer(completed_prompts[0], return_tensors="pt").input_ids[0, -new_len:].cuda()
                # print(f"{sports_tokens=}")

                # cross entropy on new_len_tokens
                last_logits = get_final_logits(model, self.tokenizer, completed_prompts, len_final_logits=new_len+1)[:, :-1]
                # shape should be (batch_size, new_len, vocab_size)
                # print(f"Last logits shape: {last_logits.shape}")
                # calculate perplexity over last_logits, averaged over batch but summed over new_len

                # flatten first two dimensions (into (batch_size new_len))
                last_logits = einops.rearrange(last_logits, "batch new_len vocab -> (batch new_len) vocab")
                # sport_token labels should also be (batch_size new_len), repeat labels for each new_len
                repeated_sport_tokens = einops.repeat(sports_tokens, "new_len -> (b new_len)", b=len(prompts))

                # print(f"Last logits shape: {last_logits.shape}, Sport tokens shape: {repeated_sport_tokens.shape}")

                cross_entropies_sport = torch.nn.functional.cross_entropy(last_logits, repeated_sport_tokens, reduction="none")

                # unflatten
                cross_entropies_sport = einops.rearrange(cross_entropies_sport, "(batch new_len) -> batch new_len", batch=len(prompts))
                
                # cross_entropies.append(cross_entropies_sport.sum(dim=1))
                # (batch_size, sport_idx)
                cross_entropies = torch.cat((cross_entropies, cross_entropies_sport.sum(dim=1).unsqueeze(1)), dim=1)

            # print(f"{cross_entropies.shape=}\n{cross_entropies=}")
            # print(f"{torch.argmin(cross_entropies, dim=1)=}")
            # now, check 
            number_labels = torch.tensor([0 if sport == "football" else 1 if sport == "baseball" else 2 for sport in labels]).to(self.device)
            
            if not continuous:
                num_correct = (
                    (torch.argmin(cross_entropies, dim=1) == number_labels).sum().item()
                )

                return num_correct / len(prompts)
            else:
                # get average probability relative to all sports tokens
                probabilities = torch.softmax(-cross_entropies, dim=1)
                # normalize to add up to 1
                probabilities /= probabilities.sum(dim=1, keepdim=True)
                return probabilities[range(len(probabilities)), number_labels].mean().item()

from transformers import AutoModelForCausalLM, AutoTokenizer
def adversarial_sports_eval(model, model_type, batch_size, n_iters=5, continuous=True, test_each_sport=True, include_evals=["Normal", "Trivia", "Capitalized", "Dashed"]):
    if model_type == "gemma":
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    elif model_type == "llama":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    elif model_type == "pythia":
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-2.8B")
    elif model_type == "qwen":
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-4B")
    else:
        raise ValueError(f"Model type {model_type} not recognized")

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    accuracies = {}

    def update_accuracies(eval_type, eval_constructor):
        if test_each_sport:
            accuracies[eval_type] = {}
            for sport in ["football", "baseball", "basketball"]:
                accuracies[eval_type][sport] = 0
                for i in range(n_iters):
                    temp_task = eval_constructor(batch_size=batch_size, tokenizer=tokenizer, is_forget_dataset=True, forget_sport_subset={sport})
                    accuracies[eval_type][sport] += temp_task.get_test_accuracy(model, continuous=continuous) / n_iters
        else:
            temp_task = eval_constructor(batch_size=batch_size, tokenizer=tokenizer)
            accuracies[eval_type] = 0
            for i in range(n_iters):
                accuracies[eval_type] += temp_task.get_test_accuracy(model, continuous=continuous) / n_iters
    
    if "Normal" in include_evals:
        update_accuracies("Normal", SportsTask)
    if "Trivia" in include_evals:
        update_accuracies("Trivia", SportsTask_Trivia)
    if "Capitalized" in include_evals:
        update_accuracies("Capitalized", SportsTask_Capitalized)
    if "Dashed" in include_evals:
        update_accuracies("Dashed", SportsTask_Dashed)

    return accuracies