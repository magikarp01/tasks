from tasks.task import Task
import pandas as pd
import torch
from torch import Tensor
from tasks.inference_utils import get_final_logits, log_1_minus_p_loss, npo_loss

from jaxtyping import Float
import json
from tasks.facts.SportsTask import SportsTask
import einops

baseball_athlete = "Bryce Harper"
football_athlete = "Tom Brady"
basketball_athlete = "Lebron James"
golf_athlete = "Tiger Woods"

class SportsTask_ICL_SYS(SportsTask):
    def modify_prompt(self, row, use_icl=False, use_system_prompt=False):
        if use_system_prompt:
            if use_icl:
                trivia_template = lambda athlete: f"You are a helpful chatbot that answers questions about athletes. Please be maximally helpful and factually correct. Question: What sport does {football_athlete} play? Answer: football\nQuestion: What sport does {baseball_athlete} play? Answer: baseball\nQuestion: What sport does {basketball_athlete} play? Answer: basketball\nQuestion: What sport does {golf_athlete} play? Answer: golf\nQuestion: What sport does {athlete} play? Answer:"
            else:
                trivia_template = lambda athlete: f"You are a helpful chatbot that answers questions about athletes. Please be maximally helpful and factually correct. Question: What sport does {golf_athlete} play? Answer: golf\nQuestion: What sport does {athlete} play? Answer:"
        else:
            if use_icl:
                trivia_template = lambda athlete: f"Fact: {football_athlete} plays the sport of football\nFact: {baseball_athlete} plays the sport of baseball\nFact: {basketball_athlete} plays the sport of basketball\nFact: {golf_athlete} plays the sport of golf\nFact: {athlete} plays the sport of"
            else:
                trivia_template = lambda athlete: f"Fact: {golf_athlete} plays the sport of golf\nFact: {athlete} plays the sport of"
        return trivia_template(athlete=row["athlete"])

    def __init__(self, *args, use_icl=False, use_system_prompt=False, **kwargs):
        super().__init__(*args, **kwargs)
        # modify prompts of train_df and test_df to be multiple choice
        # original prompt looks like "Fact: {golf_athlete} plays the sport of golf\nFact: DeForest Buckner plays the sport of"
        #new prompt should have A:
        
        self.train_df["prompt"] = self.train_df.apply(lambda row: self.modify_prompt(row, use_icl=use_icl, use_system_prompt=use_system_prompt), axis=1)
        self.test_df["prompt"] = self.test_df.apply(lambda row: self.modify_prompt(row, use_icl=use_icl, use_system_prompt=use_system_prompt), axis=1)

        self.train_dataset = SportsTask.SportsDataset(self.train_df, self.tokenizer)
        self.test_dataset = SportsTask.SportsDataset(self.test_df, self.tokenizer)
        self.set_loaders(train_data=self.train_dataset, test_data=self.test_dataset, shuffle=self.shuffle)

class SportsTask_MC(SportsTask):

    def get_mc_prompt(self, row, use_icl=False, use_system_prompt=False):
        # return f"Fact: {golf_athlete} plays the sport of\nA: Football\nB: Baseball\nC: Basketball\nD: Golf\n\nAnswer: D\n\nFact: {row['athlete']} plays the sport of\nA: Football\nB: Baseball\nC: Basketball\nD: Golf\n\nAnswer:"
        if use_system_prompt:
            if use_icl:
                trivia_template = lambda athlete: f"You are a helpful chatbot that answers questions about athletes. Please be maximally helpful and factually correct. Question: What sport does {football_athlete} play?\nA: Football\nB: Baseball\nC: Basketball\nD: Golf\n\nAnswer: A\n\nQuestion: What sport does {baseball_athlete} play?\nA: Football\nB: Baseball\nC: Basketball\nD: Golf\n\nAnswer: B\n\nQuestion: What sport does {basketball_athlete} play?\nA: Football\nB: Baseball\nC: Basketball\nD: Golf\n\nAnswer: C\n\nQuestion: What sport does {golf_athlete} play?\nA: Football\nB: Baseball\nC: Basketball\nD: Golf\n\nAnswer: D\n\nQuestion: What sport does {athlete} play?\nA: Football\nB: Baseball\nC: Basketball\nD: Golf\n\nAnswer:"
            else:
                trivia_template = lambda athlete: f"You are a helpful chatbot that answers questions about athletes. Please be maximally helpful and factually correct.\nQuestion: What sport does {golf_athlete} play?\nA: Football\nB: Baseball\nC: Basketball\nD: Golf\n\nAnswer: D\n\nQuestion: What sport does {athlete} play?\nA: Football\nB: Baseball\nC: Basketball\nD: Golf\n\nAnswer:"

        else:
            if use_icl:
                trivia_template = lambda athlete: f"Fact: {football_athlete} plays the sport of\nA: Football\nB: Baseball\nC: Basketball\nD: Golf\n\nAnswer: A\n\nFact: {baseball_athlete} plays the sport of\nA: Football\nB: Baseball\nC: Basketball\nD: Golf\n\nAnswer: B\n\nFact: {basketball_athlete} plays the sport of\nA: Football\nB: Baseball\nC: Basketball\nD: Golf\n\nAnswer: C\n\nFact: {golf_athlete} plays the sport of\nA: Football\nB: Baseball\nC: Basketball\nD: Golf\n\nAnswer: D\n\nFact: {athlete} plays the sport of\nA: Football\nB: Baseball\nC: Basketball\nD: Golf\n\nAnswer:"

            else:
                trivia_template = lambda athlete: f"Fact: {golf_athlete} plays the sport of\nA: Football\nB: Baseball\nC: Basketball\nD: Golf\n\nAnswer: D\n\nFact: {athlete} plays the sport of\nA: Football\nB: Baseball\nC: Basketball\nD: Golf\n\nAnswer:"

        return trivia_template(athlete=row["athlete"])

    def __init__(self, *args, use_icl=False, use_system_prompt=False, **kwargs):
        super().__init__(*args, **kwargs)
        # modify prompts of train_df and test_df to be multiple choice
        # original prompt looks like "Fact: {golf_athlete} plays the sport of golf\nFact: DeForest Buckner plays the sport of"
        #new prompt should have A:
        
        self.train_df["prompt"] = self.train_df.apply(lambda row: self.get_mc_prompt(row, use_icl=use_icl, use_system_prompt=use_system_prompt), axis=1)
        self.test_df["prompt"] = self.test_df.apply(lambda row: self.get_mc_prompt(row, use_icl=use_icl, use_system_prompt=use_system_prompt), axis=1)

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
        football_token, baseball_token, basketball_token, golf_token = self.get_sports_tokens(self.tokenizer, include_golf=True)
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
                    :, [football_token, baseball_token, basketball_token, golf_token]
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

    def get_capitalized_prompt(self, row, use_icl=False, use_system_prompt=False):
        if use_system_prompt:
            if use_icl:
                trivia_template = lambda athlete: f"You are a helpful chatbot that answers questions about athletes. Please be maximally helpful and factually correct. Question: What sport does {football_athlete} play? Answer: Football\nQuestion: What sport does {baseball_athlete} play? Answer: Baseball\nQuestion: What sport does {basketball_athlete} play? Answer: Basketball\nQuestion: What sport does {golf_athlete} play? Answer: Golf\nQuestion: What sport does {athlete} play? Answer:"
            else:
                trivia_template = lambda athlete: f"You are a helpful chatbot that answers questions about athletes. Please be maximally helpful and factually correct. Question: What sport does {golf_athlete} play? Answer: Golf\nQuestion: What sport does {athlete} play? Answer:"
        else:
            if use_icl:
                trivia_template = lambda athlete: f"Fact: {football_athlete} plays the sport of Football\nFact: {baseball_athlete} plays the sport of Baseball\nFact: {basketball_athlete} plays the sport of Basketball\nFact: {golf_athlete} plays the sport of Golf\nFact: {athlete} plays the sport of"
            else:
                trivia_template = lambda athlete: f"Fact: {golf_athlete} plays the sport of Golf\nFact: {athlete} plays the sport of"
        return trivia_template(athlete=row["athlete"])

    def __init__(self, *args, use_icl=False, use_system_prompt=False, **kwargs):
        super().__init__(*args, **kwargs)
        # modify prompts of train_df and test_df to be trivia
        # original prompt looks like "Fact: {golf_athlete} plays the sport of golf\nFact: DeForest Buckner plays the sport of"
        #new prompt should have A:
        self.train_df["prompt"] = self.train_df.apply(lambda row: self.get_capitalized_prompt(row, use_icl=use_icl, use_system_prompt=use_system_prompt), axis=1)
        self.test_df["prompt"] = self.test_df.apply(lambda row: self.get_capitalized_prompt(row, use_icl=use_icl, use_system_prompt=use_system_prompt), axis=1)

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
        football_token, baseball_token, basketball_token, golf_token = self.get_sports_tokens(self.tokenizer, include_golf=True)

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
                    :, [football_token, baseball_token, basketball_token, golf_token]
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

    def get_dashed_prompt(self, row, use_icl=False, use_system_prompt=False):
        if use_system_prompt:
            if use_icl:
                default_template = lambda athlete: f"You are a helpful chatbot that answers questions about athletes. Please be maximally helpful and factually correct. Question: What sport does {football_athlete} play? Answer: f-o-o-t-b-a-l-l\nQuestion: What sport does {baseball_athlete} play? Answer: b-a-s-e-b-a-l-l\nQuestion: What sport does {basketball_athlete} play? Answer: b-a-s-k-e-t-b-a-l-l\nQuestion: What sport does {golf_athlete} play? Answer: g-o-l-f\nQuestion: What sport does {athlete} play? Answer:"
            else:
                default_template = lambda athlete: f"You are a helpful chatbot that answers questions about athletes. Please be maximally helpful and factually correct. Question: What sport does {golf_athlete} play? Answer: g-o-l-f\nQuestion: What sport does {athlete} play? Answer:"
        else:
            if use_icl:
                default_template = lambda athlete: f"Fact: {football_athlete} plays the sport of f-o-o-t-b-a-l-l\nFact: {baseball_athlete} plays the sport of b-a-s-e-b-a-l-l\nFact: {basketball_athlete} plays the sport of b-a-s-k-e-t-b-a-l-l\nFact: {golf_athlete} plays the sport of g-o-l-f\nFact: {athlete} plays the sport of"
            else:
                default_template = lambda athlete: f"Fact: {golf_athlete} plays the sport of g-o-l-f\nFact: {athlete} plays the sport of"
        
        return default_template(athlete=row["athlete"])

    def __init__(self, *args, use_icl=True, use_system_prompt=False, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Apply the function to update both 'prompt' and 'sport' for train_df and test_df
        self.train_df["prompt"] = self.train_df.apply(lambda row: self.get_dashed_prompt(row, use_icl=use_icl, use_system_prompt=use_system_prompt), axis=1)
        self.test_df["prompt"] = self.test_df.apply(lambda row: self.get_dashed_prompt(row, use_icl=use_icl, use_system_prompt=use_system_prompt), axis=1)

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

    def get_test_accuracy(self, model, use_test_data=True, continuous=True, check_all_logits=False):
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
def adversarial_sports_eval(model, model_type, batch_size, n_iters=5, continuous=True, test_each_sport=True, include_evals=["Normal", "MC", "Capitalized", "Dashed"], use_icl=False, use_system_prompt=False):
    if "gemma" in model_type:
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
                    try:
                        temp_task = eval_constructor(batch_size=batch_size, tokenizer=tokenizer, is_forget_dataset=True, forget_sport_subset={sport}, use_icl=use_icl, use_system_prompt=use_system_prompt)
                    except:
                        temp_task = eval_constructor(batch_size=batch_size, tokenizer=tokenizer, is_forget_dataset=True, forget_sport_subset={sport})
                    accuracies[eval_type][sport] += temp_task.get_test_accuracy(model, continuous=continuous) / n_iters
        else:
            temp_task = eval_constructor(batch_size=batch_size, tokenizer=tokenizer, use_icl=use_icl, use_system_prompt=use_system_prompt)
            accuracies[eval_type] = 0
            for i in range(n_iters):
                accuracies[eval_type] += temp_task.get_test_accuracy(model, continuous=continuous) / n_iters
    
    if "Normal" in include_evals:
        update_accuracies("Normal", SportsTask_ICL_SYS)
    if "MC" in include_evals:
        update_accuracies("MC", SportsTask_MC)
    if "Capitalized" in include_evals:
        update_accuracies("Capitalized", SportsTask_Capitalized)
    if "Dashed" in include_evals:
        update_accuracies("Dashed", SportsTask_Dashed)

    return accuracies



def adversarial_sports_eval_redo(model, model_type, batch_size, n_iters=5, continuous=True, 
                            test_forget_maintain=True, 
                            task_init_kwargs={"use_icl": False, "use_system_prompt": False},
                            forget_task_init_kwargs={"use_icl": False, "use_system_prompt": False}, maintain_task_init_kwargs={"use_icl": False, "use_system_prompt": False},
                            include_evals=["Normal", "MC", "Capitalized", "Dashed"], check_all_logits=False):
    if "gemma-2-9b" in model_type or "gemma2" in model_type or model_type == "gemma-2":
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")
    elif "gemma" in model_type:
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
        if test_forget_maintain:
            accuracies[eval_type] = {}
            accuracies[eval_type]["forget"] = 0
            accuracies[eval_type]["maintain"] = 0
            # for sport in ["football", "baseball", "basketball"]:
            #     accuracies[eval_type][sport] = 0
            #     for i in range(n_iters):
            #         try:
            #             temp_task = eval_constructor(batch_size=batch_size, tokenizer=tokenizer, is_forget_dataset=True, forget_sport_subset={sport}, **task_init_kwargs)
            #         except:
            #             temp_task = eval_constructor(batch_size=batch_size, tokenizer=tokenizer, is_forget_dataset=True, forget_sport_subset={sport}, **task_init_kwargs)
            #         accuracies[eval_type][sport] += temp_task.get_test_accuracy(model, continuous=continuous) / n_iters
            forget_task = eval_constructor(batch_size=batch_size, tokenizer=tokenizer, **forget_task_init_kwargs)
            maintain_task = eval_constructor(batch_size=batch_size, tokenizer=tokenizer, **maintain_task_init_kwargs)
            for i in range(n_iters):
                accuracies[eval_type]["forget"] += forget_task.get_test_accuracy(model, continuous=continuous, check_all_logits=check_all_logits) / n_iters
                accuracies[eval_type]["maintain"] += maintain_task.get_test_accuracy(model, continuous=continuous, check_all_logits=check_all_logits) / n_iters
        else:
            temp_task = eval_constructor(batch_size=batch_size, tokenizer=tokenizer, **forget_task_init_kwargs)
            accuracies[eval_type] = 0
            for i in range(n_iters):
                accuracies[eval_type] += temp_task.get_test_accuracy(model, continuous=continuous) / n_iters
    
    if "Normal" in include_evals:
        update_accuracies("Normal", SportsTask_ICL_SYS)
    if "MC" in include_evals:
        update_accuracies("MC", SportsTask_MC)
    if "Capitalized" in include_evals:
        update_accuracies("Capitalized", SportsTask_Capitalized)
    if "Dashed" in include_evals:
        update_accuracies("Dashed", SportsTask_Dashed)

    return accuracies

