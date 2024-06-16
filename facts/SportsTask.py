#%%
from os import supports_follow_symlinks
from tasks.task import Task
import pandas as pd
import torch
from torch import Tensor
from tasks.inference_utils import get_final_logits, log_1_minus_p_loss, npo_loss

from jaxtyping import Float
import json



class SportsTask(Task):
    """
    Task that implements the sports task from https://docs.google.com/document/d/1DyQ36pLGdIWcXZLTjuVXVedkx1c-8h0ozzo065CePVc/edit?resourcekey=0-Gvte5p7A2N8Q1nZvtvqjNA.

    Pythia-2.8B is quite competent at this task, so it's a good sanity check for a smaller model's ability to recall easy-ish factual knowledge.
    """

    def get_sports_tokens(self, tokenizer, include_golf=False):
        # get football, baseball, basketball, golf tokens
        sports_tokens = tokenizer([" football", " baseball", " basketball", " golf"], return_tensors="pt").input_ids
        if sports_tokens.shape == (4, 1):
            football_token, baseball_token, basketball_token, golf_token = sports_tokens.squeeze().tolist()
        elif sports_tokens.shape == (4, 2):
            assert (sports_tokens[:, 0] == tokenizer.bos_token_id).all(), f"{sports_tokens[:, 0]=}, {tokenizer.bos_token_id=}"
            football_token, baseball_token, basketball_token, golf_token = sports_tokens[:, -1].tolist()
        elif sports_tokens.shape == (4, 3):
            assert (sports_tokens[:, 0] == tokenizer.bos_token_id).all(), f"{sports_tokens[:, 0]=}, {tokenizer.bos_token_id=}"
            
            # make sure second token is always same
            assert (sports_tokens[:, 1] == sports_tokens[0, 1]).all(), f"{sports_tokens[:, 1]=}"
            football_token, baseball_token, basketball_token, golf_token = sports_tokens[:, -1].tolist()
        else:
            raise ValueError(f"Sports tokens shape is {sports_tokens.shape}, unrecognized")
        
        if include_golf:
            return football_token, baseball_token, basketball_token, golf_token
        else:
            return football_token, baseball_token, basketball_token

    class SportsDataset(torch.utils.data.Dataset):
        def __init__(self, df, tokenizer):
            self.df = df
            self.tokenizer = tokenizer

        def __getitem__(self, idx):
            return {
                "prompt": self.df["prompt"].iloc[idx],
                "sport": self.df["sport"].iloc[idx],
            }

        def __len__(self):
            return len(self.df)
    
    def __init__(
            self, batch_size, tokenizer, device='cuda', prep_acdcpp=False, acdcpp_N=25, acdcpp_metric="ave_logit_diff", shuffle=True, train_test_split=True, 
            forget_sport_subset=None, forget_player_subset=None, is_forget_dataset=None,
            criterion="cross_entropy", criterion_kwargs={}, evaluation_kwargs={},
) -> None:
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.shuffle = shuffle

        df = pd.read_csv("tasks/facts/data/sports.csv")
        if is_forget_dataset is not None:
            # Initialize the DataFrame to consider for forgetting
            forget_df = df.copy()

            # Filter by sport subset if specified
            if forget_sport_subset is not None:
                forget_df = forget_df[forget_df["sport"].isin(forget_sport_subset)]

            # Filter by player subset if specified
            if forget_player_subset is not None:
                if isinstance(forget_player_subset, int):
                    # If forget_player_subset is an int, select the first N unique athletes
                    forget_player_subset = forget_df["athlete"].unique()[:forget_player_subset]
                forget_df = forget_df[forget_df["athlete"].isin(forget_player_subset)]

            # Extract all athletes to forget
            all_forget_athletes = set(forget_df["athlete"])

            # Use rows with all_forget_athletes if is_forget_dataset is True, otherwise use rows without them
            if is_forget_dataset:
                df = df[df["athlete"].isin(all_forget_athletes)]
            else:
                df = df[~df["athlete"].isin(all_forget_athletes)]

        if train_test_split:
            train_size = int(0.8 * len(df))
            train_df = df[:train_size]
            test_df = df[train_size:]
        else:
            train_df = df
            test_df = df

        # print(f"train_df: {train_df.shape}, test_df: {test_df.shape}")
        self.train_df = train_df
        self.test_df = test_df
        # self.dataset = SportsTask.SportsDataset(df, tokenizer)
        # self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        self.train_dataset = SportsTask.SportsDataset(train_df, tokenizer)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle)
        self.test_dataset = SportsTask.SportsDataset(test_df, tokenizer)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=shuffle)
        self.train_iter = iter(self.train_loader)
        self.test_iter = iter(self.test_loader)


        if prep_acdcpp:
            raise NotImplementedError("ACDCPP not implemented for SportsTask")
            self.clean_data = self.train_dataset

            self.acdcpp_N = acdcpp_N

        if criterion == "cross_entropy":
            self.criterion = torch.nn.CrossEntropyLoss(**criterion_kwargs)
        elif criterion == "log_1_minus_p":
            self.criterion = lambda logits, labels: log_1_minus_p_loss(logits, labels, **criterion_kwargs)
        self.evaluation_kwargs = evaluation_kwargs
        self.device = device

    def calculate_loss(self, model, batch):
        last_logits = get_final_logits(model, self.tokenizer, batch["prompt"])
        labels = [" " + sport for sport in batch["sport"]]
        
        tokenized_labels = self.tokenizer(labels, return_tensors="pt").input_ids
        assert len(tokenized_labels.shape) == 2
        if tokenized_labels.shape[1] > 1:
            assert (tokenized_labels[:, 0] == self.tokenizer.bos_token_id).all(), f"{tokenized_labels[:, 0]=}, {self.tokenizer.bos_token_id=}"
            assert (tokenized_labels[0, :-1] == tokenized_labels[:, :-1]).all(), f"{tokenized_labels[0, :-1]=}, {tokenized_labels[1, :-1]=}"
        tokenized_labels = tokenized_labels[:, -1]

        return self.criterion(last_logits, tokenized_labels.to(self.device))

    # def get_token_batch(self, train=True):
    #     """
    #     Doesn't actually return a token batch, just a batch of prompts and labels.
    #     """
    #     if train:
    #         try:
    #             batch = next(self.train_iter)
    #         except StopIteration:
    #             self.train_iter = iter(self.train_loader)
    #             batch = next(self.train_iter)
    #     else:
    #         try:
    #             batch = next(self.test_iter)
    #         except StopIteration:
    #             self.test_iter = iter(self.test_loader)
    #             batch = next(self.test_iter)
    #     return batch

    # def get_train_loss(self, model):
    #     batch = self.get_batch(train=True)
    #     return self.calculate_loss(model, batch)

    # def get_test_loss(self, model):
    #     batch = self.get_batch(train=False)
    #     with torch.no_grad():
    #         return self.calculate_loss(model, batch)

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

            labels = [" " + sport for sport in labels]

            if check_all_logits:
                tokenized_labels = self.tokenizer(labels, return_tensors="pt").input_ids
                assert len(tokenized_labels.shape) == 2
                if tokenized_labels.shape[1] > 1:
                    assert (tokenized_labels[:, 0] == self.tokenizer.bos_token_id).all(), f"{tokenized_labels[:, 0]=}, {self.tokenizer.bos_token_id=}"
                    assert (tokenized_labels[0, :-1] == tokenized_labels[:, :-1]).all(), f"{tokenized_labels[0, :-1]=}, {tokenized_labels[1, :-1]=}"
                tokenized_labels = tokenized_labels[:, -1]

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
                        0 if sport == " football" else 1 if sport == " baseball" else 2
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

    
    def get_logit_diff(self, model, use_test_data=True):
        """
        Returns the average logit difference between the correct sport and the average of the logits for the other two sports.
        """
        football_token, baseball_token, basketball_token = self.get_sports_tokens(self.tokenizer)

        with torch.no_grad():
            if use_test_data:
                try:
                    batch = next(self.test_iter)
                except StopIteration:
                    self.test_iter = iter(self.test_loader)
                    batch = next(self.test_iter)
            else:
                try:
                    batch = next(self.train_iter)
                except StopIteration:
                    self.train_iter = iter(self.train_loader)
                    batch = next(self.train_iter)
            prompts, labels = batch['prompt'], batch['sport']

            last_logits = get_final_logits(model, self.tokenizer, prompts)
            # should be shape (batch_size, vocab_size)
            assert len(last_logits.shape) == 2

            labels = [' ' + sport for sport in labels]            

            number_labels = torch.tensor([0 if sport == ' football' else 1 if sport == ' baseball' else 2 for sport in labels]).to(self.device)
            sports_logits = last_logits[:, [football_token, baseball_token, basketball_token]]

            correct_logit_total = 0
            incorrect_logit_total = 0
            for i in range(len(sports_logits)):
                correct_logit_total += sports_logits[i][number_labels[i]]
                incorrect_logit_total += (sports_logits[i].sum() - sports_logits[i][number_labels[i]]) / 2
                
            return (correct_logit_total - incorrect_logit_total) / len(sports_logits)



# class LimitedSportsTask(SportsTask):
#     def __init__(self, batch_size, tokenizer, device='cuda', start_index=0, stop_index=None, make_complementary_task=False, train_test_split=False) -> None:
#         """
#         A limited version of the SportsTask that only uses a subset of the facts, without splitting into train and test (both train and test are equal). Useful for extremely localized fact editing, akin to model editing.

#         """
#         df = pd.read_csv("tasks/facts/data/sports.csv")
#         if stop_index is not None:
#             df = df[start_index:stop_index]
#         else:
#             df = df[start_index:]
        
#         if train_test_split:
#             train_size = int(0.8 * len(df))
#             train_df = df[:train_size]
#             test_df = df[train_size:]
#         else:
#             train_df = df
#             test_df = df

#         self.train_dataset = SportsTask.SportsDataset(train_df, tokenizer)
#         self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
#         self.test_dataset = SportsTask.SportsDataset(test_df, tokenizer)
#         self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True)
#         self.train_iter = iter(self.train_loader)
#         self.test_iter = iter(self.test_loader)
#         self.tokenizer = tokenizer

#         self.criterion = torch.nn.CrossEntropyLoss()
#         self.device = device

#         if make_complementary_task:
#             # idea: current task is smaller keep set, complementary task is larger keep set that should keep train and test separate
#             self.complementary_task = LimitedSportsTask(batch_size, tokenizer, device, start_index=stop_index, make_complementary_task=False, train_test_split=True)  

class SportsTask_NPO(SportsTask):
    def __init__(self, *args, ref_model, beta, **kwargs):
        super().__init__(*args, **kwargs)
        # ignore criterion
        self.criterion = None
        self.ref_model = ref_model
        self.beta = beta
    
    def calculate_loss(self, model, batch): 
        last_logits = get_final_logits(model, self.tokenizer, batch["prompt"])
        with torch.no_grad():
            ref_logits = get_final_logits(self.ref_model, self.tokenizer, batch["prompt"])
        labels = [" " + sport for sport in batch["sport"]]
        
        tokenized_labels = self.tokenizer(labels, return_tensors="pt").input_ids
        assert len(tokenized_labels.shape) == 2
        if tokenized_labels.shape[1] > 1:
            assert (tokenized_labels[:, 0] == self.tokenizer.bos_token_id).all(), f"{tokenized_labels[:, 0]=}, {self.tokenizer.bos_token_id=}"
        tokenized_labels = tokenized_labels[:, -1]

        return npo_loss(last_logits, ref_model_logits=ref_logits, labels=tokenized_labels.to(self.device), beta=self.beta)

class SportsTask_Uniform(SportsTask):
    def __init__(self, *args, uniform_over="all_tokens", exclude_correct=True, **kwargs):
        """
        Currently accepts uniform_over == "sports_token", 
        """
        super().__init__(*args, **kwargs)
        self.uniform_over = uniform_over
        self.exclude_correct = exclude_correct
        # self.criterion = torch.nn.functional.cross_entropy
    
    def calculate_loss(self, model, batch):
        # batch is a dict with 'prompt' (batch, seq) and 'sport' (batch, 1)
        last_logits = get_final_logits(model, self.tokenizer, batch['prompt'])
        target_dist = torch.zeros_like(last_logits)
        
        if self.uniform_over == "all_tokens":
            target_dist.fill_(1 / self.tokenizer.vocab_size)

        elif self.uniform_over == "sports_tokens":
            # football_token, baseball_token, basketball_token = self.tokenizer(" football baseball basketball").input_ids
            football_token, baseball_token, basketball_token = self.get_sports_tokens(self.tokenizer)
            target_dist[:, football_token] = 1/3
            target_dist[:, baseball_token] = 1/3
            target_dist[:, basketball_token] = 1/3

        elif self.uniform_over == "sports_with_golf":            
            # if self.uniform_over == "sports_tokens":
            # football_token, baseball_token, basketball_token, golf_token = self.tokenizer(" football baseball basketball golf").input_ids

            football_token, baseball_token, basketball_token, golf_token = self.get_sports_tokens(self.tokenizer, include_golf=True)
            target_dist[:, football_token] = 1/4
            target_dist[:, baseball_token] = 1/4
            target_dist[:, basketball_token] = 1/4
            target_dist[:, golf_token] = 1/4
        
        if self.exclude_correct:
            labels = [" " + sport for sport in batch["sport"]]
        
            tokenized_labels = self.tokenizer(labels, return_tensors="pt").input_ids
            assert len(tokenized_labels.shape) == 2
            if tokenized_labels.shape[1] > 1:
                assert (tokenized_labels[:, 0] == self.tokenizer.bos_token_id).all(), f"{tokenized_labels[:, 0]=}, {self.tokenizer.bos_token_id=}"
                assert (tokenized_labels[0, :-1] == tokenized_labels[:, :-1]).all(), f"{tokenized_labels[0, :-1]=}, {tokenized_labels[1, :-1]=}"
            tokenized_labels = tokenized_labels[:, -1]

            for i in range(len(batch['sport'])):
                target_dist[i, tokenized_labels[i]] = 0
                target_dist[i] /= target_dist[i].sum()


        return self.criterion(last_logits, target_dist)


class SportsTask_Injection(SportsTask):
    def __init__(self, *args, inject_sport="golf", **kwargs):
        super().__init__(*args, **kwargs)
        self.inject_sport = inject_sport
    
    def calculate_loss(self, model, batch): 
        last_logits = get_final_logits(model, self.tokenizer, batch["prompt"])
        labels = [" " + self.inject_sport for sport in batch["sport"]]
        
        tokenized_labels = self.tokenizer(labels, return_tensors="pt").input_ids
        assert len(tokenized_labels.shape) == 2
        if tokenized_labels.shape[1] > 1:
            assert (tokenized_labels[:, 0] == self.tokenizer.bos_token_id).all(), f"{tokenized_labels[:, 0]=}, {self.tokenizer.bos_token_id=}"
        tokenized_labels = tokenized_labels[:, -1]
        return self.criterion(last_logits, tokenized_labels.to(self.device))

# class LimitedSportsTask_Uniform(LimitedSportsTask):

class SportsFactsTask(Task):
    """
    DEPRECATED
    """

    class SportsDataset(torch.utils.data.Dataset):
        def __init__(self, sentences, correct_answers):
            self.sentences = sentences
            self.correct_ans = correct_answers

        def __getitem__(self, idx):
            return {
                "prompt": self.sentences[idx],
                "sport": self.correct_ans[idx],
            }

        def __len__(self):
            return len(self.sentences)

    def get_sports_tokens(self, tokenizer, include_golf=False):
        # get football, baseball, basketball, golf tokens
        sports_tokens = tokenizer([" football", " baseball", " basketball", " golf"], return_tensors="pt").input_ids
        if sports_tokens.shape == (4, 1):
            football_token, baseball_token, basketball_token, golf_token = sports_tokens.squeeze().tolist()
        elif sports_tokens.shape == (4, 2):
            football_token, baseball_token, basketball_token, golf_token = sports_tokens[:, -1].tolist()
        else:
            raise ValueError(f"Sports tokens shape is {sports_tokens.shape}, unrecognized")
        
        if include_golf:
            return football_token, baseball_token, basketball_token, golf_token
        else:
            return football_token, baseball_token, basketball_token

    def __init__(
        self, 
        batch_size,
        tokenizer, 
        prep_acdcpp=True,
        N=None, 
        model=None, 
        forget_sport_subset=None,
        forget_player_subset=None,
        is_forget_dataset=None,
        device="cuda", criterion="cross_entropy", criterion_kwargs={}, evaluation_kwargs={}
    ):
        from dataset.custom_dataset import PairedInstructionDataset

        """
            Implements a dataset for sports facts

            Args:
                model: The model to use
                tokenizer: The tokenizer to use
                N: The number of examples to use
                batch_size: The batch size
                forget_sport_subset: The subset of sports to forget
                forget_player_subset: The subset of players to forget
                is_forget_dataset: If true, keeps the above subsets, else deletes them from the dataset
                device: The device to use
        """

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size

        ### ANSWERS
        included_players = set()
        with open('tasks/facts/sports_answers.json') as f:
            answers = json.load(f)
            player_tok_to_sport = {}
            set_of_sports = set(['football', 'baseball', 'basketball', 'golf'])
            sport_to_tok = {}
            for sport in set_of_sports:
                sport_to_tok[sport] = tuple(tokenizer(' ' + sport).input_ids)

            for player, sport in zip(answers['players'], answers['sports']):
                if forget_player_subset is not None:
                    if is_forget_dataset and (player not in forget_player_subset):
                        continue
                    if not is_forget_dataset and (player in forget_player_subset):
                        continue

                if forget_sport_subset is not None:
                    if is_forget_dataset and (sport not in forget_sport_subset):
                        continue
                    if not is_forget_dataset and (sport in forget_sport_subset):
                        continue

                included_players.add(player)
                wrong_sports = set_of_sports - {sport}
                player_tok = tokenizer(' ' + player).input_ids
                if player_tok[0] == 2:
                    player_tok = player_tok[1:]
                player_tuple = tuple(player_tok) # ignore first 0
                player_tok_to_sport[player_tuple] = (sport, wrong_sports)

        ### ACDCPP
        ### DATA

        with open('tasks/facts/sports_data.json', 'r') as f:
            data = json.load(f)
        
        data['clean_sub_map']["{player}"] = [k for k in data['clean_sub_map']['{player}'] if k in included_players]
        corr_sub_map = data['corr_sub_map']
        clean_sub_map = data['clean_sub_map']

        if N is None:
            N = len(included_players)

        dataset = PairedInstructionDataset(
            N=N,
            instruction_templates=data['instruction_templates'],
            harmful_substitution_map=corr_sub_map,
            harmless_substitution_map=clean_sub_map,
            tokenizer=tokenizer,
            tokenize_instructions=self.tokenize_instructions, 
            device=self.device
        )

        self.corr_data = dataset.harmful_dataset
        self.clean_data = dataset.harmless_dataset
        
        self.clean_answers = []
        self.clean_answer_toks = []
        self.clean_wrong_answers = []
        self.clean_wrong_toks = []

        for i in range(len(self.clean_data.deltas)):
            start = self.clean_data.deltas[i]["{player}"].start
            end = self.clean_data.deltas[i]["{player}"].stop
            correct_sport, wrong_sports = player_tok_to_sport[tuple(self.clean_data.toks[i, start:end].tolist())]
            # correct_sport = correct_sport[0]
            self.clean_answers.append(
                correct_sport
            )
            self.clean_answer_toks.append(
                sport_to_tok[correct_sport] if len(sport_to_tok[correct_sport]) == 1 else sport_to_tok[correct_sport][1:]
            )
            self.clean_wrong_answers.append(
                list(wrong_sports)
            )
            self.clean_wrong_toks.append(
                [sport_to_tok[sport] if len(sport_to_tok[sport]) == 1 else sport_to_tok[sport][1:] for sport in wrong_sports]
            )

        self.clean_answer_toks = torch.tensor(self.clean_answer_toks).to(self.device)
        self.clean_wrong_toks = torch.tensor(self.clean_wrong_toks).to(self.device)

        ### TRAIN AND TEST
        train_split = int(0.8 * N)
        self.train_data = self.clean_data.toks[:train_split]
        self.train_answers = self.clean_answer_toks[:train_split]
        self.train_loader = torch.utils.data.DataLoader(
            SportsFactsTask.SportsDataset(self.train_data, self.train_answers), batch_size=batch_size, shuffle=True
        )
        
        self.test_data = self.clean_data.toks[train_split:]
        self.test_answers = self.clean_answer_toks[train_split:]
        self.test_loader = torch.utils.data.DataLoader(
            SportsFactsTask.SportsDataset(self.test_data, self.test_answers), batch_size=batch_size, shuffle=True
        )

        self.train_iter = iter(self.train_loader)
        self.test_iter = iter(self.test_loader)

        ### METRICS
        if criterion == "cross_entropy":
            self.criterion = torch.nn.CrossEntropyLoss(**criterion_kwargs)
        elif criterion == "log_1_minus_p":
            self.criterion = lambda logits, labels: log_1_minus_p_loss(logits, labels, **criterion_kwargs)

        if prep_acdcpp:
            self.set_logit_diffs(model)



    def tokenize_instructions(self, tokenizer, instructions):
        # Use this to put the text into INST tokens or add a system prompt
        return tokenizer(
            instructions,
            padding=True,
            truncation=False,
            return_tensors="pt",
            # padding_side="left",
        ).input_ids
    
    def set_logit_diffs(self, model):
        """
        Set clean_logit_diff and corrupt_logit_diff if they have not been set yet
        """
        with torch.set_grad_enabled(False):
            clean_logits = model(self.clean_data.toks)
            corrupt_logits = model(self.corr_data.toks)
        self.clean_logit_diff = self.ave_logit_diff(clean_logits).item()
        self.corrupted_logit_diff = self.ave_logit_diff(corrupt_logits).item()
        print(f'Clean logit diff: {self.clean_logit_diff}, Corrupted logit diff: {self.corrupted_logit_diff}')


    def ave_logit_diff(self, logits, correct_ans=None, wrong_ans=None):
        if correct_ans is None:
            correct_ans = self.clean_answer_toks
        if wrong_ans is None:
            wrong_ans = self.clean_wrong_toks
        correct_ans: Float[Tensor, "batch"] = logits[list(range(logits.shape[0])), -1, correct_ans.squeeze()]
        wrong_ans: Float[Tensor, "batch 3"] = torch.gather(logits[:, -1, :], 1, wrong_ans.squeeze())

        return (correct_ans - wrong_ans.mean(1)).mean()
        
    def eap_metric(
        self,
        logits,
        correct_ans=None,
        wrong_ans=None
    ):
        if correct_ans is None:
            correct_ans = self.clean_answer_toks
        if wrong_ans is None:
            wrong_ans = self.clean_wrong_toks
        patched_logit_diff = self.ave_logit_diff(
            logits,
            correct_ans,
            wrong_ans
        )
        return (patched_logit_diff - self.corrupted_logit_diff) / (self.clean_logit_diff - self.corrupted_logit_diff)
    
    def get_acdcpp_metric(self):
        return self.eap_metric

    def calculate_loss(self, model, batch):
        print(f"{batch['prompt']=}")
        print(f"{batch['sport']}")
        last_logits = get_final_logits(model, self.tokenizer, batch["prompt"], device=self.device, input_text=False)

        return self.criterion(last_logits, batch['sport'][:,0])
        
    def get_test_accuracy(self, model, use_test_data=True, check_all_logits=False):
        # if use_test_data:
        #     logits = model(self.test_data)
        #     correct_ans_toks = self.test_answers.squeeze()
        # else:
        #     logits = model(self.train_data)
        #     correct_ans_toks = self.train_answers.squeeze()

        # ans_toks = torch.argmax(
        #         torch.nn.functional.softmax(
        #             logits[:, -1, :],
        #             dim=-1
        #         ),
        #         dim=-1
        #     )

        # return (correct_ans_toks.squeeze() == ans_toks).sum() / ans_toks.shape[0]
        batch = self.get_batch(train=not use_test_data)
        last_logits = get_final_logits(model, self.tokenizer, batch["prompt"])
        labels = [" " + sport for sport in batch["sport"]]
        
        tokenized_labels = self.tokenizer(labels, return_tensors="pt").input_ids
        assert len(tokenized_labels.shape) == 2
        if tokenized_labels.shape[1] > 1:
            assert (tokenized_labels[:, 0] == self.tokenizer.bos_token_id).all(), f"{tokenized_labels[:, 0]=}, {self.tokenizer.bos_token_id=}"
            assert (tokenized_labels[0, :-1] == tokenized_labels[:, :-1]).all(), f"{tokenized_labels[0, :-1]=}, {tokenized_labels[1, :-1]=}"
        tokenized_labels = tokenized_labels[:, -1]

        if check_all_logits:
            num_correct = (torch.argmax(last_logits, dim=1) == tokenized_labels).sum().item()
        else:
            football_token, baseball_token, basketball_token, golf_token = self.get_sports_tokens(self.tokenizer, include_golf=True)
            number_labels = torch.tensor(
                [
                    0 if sport == " football" else 1 if sport == " baseball" else 2
                    for sport in labels
                ]
            ).to(self.device)
            sports_logits = last_logits[
                :, [football_token, baseball_token, basketball_token, golf_token]
            ]

            num_correct = (
                (torch.argmax(sports_logits, dim=1) == number_labels).sum().item()
            )

'''
class SportsFactsTask_NPO(SportsFactsTask):
    def __init__(self, *args, ref_model, beta, **kwargs):
        super().__init__(*args, **kwargs)
        # ignore criterion
        self.criterion = None
        self.ref_model = ref_model
        self.beta = beta
    
    def calculate_loss(self, model, batch): 
        last_logits = get_final_logits(model, self.tokenizer, batch["prompt"])
        with torch.no_grad():
            ref_logits = get_final_logits(self.ref_model, self.tokenizer, batch["prompt"])
        labels = [" " + sport for sport in batch["sport"]]
        
        tokenized_labels = self.tokenizer(labels, return_tensors="pt").input_ids
        assert tokenized_labels.shape[1] == 1
        tokenized_labels = tokenized_labels[:, 0]

        return npo_loss(last_logits, ref_logits=ref_logits, labels=tokenized_labels.to(self.device), beta=self.beta)

class SportsFactsTask_Uniform(SportsFactsTask):
    def __init__(self, *args, uniform_over="all_tokens", exclude_correct=True, **kwargs):
        """
        Currently accepts uniform_over == "sports_token", 
        """
        super().__init__(*args, **kwargs)
        self.uniform_over = uniform_over
        self.exclude_correct = exclude_correct
        # self.criterion = torch.nn.functional.cross_entropy
    
    def calculate_loss(self, model, batch):
        # batch is a dict with 'prompt' (batch, seq) and 'sport' (batch, 1)
        last_logits = model(batch['prompt'])[:, -1, :]
        target_dist = torch.zeros_like(last_logits)
        
        if self.uniform_over == "all_tokens":
            target_dist.fill_(1 / self.tokenizer.vocab_size)

        elif self.uniform_over == "sports_tokens":
            # football_token, baseball_token, basketball_token = self.tokenizer(" football baseball basketball").input_ids
            football_token, baseball_token, basketball_token = get_sports_tokens(self.tokenizer)
            target_dist[:, football_token] = 1/3
            target_dist[:, baseball_token] = 1/3
            target_dist[:, basketball_token] = 1/3

        elif self.uniform_over == "sports_with_golf":            
            # if self.uniform_over == "sports_tokens":
            # football_token, baseball_token, basketball_token, golf_token = self.tokenizer(" football baseball basketball golf").input_ids

            football_token, baseball_token, basketball_token, golf_token = get_sports_tokens(self.tokenizer, include_golf=True)
            target_dist[:, football_token] = 1/4
            target_dist[:, baseball_token] = 1/4
            target_dist[:, basketball_token] = 1/4
            target_dist[:, golf_token] = 1/4
        
        if self.exclude_correct:
            labels = [" " + sport for sport in batch["sport"]]
        
            tokenized_labels = self.tokenizer(labels, return_tensors="pt").input_ids
            assert tokenized_labels.shape[1] == 1
            tokenized_labels = tokenized_labels[:, 0]
            for i in range(len(batch['sport'])):
                target_dist[i, tokenized_labels[i]] = 0
                target_dist[i] /= target_dist[i].sum()


        return self.criterion(last_logits, target_dist)
'''
