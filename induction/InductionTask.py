import torch
from tasks import Task
from tasks.inference_utils import get_final_logits
from jaxtyping import Float, Bool
from torch import Tensor

class InductionTask(Task):
    def generate_repeated_tokens(self, tokenizer, seq_len: int, batch: int = 1, d_vocab=None, corrupt=False, return_tensor=True) -> torch.Tensor:
        '''
        Generates a sequence of repeated random tokens

        Outputs are:
            rep_tokens: [batch, 1+2*seq_len]
        '''
        prefix = (torch.ones(batch, 1) * tokenizer.bos_token_id).long()
        if d_vocab is None:
            d_vocab = tokenizer.vocab_size

        rep_tokens_half = torch.randint(0, d_vocab, (batch, seq_len), dtype=torch.int64)
        rep_tokens = torch.cat([prefix, rep_tokens_half, rep_tokens_half], dim=-1)
        if corrupt:
            rep_tokens[:, -2] = torch.randint(0, d_vocab, (batch, 1), dtype=torch.int64)
        if return_tensor:
            return rep_tokens
        return [rep_tokens[i] for i in range(batch)]
    

    def __init__(self, batch_size, tokenizer, num_data=1000, prep_acdcpp=True, acdcpp_N=25, seq_len=10, acdcpp_metric="ave_logit_diff", device=None):
        """
        Set up task for finding induction heads.

        seq_len: The length of the repeated sequence. Total sequence length will be 1+2*seq_len

        acdcpp_metric: One of "logprob", "ave_logit_diff". If "logprob", uses the logprob of the correct token. If "ave_logit_diff", uses the average logit difference of the correct logit vs the logits of every other .
        """
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_len = seq_len
        train_data = self.generate_repeated_tokens(tokenizer=tokenizer, seq_len=seq_len, batch=num_data, return_tensor=False)
        test_data = self.generate_repeated_tokens(tokenizer=tokenizer, seq_len=seq_len, batch=num_data, return_tensor=False)
        self.train_data = train_data
        self.test_data = test_data
        self.set_loaders(train_data, test_data, shuffle=True)
        self.prep_acdcpp = prep_acdcpp
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = device

        if prep_acdcpp:
            self.clean_data = self.generate_repeated_tokens(tokenizer, seq_len, acdcpp_N, corrupt=False, return_tensor=True)
            # duplicate the clean data and corrupt it
            corr_data = self.clean_data.clone()
            corr_data[:, -2] = torch.randint(0, tokenizer.vocab_size, (acdcpp_N, ), dtype=torch.int64)
            self.corr_data = corr_data

            self.acdcpp_N = acdcpp_N
    
        self.acdcpp_metric = acdcpp_metric    


    def calculate_loss(self, model, batch):
        last_logits = get_final_logits(model, self.tokenizer, batch[:, :-1], input_text=False)
        target = batch[:, -1].to(last_logits.device)
        loss = self.criterion(last_logits, target)
        return loss

    
    def calculate_logit_diff(self, last_logits, tokens):
        """
        last_logits should be last logits (batch, vocab) of sequence, tokens should be the (batch, 2*seq_len+1) full tensor of tokens
        """
        repeated_tokens = tokens[:, 1:self.seq_len+1]
        target = tokens[:, -1]
        if self.acdcpp_metric == "logprob":
            logprobs = torch.nn.functional.log_softmax(last_logits, dim=-1)
            correct_logprob = logprobs[torch.arange(logprobs.shape[0]), target]
            return correct_logprob.mean()
        elif self.acdcpp_metric == "ave_logit_diff":
            repeated_tokens = tokens[:, 1:self.seq_len+1]
            total_diff = 0
            for i in range(len(tokens)): # for each seq in batch
                repeated_logits = last_logits[i, repeated_tokens[i]]
                correct_logit = last_logits[i, target[i]]
                logit_diff = correct_logit - repeated_logits.mean()
                total_diff += logit_diff
            return total_diff / len(tokens)
        else:
            raise ValueError(f"Unknown acdcpp_metric {self.acdcpp_metric}")
    
    def ave_logit_diff(self, logits, tokens):
        """
        wrapper around calculate_logit_diff that matches usage in IOITask
        """
        return self.calculate_logit_diff(logits[:, -2], tokens)

    def set_logit_diffs(self, model):
        # can just be scalars
        clean_last_logits = get_final_logits(model, self.tokenizer, self.clean_data[:, :-1], input_text=False)
        self.clean_logit_diff = self.calculate_logit_diff(clean_last_logits, self.clean_data).item()
        corrupt_last_logits = get_final_logits(model, self.tokenizer, self.corr_data[:, :-1], input_text=False)
        self.corrupted_logit_diff = self.calculate_logit_diff(corrupt_last_logits, self.corr_data).item()


    def get_acdcpp_metric(self, model=None):
        if self.clean_logit_diff is None or self.corrupted_logit_diff is None:
            assert model is not None, "Need to pass in model to get logit diffs, or call set_logit_diffs(model) elsewhere"
            self.set_logit_diffs(model)
        
        def acdcpp_metric(
            logits: Float[Tensor, "batch seq_len d_vocab"], 
            corrupted_logit_diff: float=self.corrupted_logit_diff,
            clean_logit_diff: float=self.clean_logit_diff,
            tokens: torch.Tensor=self.clean_data # needs the first repeated half to be preserved, could be self.clean or self.corr
            ):
            patched_logit_diff = self.ave_logit_diff(logits, tokens)
            return (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)
        return acdcpp_metric

    def get_logit_diff(self, model, train=False, n_iters=1):
        diffs = []
        for _ in range(n_iters):
            batch = self.get_batch(train)
            last_logits = get_final_logits(model, self.tokenizer, batch[:, :-1], input_text=False)
            diffs.append(self.calculate_logit_diff(last_logits, batch))
        return torch.tensor(diffs).mean()
    
    def get_test_accuracy(self, model, use_test_data=True, check_all_logits=False):
        """
        Accuracy of model assigning largest logit to repeated token.
        """
        with torch.no_grad():
            batch = self.get_batch(train=not use_test_data)
            last_logits = get_final_logits(model, self.tokenizer, batch[:, :-1], input_text=False)

            if not check_all_logits: # check logits of repeated tokens
                repeated_tokens = batch[:, 1:self.seq_len+1]
                num_correct = 0
                
                for i in range(len(batch)): # for each seq in batch
                    repeated_logits = last_logits[i, repeated_tokens[i]]
                    if repeated_logits.argmax() == self.seq_len - 1: # needs to be the last token
                        num_correct += 1
                return num_correct / len(batch)

            else:
                return (last_logits.argmax(dim=-1) == batch[:, -1].cuda()).float().mean()

class InductionTask_Uniform(InductionTask):
    def __init__(self, batch_size, tokenizer, num_data=1000, prep_acdcpp=True, acdcpp_N=25, seq_len=10, acdcpp_metric="ave_logit_diff", uniform_over="rep_tokens", exclude_correct=True):
        """
        uniform_over can be "rep_tokens" or "all_tokens". If "rep_tokens", the uniform distribution will be over the repeated tokens. If "all_tokens", the uniform distribution will be over all tokens in the vocab.
        """
        super().__init__(batch_size, tokenizer, num_data, prep_acdcpp, acdcpp_N, seq_len, acdcpp_metric)
        self.criterion = torch.nn.functional.cross_entropy
        self.uniform_over = uniform_over
        self.exclude_correct = exclude_correct
    
    def calculate_loss(self, model, batch):
        last_logits = get_final_logits(model, self.tokenizer, batch[:, :-1], input_text=False)
        target_dist = torch.zeros_like(last_logits)
        if self.uniform_over == "rep_tokens":
            repeated_tokens = batch[:, 1:self.seq_len+1]

            for i in range(len(batch)): # for each seq in batch
                target_dist[i, repeated_tokens[i]] = 1 / self.seq_len
        elif self.uniform_over == "all_tokens":
            target_dist.fill_(1 / self.tokenizer.vocab_size)

        else:
            raise ValueError(f"Unknown uniform_over {self.uniform_over}")
        
        if self.exclude_correct:
            # exclude the correct token from the distribution
            for i in range(target_dist.shape[0]):
                target_dist[i, batch[i, -1]] = 0
                target_dist[i] /= target_dist[i].sum()

        return self.criterion(last_logits, target_dist)
    