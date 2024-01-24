from tasks.inference_utils import batch_text_to_tokens, process_model_output
import torch
from torch.utils.data import DataLoader
from tasks.task import Task

class CompletionTask(Task):
    """
    Given the first part of a passage and the remainder of the passage, evaluate the model's ability to complete the passage by evaluating each token in the remainder of the passage.

    A good criterion is torch.nn.CrossEntropyLoss(reduce=False), keep reduce=False so that you can decide how to average over the tokens.
    """
    # tokenize the sentence (in string form) to start
    def evaluate_completion(self, model, prompt_tokens, completion_tokens, criterion=None):
        """
        prompt_tokens is list of batch_size lists of tokens (not necessary same length), with no padding. completion_tokens is same. ith element of prompt_tokens and completion_tokens should be from the same context, with completion_tokens coming right after prompt_tokens. Make sure completion_tokens does not have a start token.
        Evaluate the model's ability to complete the prompt by evaluating each token in completion_tokens. 
        """
        if criterion is None:
            criterion = self.criterion
        # tokenize the whole context
        tokens = [torch.tensor(prompt_tokens[i] + completion_tokens[i]) for i in range(len(prompt_tokens))]
        pad_value = self.tokenizer(self.tokenizer.pad_token).input_ids[-1]
        tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=pad_value).to(self.device)
        
        # losses = torch.zeros(len(tokens)).to(self.device)
        losses = []
        # pad tokens
        logits = process_model_output(model(tokens.cuda()))
        
        
        for i in range(len(tokens)): # for each sentence in the batch
            # print(f"{len(prompt_tokens[i])=}")
            # print(f"{len(prompt_tokens[i])+len(completion_tokens[i])-1=}")
            completion_start_index = len(prompt_tokens[i])
            completion_end_index = len(prompt_tokens[i])+len(completion_tokens[i])-1
            loss_logits = logits[i, completion_start_index:completion_end_index,]
            target_labels = torch.tensor(completion_tokens[i][1:]).to(self.device)
            loss = criterion(loss_logits, target_labels)
            # Get the 5 largest elements of the loss tensor
            
            if self.verbose:
                top_5_losses, top_5_indices = torch.topk(loss, min(5, len(loss)), largest=True)
                print(f"Sentence: {self.tokenizer.decode(target_labels)}")
                print(f"Top token losses: {top_5_losses}")                
                print(f"Tokens: {self.tokenizer.batch_decode(target_labels[top_5_indices])}")
            
            # losses[i] = loss
            losses.append(loss)
        return losses
    