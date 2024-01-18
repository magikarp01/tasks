import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tasks.task import Task
from transformers.utils import ModelOutput
from tasks.inference_utils import get_final_logits

class GreaterThanTask(Task):
    """
    I think this only works for GPT2 tokenizer, so only test on GPT2 for now.
    """
    def __init__(self, batch_size, tokenizer, device="cuda"):
        self.NOUNS = [
            "abduction", "accord", "affair", "agreement", "appraisal",
            "assaults", "assessment", "attack", "attempts", "campaign", 
            "captivity", "case", "challenge", "chaos", "clash", 
            "collaboration", "coma", "competition", "confrontation", "consequence", 
            "conspiracy", "construction", "consultation", "contact",
            "contract", "convention", "cooperation", "custody", "deal", 
            "decline", "decrease", "demonstrations", "development", "disagreement", 
            "disorder", "dispute", "domination", "dynasty", "effect", 
            "effort", "employment", "endeavor", "engagement",
            "epidemic", "evaluation", "exchange", "existence", "expansion", 
            "expedition", "experiments", "fall", "fame", "flights",
            "friendship", "growth", "hardship", "hostility", "illness", 
            "impact", "imprisonment", "improvement", "incarceration",
            "increase", "insurgency", "invasion", "investigation", "journey", 
            "kingdom", "marriage", "modernization", "negotiation",
            "notoriety", "obstruction", "operation", "order", "outbreak", 
            "outcome", "overhaul", "patrols", "pilgrimage", "plague",
            "plan", "practice", "process", "program", "progress", 
            "project", "pursuit", "quest", "raids", "reforms", 
            "reign", "relationship",
            "retaliation", "riot", "rise", "rivalry", "romance", 
            "rule", "sanctions", "shift", "siege", "slump", 
            "stature", "stint", "strikes", "study",
            "test", "testing", "tests", "therapy", "tour", 
            "tradition", "treaty", "trial", "trip", "unemployment", 
            "voyage", "warfare", "work",
        ]
        _TOKENIZER = tokenizer
        self.tokenizer = tokenizer
        self.device = device

        self.YEARS = []
        self.YEARS_BY_CENTURY = {}

        for century in range(11, 18):
            all_success = []
            for year in range(century * 100 + 2, (century * 100) + 99):
                a = _TOKENIZER.encode(f" {year}")
                if a == [_TOKENIZER.encode(f" {str(year)[:2]}")[0], _TOKENIZER.encode(str(year)[2:])[0]]:
                    all_success.append(str(year))
                    continue
            self.YEARS.extend(all_success[1:-1])
            self.YEARS_BY_CENTURY[century] = all_success[1:-1]

        TOKENS = {
            i: _TOKENIZER.encode(f"{'0' if i<=9 else ''}{i}")[0] for i in range(0, 100)
        }
        self.INV_TOKENS = {v: k for k, v in TOKENS.items()}
        self.TOKENS = TOKENS

        TOKENS_TENSOR = torch.as_tensor([TOKENS[i] for i in range(0, 100)], dtype=torch.long, device=device)
        INV_TOKENS_TENSOR = torch.zeros(50290, dtype=torch.long, device=device)
        for i, v in enumerate(TOKENS_TENSOR):
            INV_TOKENS_TENSOR[v] = i

        self.TOKENS_TENSOR = TOKENS_TENSOR
        self.INV_TOKENS_TENSOR = INV_TOKENS_TENSOR

        # Getting data
        data, prompts = self.get_year_data(batch_size*2)
        self.train_data = data[:batch_size]
        self.test_data = data[batch_size:]
        self.batch_size = batch_size

        self.set_loaders(self.train_data, self.test_data, shuffle=True)

        # Setting up criterion
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = device

    def get_year_data(self, num_examples):
        template = "The {noun} lasted from the year {year1} to "

        # set some random seed
        torch.random.manual_seed(54)
        nouns_perm = torch.randint(0, len(self.NOUNS), (num_examples,))
        years_perm = torch.randint(0, len(self.YEARS), (num_examples,))

        prompts = []
        prompts_tokenized = []
        for i in range(num_examples):
            year = self.YEARS[years_perm[i]]
            prompts.append(
                template.format(
                    noun=self.NOUNS[nouns_perm[i]],
                    year1=year,
                ) + year[:2]
            )
            prompts_tokenized.append(self.tokenizer.encode(prompts[-1], return_tensors="pt").to(self.device))
            assert prompts_tokenized[-1].shape == prompts_tokenized[0].shape, (prompts_tokenized[-1].shape, prompts_tokenized[0].shape)
        prompts_tokenized = torch.cat(prompts_tokenized, dim=0)
        assert len(prompts_tokenized.shape) == 2, prompts_tokenized.shape

        return prompts_tokenized, prompts

    def compute_means(self,
        model,
        num_data = None
    ):
        """
        Computes the mean of the activations across 
        the data for each component of the model. Used in mean edge ablation.
        """
        raise NotImplementedError

    def calculate_loss(self, model, batch): 
        # logits = model(batch)
        # # should pro
        # if isinstance(logits, tuple) or isinstance(logits, list):
        #     logits = logits[0]#.to('cpu')
        # elif isinstance(logits, ModelOutput):
        #     logits = logits.logits
        # else:
        #     assert isinstance(logits, torch.Tensor), logits
        #     logits = logits#.to('cpu')
        logits = get_final_logits(model, self.tokenizer, batch, input_text=False)

        # Gets the last 2 digits of the year
        yearend = self.INV_TOKENS_TENSOR[batch[:, 7]].to(logits.device)

        # Construct target distribution that is uniform for all years > yearend
        target = torch.zeros_like(logits) # Shape batch, vocab

        # For each year, set the probability of all years after it to 1 / p
        for i in range(len(yearend)):
            p = 100 - yearend[i] + 1
            target[i, self.TOKENS_TENSOR[yearend[i]+1:]] = 1 / p
        # Compute cross entropy loss
        return self.criterion(logits, target)
    
    def get_test_accuracy(self, model, use_test_data=True, check_all_logits=False):
        """
        check if probabilities for all years after are greter than probabilities for all years before, or 50% if check_all_logits is True
        """
        with torch.no_grad():
            batch = self.get_batch(train=not use_test_data)
            logits = get_final_logits(model, self.tokenizer, batch, input_text=False)
            probs = F.softmax(logits, dim=-1)

            yearend = self.INV_TOKENS_TENSOR[batch[:, 7]].to(logits.device)

            num_correct = 0
            num_total = 0
            # For each year, set the probability of all years after it to 1 / p
            for i in range(len(yearend)):
                # check probs of all years after are greater than probs of all years before
                correct_probs = probs[i, self.TOKENS_TENSOR[yearend[i]+1:]]
                
                if not check_all_logits:
                    incorrect_probs = probs[i, self.TOKENS_TENSOR[:yearend[i]+1]]
                    incorrect_prob = incorrect_probs.sum()
                else:
                    incorrect_prob = 0.5
                
                if correct_probs.sum() > incorrect_prob:
                    num_correct += 1
                num_total += 1
            return num_correct / num_total
