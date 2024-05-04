from tasks.task import Task
import pandas as pd
import torch
from torch import Tensor
from tasks.inference_utils import get_final_logits, log_1_minus_p_loss, npo_loss
from datasets import load_datasets

from jaxtyping import Float
import json
import einops

class CounterFactTask(Task):
    def __init__(self, *args, short_prompt=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.short_prompt = short_prompt
        