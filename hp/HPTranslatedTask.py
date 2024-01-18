from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import pandas as pd
import openai
from dotenv import load_dotenv
import os
from datetime import datetime
from tasks.task import Task
from tasks.hp.HPSAQ import HPSAQ
from tasks.hp.HPTask import HPTriviaTask, HPVerbatimTask
from tasks.hp.data.translated_v2_hptrivia.templates import (
    SPANISH_QA_TEMPLATE,
    SPANISH_Q_TEMPLATE,
    SPANISH_SAQ_SYSTEM_PROMPT,
    SPANISH_ZERO_SHOT_TEMPLATE,
    SPANISH_FEW_SHOT_TEMPLATE,
    SPANISH_UNRELATED_FEW_SHOT_TEMPLATE,
    SPANISH_TRIVIA_SYSTEM_PROMPT,
    CHINESE_QA_TEMPLATE,
    CHINESE_Q_TEMPLATE,
    CHINESE_SAQ_SYSTEM_PROMPT,
    CHINESE_ZERO_SHOT_TEMPLATE,
    CHINESE_FEW_SHOT_TEMPLATE,
    CHINESE_UNRELATED_FEW_SHOT_TEMPLATE,
    CHINESE_TRIVIA_SYSTEM_PROMPT,
)


class HPSAQSpanishTask(HPSAQ):

    def __init__(self, *args, **kwargs):

        script_dir = os.path.dirname(__file__)
        spanish_dataset_path = os.path.join(script_dir, "data/translated_v2_hptrivia/Spanish_datapoints_2024-01-16_00-17-23.jsonl")

        super().__init__(
            dataset_path=spanish_dataset_path,
            system_prompt=SPANISH_SAQ_SYSTEM_PROMPT,
            zero_shot_template=SPANISH_ZERO_SHOT_TEMPLATE,
            few_shot_template=SPANISH_FEW_SHOT_TEMPLATE,
            unrelated_few_shot_template=SPANISH_UNRELATED_FEW_SHOT_TEMPLATE,
            *args,
            **kwargs,
        )

class HPTriviaSpanishTask(HPTriviaTask):
    
    def __init__(self, *args, **kwargs):
        super().__init__(
            sys_msg=SPANISH_TRIVIA_SYSTEM_PROMPT,
            train_data_location='tasks/hp/data/translated_v2_hptrivia/Spanish_datapoints__train.jsonl',
            test_data_location='tasks/hp/data/translated_v2_hptrivia/Spanish_datapoints__test.jsonl',
            *args, 
            **kwargs,
        )
        # TODO: test data and train datasets
