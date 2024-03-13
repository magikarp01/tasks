from tasks import HarmBenchTask
from tasks.harmbench.HarmBenchTask import 

class FastEvals:
    """
    A class to quickly evaluate a model using 
    """
    def __init__(self, model, model_tokenizer, harmbench_tasks=None, device="cuda", generation_kwargs={}):
        self.model = model
        self.model_tokenizer = model_tokenizer
        self.harbench_tasks = harmbench_tasks
        self.device = device
        self.generation_kwargs = generation_kwargs

    # def evaluate(self, eval_type, eval_data, eval_kwargs):

class FastGCGEvals:
    def __init__(self, eval_type, func_categories, gen_batch_size=24, cls_batch_size=12, *args, **kwargs):

        # get the model and tokenizer from kwargs

        if eval_type == "simple":
            self.harmbench_tasks = {
                "harmbench": HarmBenchTask(tokenizer=model_tokenizer, gen_batch_size=gen_batch_size, cls_batch_size=cls_batch_size, pretrained_cls="simple"),
                "harmbench_gcg": HarmBenchTask(tokenizer=model_tokenizer, gen_batch_size=gen_batch_size, cls_batch_size=cls_batch_size, pretrained_cls="gcg"),

            }