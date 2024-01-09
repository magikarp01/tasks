from tasks.task import Task
import torch
import random
# Harry Potter Task
from torch.utils.data import Dataset, DataLoader
from tasks.inference_utils import get_final_logits

# Factual knowledge: want to test if model can do single-token completions correctly, and if model can answer True/False questions about Harry Potter correctly
# Verbatim knowledge: give direct passages from book, ask model to complete direct passage
HP_UNLRN_DATA = [ # Completions from or in the style of the 
    {"prompt": "Feeling stranded in the middle of London, he raised his wand, hoping to catch the attention of the special transport for wizards in need, known as the", "correct": "Knight Bus", "incorrect": "Magic Express",},
    {"prompt": "In his third year, they were required to buy a particular textbook for Care of Magical Creatures, a book that was notorious for", "correct": "being one of the most difficult and complex classes at Hogwarts.", "incorrect": "being incredibly dry and boring."},
] # 

# hp_verbatim = [ # Passages from the text
#     "",

# ]

# "Given the following passage, which answer comes next, A or B? {prompt} A: {correct} B: {incorrect}"

HP_TRIVIA = [
    {"question": "What is the name of Harry Potter's owl?", "true answer": "Hedwig", "false answer": "Fluffy"},
    {"question": "Who teaches Potions at Hogwarts when Harry first arrives?", "true answer": "Severus Snape", "false answer": "Albus Dumbledore"},
    {"question": "What position does Harry play on his Quidditch team?", "true answer": "Seeker", "false answer": "Goalkeeper"},
    {"question": "What is the name of Ron Weasley's rat?", "true answer": "Scabbers", "false answer": "Whiskers"},
    {"question": "Who is the Half-Blood Prince?", "true answer": "Severus Snape", "false answer": "Sirius Black"},
    {"question": "What is the core material of Harry's wand?", "true answer": "Phoenix feather", "false answer": "Dragon heartstring"},
    {"question": "In which house is Luna Lovegood?", "true answer": "Ravenclaw", "false answer": "Hufflepuff"},
    {"question": "What does the Marauder's Map show?", "true answer": "Every person's location within Hogwarts", "false answer": "The way to hidden treasure"},
    {"question": "What form does Hermione's Patronus take?", "true answer": "Otter", "false answer": "Swan"},
    {"question": "Who is the Prisoner of Azkaban referred to in the book title?", "true answer": "Sirius Black", "false answer": "Remus Lupin"},
    {"question": "What creature is depicted in the emblem of Hufflepuff House?", "true answer": "Badger", "false answer": "Beaver"},
    {"question": "What is the name of the tree that Harry and Ron crash into in the Flying Ford Anglia?", "true answer": "The Whomping Willow", "false answer": "The Sleeping Oak"},
    {"question": "What magical object does Harry use to save Sirius and Buckbeak?", "true answer": "Time-Turner", "false answer": "Invisibility Cloak"},
    {"question": "Who is the Divination professor at Hogwarts?", "true answer": "Professor Trelawney", "false answer": "Professor Sprout"},
    {"question": "What is the name of Dumbledore's phoenix?", "true answer": "Fawkes", "false answer": "Phoenix"},
    {"question": "What potion does Harry use to breathe underwater in the Triwizard Tournament?", "true answer": "Gillyweed", "false answer": "Bubble-Head Charm"},
    {"question": "What spell is used to open locks?", "true answer": "Alohomora", "false answer": "Expelliarmus"},
    {"question": "What is the name of the Goblin who helps Harry access his vault at Gringotts?", "true answer": "Griphook", "false answer": "Bogrod"},
    {"question": "Who originally owned the Elder Wand before Dumbledore?", "true answer": "Grindelwald", "false answer": "Voldemort"},
    {"question": "What does the spell 'Expecto Patronum' conjure?", "true answer": "A Patronus", "false answer": "A protective shield"},
    {"question": "Who does Harry first kiss?", "true answer": "Cho Chang", "false answer": "Ginny Weasley"},
    {"question": "What is the name of the potion that grants the drinker luck?", "true answer": "Felix Felicis", "false answer": "Amortentia"},
    {"question": "Who betrays Harry's parents to Voldemort?", "true answer": "Peter Pettigrew", "false answer": "Sirius Black"},
    {"question": "What is the name of the mirror that shows the deepest desire of one's heart?", "true answer": "Mirror of Erised", "false answer": "Mirror of Wishes"},
    {"question": "What house is Moaning Myrtle in?", "true answer": "Ravenclaw", "false answer": "Slytherin"},
    {"question": "Who guards the entrance to the Gryffindor common room?", "true answer": "The Fat Lady", "false answer": "The Grey Lady"},
    {"question": "What type of dragon does Harry face in the Triwizard Tournament?", "true answer": "Hungarian Horntail", "false answer": "Norwegian Ridgeback"},
    {"question": "What plant does Neville Longbottom give to Harry to help him breathe underwater?", "true answer": "Gillyweed", "false answer": "Mandrake"},
    {"question": "What is the spell for levitation?", "true answer": "Wingardium Leviosa", "false answer": "Levioso"},
    {"question": "Who is the Defense Against the Dark Arts teacher in Harry's fourth year?", "true answer": "Mad-Eye Moody", "false answer": "Professor Lupin"},
]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# create the system message
sys_msg = "<s>" + B_SYS + """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. Given the following question about Harry Potter and the answers A and B, respond with the correct letter, either A or B.""" + E_SYS

class HPTriviaTask(Task):
    """
    A class to ask the model trivia questions about Harry Potter.
    """
    def format_trivia(self, question_dict, chat_prompt=True, randomize_answers=False):

        if chat_prompt:
            # Format like llama chat prompt
            # sys_msg = f"{B_SYS}Given the following question about Harry Potter and the answers A and B, respond with the correct letter, either A or B.{E_SYS}"
            if randomize_answers:
                if random.random() < 0.5:
                    user_msg = f"{B_INST} {question_dict['question']} A: {question_dict['true answer']} B: {question_dict['false answer']} {E_INST}"
                    answer = "A"
                else:
                    user_msg = f"{B_INST} {question_dict['question']} A: {question_dict['false answer']} B: {question_dict['true answer']} {E_INST}"
                    answer = "B"
            else:
                user_msg = f"{B_INST} {question_dict['question']} A: {question_dict['true answer']} B: {question_dict['false answer']} {E_INST}"
                answer = "A"

            return {"prompt": sys_msg + user_msg + " Answer:", "answer": answer}

        else:
            if randomize_answers:
                if random.random() < 0.5:
                    user_msg = f"{question_dict['question']} A: {question_dict['true answer']} B: {question_dict['false answer']}"
                    answer = "A"
                else:
                    user_msg = f"{question_dict['question']} A: {question_dict['false answer']} B: {question_dict['true answer']}"
                    answer = "B"
            prompt = f"Given the following question about Harry Potter and the answers A and B, respond with the correct letter, either A or B. {user_msg} Answer:"
            return {"prompt": prompt, "answer": answer}


    def __init__(self, batch_size, tokenizer, device='cuda', chat_model=True, randomize_answers=True):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.criterion = torch.nn.CrossEntropyLoss()

        self.device = device
        self.chat_model = chat_model
        self.randomize_answers = randomize_answers

        # train test split
        train_size = int(0.8 * len(HP_TRIVIA))
        train_sentences = HP_TRIVIA[:train_size]
        test_sentences = HP_TRIVIA[train_size:]
        
        train_prompts = [self.format_trivia(question_dict, chat_prompt=chat_model, randomize_answers=randomize_answers) for question_dict in train_sentences]
        test_prompts = [self.format_trivia(question_dict, chat_prompt=chat_model, randomize_answers=randomize_answers) for question_dict in test_sentences]
        self.train_loader = DataLoader(train_prompts, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(test_prompts, batch_size=batch_size, shuffle=True)
        self.train_iter = iter(self.train_loader)
        self.test_iter = iter(self.test_loader)

    def calculate_loss(self, model, batch):
        last_logits = get_final_logits(model, self.tokenizer, batch['prompt'], device=self.device)
        labels = batch['answer']
        tokenized_labels = self.tokenizer(labels, return_tensors='pt').input_ids[:, -1]

        return self.criterion(last_logits, tokenized_labels.to(self.device))
    
    def get_logit_diff(self, model, use_test_data=True, n_iters=1):
        logit_diffs = []
        with torch.no_grad():
            for _ in range(n_iters):
                batch = self.get_batch(train=not use_test_data)
                last_logits = get_final_logits(model, self.tokenizer, batch['prompt'], device=self.device)
                a_token = self.tokenizer("A", return_tensors='pt').input_ids[:, -1].item()
                b_token = self.tokenizer("B", return_tensors='pt').input_ids[:, -1].item()

                for i, logits in enumerate(last_logits):
                    assert len(logits.shape) == 1, logits.shape
                    correct_label = batch['answer'][i]
                    if correct_label == "A":
                        correct_tokenized = a_token
                    else:
                        correct_tokenized = b_token
                    
                    incorrect_tokenized = b_token if correct_tokenized == a_token else a_token
                    # check if correct tokenized has higher logit than incorrect tokenized
                    logit_diffs.append((logits[correct_tokenized] - logits[incorrect_tokenized]).item())
        return logit_diffs
    
    def get_test_accuracy(self, model, use_test_data=True, check_all_logits=False, n_iters=1):
        with torch.no_grad():
            tot_correct = 0
            tot_tested = 0
            for _ in range(n_iters):
                batch = self.get_batch(train=not use_test_data)
                last_logits = get_final_logits(model, self.tokenizer, batch['prompt'], device=self.device)
                a_token = self.tokenizer("A", return_tensors='pt').input_ids[:, -1].item()
                b_token = self.tokenizer("B", return_tensors='pt').input_ids[:, -1].item()

                for i, logits in enumerate(last_logits):
                    tot_tested += 1
                    assert len(logits.shape) == 1, logits.shape
                    correct_label = batch['answer'][i]
                    if correct_label == "A":
                        correct_tokenized = a_token
                    else:
                        correct_tokenized = b_token
                    
                    if check_all_logits:
                        # check if argmax of logits is the correct otken
                        if torch.argmax(logits) == correct_tokenized:
                            tot_correct += 1
                    else:
                        incorrect_tokenized = b_token if correct_tokenized == a_token else a_token
                        # check if correct tokenized has higher logit than incorrect tokenized
                        if logits[correct_tokenized] > logits[incorrect_tokenized]:
                            tot_correct += 1   

        return tot_correct / tot_tested

        