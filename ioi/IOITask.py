from ast import Dict, Tuple
from typing import Union, List, Optional
from dataclasses import dataclass

import warnings
import torch as t
from torch import Tensor
from torch.utils.data import Dataset
import torch.nn.functional as F

import numpy as np
from transformers import AutoTokenizer
import random
import copy
import re
from rich import print as rprint
from rich.table import Table

from jaxtyping import Float, Bool
from tasks.task import Task

NAMES = [
    "Aaron",
    "Adam",
    "Alan",
    "Alex",
    "Alice",
    "Amy",
    "Anderson",
    "Andre",
    "Andrew",
    "Andy",
    "Anna",
    "Anthony",
    "Arthur",
    "Austin",
    "Blake",
    "Brandon",
    "Brian",
    "Carter",
    "Charles",
    "Charlie",
    "Christian",
    "Christopher",
    "Clark",
    "Cole",
    "Collins",
    "Connor",
    "Crew",
    "Crystal",
    "Daniel",
    "David",
    "Dean",
    "Edward",
    "Elizabeth",
    "Emily",
    "Eric",
    "Eva",
    "Ford",
    "Frank",
    "George",
    "Georgia",
    "Graham",
    "Grant",
    "Henry",
    "Ian",
    "Jack",
    "Jacob",
    "Jake",
    "James",
    "Jamie",
    "Jane",
    "Jason",
    "Jay",
    "Jennifer",
    "Jeremy",
    "Jessica",
    "John",
    "Jonathan",
    "Jordan",
    "Joseph",
    "Joshua",
    "Justin",
    "Kate",
    "Kelly",
    "Kevin",
    "Kyle",
    "Laura",
    "Leon",
    "Lewis",
    "Lisa",
    "Louis",
    "Luke",
    "Madison",
    "Marco",
    "Marcus",
    "Maria",
    "Mark",
    "Martin",
    "Mary",
    "Matthew",
    "Max",
    "Michael",
    "Michelle",
    "Morgan",
    "Patrick",
    "Paul",
    "Peter",
    "Prince",
    "Rachel",
    "Richard",
    "River",
    "Robert",
    "Roman",
    "Rose",
    "Ruby",
    "Russell",
    "Ryan",
    "Sarah",
    "Scott",
    "Sean",
    "Simon",
    "Stephen",
    "Steven",
    "Sullivan",
    "Taylor",
    "Thomas",
    "Tyler",
    "Victoria",
    "Warren",
    "William",
]

ABC_TEMPLATES = [
    "Then, [A], [B] and [C] went to the [PLACE]. [B] and [C] gave a [OBJECT] to [A]",
    "Afterwards [A], [B] and [C] went to the [PLACE]. [B] and [C] gave a [OBJECT] to [A]",
    "When [A], [B] and [C] arrived at the [PLACE], [B] and [C] gave a [OBJECT] to [A]",
    "Friends [A], [B] and [C] went to the [PLACE]. [B] and [C] gave a [OBJECT] to [A]",
]

BAC_TEMPLATES = [
    template.replace("[B]", "[A]", 1).replace("[A]", "[B]", 1)
    for template in ABC_TEMPLATES
]

BABA_TEMPLATES = [
    "Then, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to [A]",
    "Then, [B] and [A] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to [A]",
    "Then, [B] and [A] had a long argument, and afterwards [B] said to [A]",
    "After [B] and [A] went to the [PLACE], [B] gave a [OBJECT] to [A]",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give it to [A]",
    "When [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give the [OBJECT] to [A]",
    "While [B] and [A] were working at the [PLACE], [B] gave a [OBJECT] to [A]",
    "While [B] and [A] were commuting to the [PLACE], [B] gave a [OBJECT] to [A]",
    "After the lunch, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Afterwards, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a long argument. Afterwards [B] said to [A]",
    "The [PLACE] [B] and [A] went to had a [OBJECT]. [B] gave it to [A]",
    "Friends [B] and [A] found a [OBJECT] at the [PLACE]. [B] gave it to [A]",
]

BABA_LONG_TEMPLATES = [
    "Then in the morning, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then in the morning, [B] and [A] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then in the morning, [B] and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to [A]",
    "Then in the morning, [B] and [A] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to [A]",
    "Then in the morning, [B] and [A] had a long argument, and afterwards [B] said to [A]",
    "After taking a long break [B] and [A] went to the [PLACE], [B] gave a [OBJECT] to [A]",
    "When soon afterwards [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give it to [A]",
    "When soon afterwards [B] and [A] got a [OBJECT] at the [PLACE], [B] decided to give the [OBJECT] to [A]",
    "While spending time together [B] and [A] were working at the [PLACE], [B] gave a [OBJECT] to [A]",
    "While spending time together [B] and [A] were commuting to the [PLACE], [B] gave a [OBJECT] to [A]",
    "After the lunch in the afternoon, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Afterwards, while spending time together [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then in the morning afterwards, [B] and [A] had a long argument. Afterwards [B] said to [A]",
    "The local big [PLACE] [B] and [A] went to had a [OBJECT]. [B] gave it to [A]",
    "Friends separated at birth [B] and [A] found a [OBJECT] at the [PLACE]. [B] gave it to [A]",
]

BABA_LATE_IOS = [
    "Then, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a lot of fun at the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] were working at the [PLACE]. [B] decided to give a [OBJECT] to [A]",
    "Then, [B] and [A] were thinking about going to the [PLACE]. [B] wanted to give a [OBJECT] to [A]",
    "Then, [B] and [A] had a long argument and after that [B] said to [A]",
    "After the lunch, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Afterwards, [B] and [A] went to the [PLACE]. [B] gave a [OBJECT] to [A]",
    "Then, [B] and [A] had a long argument. Afterwards [B] said to [A]",
]

BABA_EARLY_IOS = [
    "Then [B] and [A] went to the [PLACE], and [B] gave a [OBJECT] to [A]",
    "Then [B] and [A] had a lot of fun at the [PLACE], and [B] gave a [OBJECT] to [A]",
    "Then [B] and [A] were working at the [PLACE], and [B] decided to give a [OBJECT] to [A]",
    "Then [B] and [A] were thinking about going to the [PLACE], and [B] wanted to give a [OBJECT] to [A]",
    "Then [B] and [A] had a long argument, and after that [B] said to [A]",
    "After the lunch [B] and [A] went to the [PLACE], and [B] gave a [OBJECT] to [A]",
    "Afterwards [B] and [A] went to the [PLACE], and [B] gave a [OBJECT] to [A]",
    "Then [B] and [A] had a long argument, and afterwards [B] said to [A]",
]

ABBA_TEMPLATES = BABA_TEMPLATES[:]
ABBA_LATE_IOS = BABA_LATE_IOS[:]
ABBA_EARLY_IOS = BABA_EARLY_IOS[:]

for TEMPLATES in [ABBA_TEMPLATES, ABBA_LATE_IOS, ABBA_EARLY_IOS]:
    for i in range(len(TEMPLATES)):
        first_clause = True
        for j in range(1, len(TEMPLATES[i]) - 1):
            if TEMPLATES[i][j - 1 : j + 2] == "[B]" and first_clause:
                TEMPLATES[i] = TEMPLATES[i][:j] + "A" + TEMPLATES[i][j + 1 :]
            elif TEMPLATES[i][j - 1 : j + 2] == "[A]" and first_clause:
                first_clause = False
                TEMPLATES[i] = TEMPLATES[i][:j] + "B" + TEMPLATES[i][j + 1 :]

VERBS = [" tried", " said", " decided", " wanted", " gave"]

PLACES = [
    "store",
    "garden",
    "restaurant",
    "school",
    "hospital",
    "office",
    "house",
    "station",
]

OBJECTS = [
    "ring",
    "kiss",
    "bone",
    "basketball",
    "computer",
    "necklace",
    "drink",
    "snack",
]


def gen_prompt_uniform(
    templates, names, nouns_dict, N, symmetric, prefixes=None, abc=False
):
    nb_gen = 0
    ioi_prompts = []
    while nb_gen < N:
        temp = random.choice(templates)
        temp_id = templates.index(temp)
        name_1 = ""
        name_2 = ""
        name_3 = ""
        while len(set([name_1, name_2, name_3])) < 3:
            name_1 = random.choice(names)
            name_2 = random.choice(names)
            name_3 = random.choice(names)

        nouns = {}
        ioi_prompt = {}
        for k in nouns_dict:
            nouns[k] = random.choice(nouns_dict[k])
            ioi_prompt[k] = nouns[k]
        prompt = temp
        for k in nouns_dict:
            prompt = prompt.replace(k, nouns[k])

        if prefixes is not None:
            L = random.randint(30, 40)
            pref = ".".join(random.choice(prefixes).split(".")[:L])
            pref += "<|endoftext|>"
        else:
            pref = ""

        prompt1 = prompt.replace("[A]", name_1)
        prompt1 = prompt1.replace("[B]", name_2)
        if abc:
            prompt1 = prompt1.replace("[C]", name_3)
        prompt1 = pref + prompt1
        ioi_prompt["text"] = prompt1
        ioi_prompt["IO"] = name_1
        ioi_prompt["S"] = name_2
        ioi_prompt["TEMPLATE_IDX"] = temp_id
        ioi_prompts.append(ioi_prompt)
        if abc:
            ioi_prompts[-1]["C"] = name_3

        nb_gen += 1

        if symmetric and nb_gen < N:
            prompt2 = prompt.replace("[A]", name_2)
            prompt2 = prompt2.replace("[B]", name_1)
            prompt2 = pref + prompt2
            ioi_prompts.append(
                {"text": prompt2, "IO": name_2, "S": name_1, "TEMPLATE_IDX": temp_id}
            )
            nb_gen += 1
    return ioi_prompts



def flip_words_in_prompt(prompt: str, word1: str, word2: str, instances: Optional[Union[int, List[int]]] = None):
    '''
    Flips instances of word `word1` with `word2` in the string `string`.

    By default it flips all instances, but the optional `instances` argument specifies which
    instances to flip (e.g. if instances = 0, then it only flips the 0th instance of either
    word1 or word2.

    Examples of (arguments) -> return value:

        ("ABA", "A", "B") -> "BAB"
        ("ABA", "A", "B", 1) -> "AAA"
        ("ABA", "A", "B", [0, 1]) -> "BAA
    '''
    split_prompt = re.split("({}|{})".format(word1, word2), prompt)
    indices_of_names = [i for i, s in enumerate(split_prompt) if s in (word1, word2)]
    indices_to_flip = [indices_of_names[i] for i in instances]
    for i in indices_to_flip:
        split_prompt[i] = word1 if split_prompt[i] == word2 else word1
    prompt = "".join(split_prompt)
    return prompt



def gen_flipped_prompts(prompts: List[dict], templates_by_prompt: List[str], flip: str, names: List[str], seed: int) -> List[dict]:
    '''
    Flip prompts in a way described by the flip argument. Returns new prompts.

    prompts: List[dict]
        list of prompts, each prompt is a dict with keys "S", "IO", "text", etc

    templates_by_prompt: List[str]
        each element is "ABBA" or "BABA"

    flip: str
        "ABB -> XYZ, BAB -> XYZ" means that the prompt "A and B went to [place], B gave [object] to A" becomes "X and Y went to [place], Z gave [object] to A" (and equivalent for the BABA case)

    names: List[str]
        list of names, for when flip involves random tokens

    seed: int
        provides reproducibility

    Note that we don't bother flipping the last token in the prompt (IO2), since
    we don't use it for anything (intuitively, we use this function to create
    datasets to provide us with corrupted signals, but we still use the IO2 from
    the original uncorrupted IOI database as our "correct answer", so we don't
    care about what the correct answer (IO2) for the corrupted set is).
    '''
    random.seed(seed)
    np.random.seed(seed)
    # print
    abba_flip, baba_flip = flip.split(",")
    flip_dict = {
        "ABB": [flip.strip() for flip in abba_flip.split("->")],
        "BAB": [flip.strip() for flip in baba_flip.split("->")]
    }

    new_prompts = []

    for idx, (prompt, template) in enumerate(zip(prompts, templates_by_prompt)):

        flip_orig, flip_new = flip_dict[template[:-1]]

        prompt = copy.copy(prompt)

        # Get indices and original values of first three names int the prompt
        prompt_split = prompt["text"].split(" ")
        orig_names_and_posns = [(i, s) for i, s in enumerate(prompt_split) if s in names][:3]
        orig_names = list(zip(*orig_names_and_posns))[1]

        # Get a dictionary of the correspondence between orig names and letters in flip_orig
        # (and get a subdict for those names which are kept in flip_new)
        orig_names_key = {
            letter: s
            for s, letter in zip(orig_names, flip_orig)
        }
        kept_names_key = {
            k: v
            for k, v in orig_names_key.items() if k in flip_new
        }
        # This line will throw an error if flip_orig is wrong (e.g. if it says "SOS" but the
        # S1 and S2 tokens don't actually match
        assert len(orig_names_key) == len(set(flip_orig))

        # Get all random names we'll need, in the form of a dictionary
        rand_names = {
            letter: np.random.choice(list(set(names) - set(orig_names)))
            for letter in set(flip_new) - set(flip_orig)
        }

        # Get a "full dictionary" which maps letters in flip_new to the new values they will have
        name_replacement_dict = {**kept_names_key, **rand_names}
        assert len(name_replacement_dict) == len(set(flip_new)), (name_replacement_dict, flip_new)

        # Populate the new names, with either random names or with the corresponding orig names
        for (i, s), letter in zip(orig_names_and_posns, flip_new):
            prompt_split[i] = name_replacement_dict[letter]

        # Join the prompt back together
        prompt["text"] = " ".join(prompt_split)

        # Change the identity of the S and IO tokens.
        # S token is just same as S2, but IO is a bit messier because it might not be
        # well-defined (it's defined as the unique non-duplicated name of the first
        # two). If it's ill-defined, WLOG set it to be the second name.
        prompt["S"] = name_replacement_dict[flip_new[-1]]
        possible_IOs = [name_replacement_dict[letter] for letter in flip_new[:2] if list(flip_new).count(letter) == 1]
        # Case where IO is well-defined
        if len(possible_IOs) == 1:
            prompt["IO"] = possible_IOs[0]
        # Case where it isn't well-defined
        else:
            prompt["IO"] = name_replacement_dict[flip_new[1]]

        new_prompts.append(prompt)

    return new_prompts



def get_name_idxs(prompts, tokenizer, idx_types=["IO", "S1", "S2"], prepend_bos=False):
    name_idx_dict = dict((idx_type, []) for idx_type in idx_types)
    for prompt in prompts:
        text_split = prompt["text"].split(" ")
        toks = tokenizer.tokenize(" ".join(text_split[:-1]))
        # Get the first instance of IO token
        name_idx_dict["IO"].append(
            toks.index(tokenizer.tokenize(" " + prompt["IO"])[0])
        )
        # Get the first instance of S token
        name_idx_dict["S1"].append(
            toks.index(tokenizer.tokenize(" " + prompt["S"])[0])
        )
        # Get the last instance of S token
        name_idx_dict["S2"].append(
            len(toks) - toks[::-1].index(tokenizer.tokenize(" " + prompt["S"])[0]) - 1
        )

    return [
        int(prepend_bos) + t.tensor(name_idx_dict[idx_type])
        for idx_type in idx_types
    ]


def get_word_idxs(prompts, word_list, tokenizer):
    """Get the index of the words in word_list in the prompts. Exactly one of the word_list word has to be present in each prompt"""
    idxs = []
    tokenized_words = [
        tokenizer.decode(tokenizer(word)["input_ids"][0]) for word in word_list
    ]
    for prompt in prompts:
        toks = [
            tokenizer.decode(t)
            for t in tokenizer(prompt["text"], return_tensors="pt", padding=True)[
                "input_ids"
            ][0]
        ]
        idx = None
        for i, w_tok in enumerate(tokenized_words):
            if word_list[i] in prompt["text"]:
                try:
                    idx = toks.index(w_tok)
                    if toks.count(w_tok) > 1:
                        idx = len(toks) - toks[::-1].index(w_tok) - 1
                except:
                    idx = toks.index(w_tok)
                    # raise ValueError(toks, w_tok, prompt["text"])
        if idx is None:
            raise ValueError(f"Word {word_list} and {i} not found {prompt}")
        idxs.append(idx)
    return t.tensor(idxs)


def get_end_idxs(toks, tokenizer, name_tok_len=1, prepend_bos=False):
    relevant_idx = int(prepend_bos)
    # if the sentence begins with an end token
    # AND the model pads at the end with the same end token,
    # then we need make special arrangements

    pad_token_id = tokenizer.pad_token_id

    end_idxs_raw = []
    for i in range(toks.shape[0]):
        if pad_token_id not in toks[i][1:]:
            end_idxs_raw.append(toks.shape[1])
            continue
        nonzers = (toks[i] == pad_token_id).nonzero()[relevant_idx][0].item()
        end_idxs_raw.append(nonzers)
    end_idxs = t.tensor(end_idxs_raw)
    end_idxs = end_idxs - 1 - name_tok_len

    for i in range(toks.shape[0]):
        assert toks[i][end_idxs[i] + 1] != 0 and (
            toks.shape[1] == end_idxs[i] + 2 or toks[i][end_idxs[i] + 2] == pad_token_id
        ), (
            toks[i],
            end_idxs[i],
            toks[i].shape,
            "the END idxs aren't properly formatted",
        )

    return end_idxs





def get_idx_dict(ioi_prompts, tokenizer, prepend_bos=False, toks=None):
    (IO_idxs, S1_idxs, S2_idxs,) = get_name_idxs(
        ioi_prompts,
        tokenizer,
        idx_types=["IO", "S1", "S2"],
        prepend_bos=prepend_bos,
    )

    end_idxs = get_end_idxs(
        toks,
        tokenizer,
        name_tok_len=1,
        prepend_bos=prepend_bos,
    )

    punct_idxs = get_word_idxs(ioi_prompts, [",", "."], tokenizer)

    return {
        "IO": IO_idxs,
        "IO-1": IO_idxs - 1,
        "IO+1": IO_idxs + 1,
        "S1": S1_idxs,
        "S1-1": S1_idxs - 1,
        "S1+1": S1_idxs + 1,
        "S2": S2_idxs,
        "end": end_idxs,
        "starts": t.zeros_like(end_idxs),
        "punct": punct_idxs,
    }

def format_prompt(sentence: str) -> str:
    '''Format a prompt by underlining names (for rich print)'''
    return re.sub("(" + "|".join(NAMES) + ")", lambda x: f"[u bold dark_orange]{x.group(0)}[/]", sentence) + "\n"


def make_table(cols, colnames, title="", n_rows=5, decimals=4):
    '''Makes and displays a table, from cols rather than rows (using rich print)'''
    table = Table(*colnames, title=title)
    rows = list(zip(*cols))
    f = lambda x: x if isinstance(x, str) else f"{x:.{decimals}f}"
    for row in rows[:n_rows]:
        table.add_row(*list(map(f, row)))
    rprint(table)

class IOIData:
    def __init__(
        self,
        prompt_type: Union[
            str, List[str]
        ],  # if list, then it will be a list of templates
        N=500,
        tokenizer=None,
        prompts=None,
        symmetric=False,
        prefixes=None,
        nb_templates=None,
        prepend_bos=False,
        manual_word_idx=None,
        has_been_flipped:bool=False,
        seed=0,
        device="cuda"
    ):
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        if not (
            N == 1
            or prepend_bos == False
            or tokenizer.bos_token_id == tokenizer.eos_token_id
        ):
            warnings.warn(
                "Probably word_idx will be calculated incorrectly due to this formatting"
            )
        self.has_been_flipped = has_been_flipped
        assert not (symmetric and prompt_type == "ABC")
        assert (
            (prompts is not None) or (not symmetric) or (N % 2 == 0)
        ), f"{symmetric} {N}"
        self.prompt_type = prompt_type

        if nb_templates is None:
            nb_templates = len(BABA_TEMPLATES)

        if prompt_type == "ABBA":
            self.templates = ABBA_TEMPLATES[:nb_templates].copy()
        elif prompt_type == "BABA":
            self.templates = BABA_TEMPLATES[:nb_templates].copy()
        elif prompt_type == "mixed":
            self.templates = (
                BABA_TEMPLATES[: nb_templates // 2].copy()
                + ABBA_TEMPLATES[: nb_templates // 2].copy()
            )
            random.shuffle(self.templates)
        elif prompt_type == "ABC":
            self.templates = ABC_TEMPLATES[:nb_templates].copy()
        elif prompt_type == "BAC":
            self.templates = BAC_TEMPLATES[:nb_templates].copy()
        elif prompt_type == "ABC mixed":
            self.templates = (
                ABC_TEMPLATES[: nb_templates // 2].copy()
                + BAC_TEMPLATES[: nb_templates // 2].copy()
            )
            random.shuffle(self.templates)
        elif isinstance(prompt_type, list):
            self.templates = prompt_type
        else:
            raise ValueError(prompt_type)

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer

        self.prefixes = prefixes
        self.prompt_type = prompt_type
        if prompts is None:
            self.ioi_prompts = gen_prompt_uniform(  # list of dict of the form {"text": "Alice and Bob bla bla. Bob gave bla to Alice", "IO": "Alice", "S": "Bob"}
                self.templates,
                NAMES,
                nouns_dict={"[PLACE]": PLACES, "[OBJECT]": OBJECTS},
                N=N,
                symmetric=symmetric,
                prefixes=self.prefixes,
                abc=(prompt_type in ["ABC", "ABC mixed", "BAC"]),
            )
        else:
            assert N == len(prompts), f"{N} and {len(prompts)}"
            self.ioi_prompts = prompts

        all_ids = [prompt["TEMPLATE_IDX"] for prompt in self.ioi_prompts]
        all_ids_ar = np.array(all_ids)
        self.groups = []
        for id in list(set(all_ids)):
            self.groups.append(np.where(all_ids_ar == id)[0])

        small_groups = []
        for group in self.groups:
            if len(group) < 5:
                small_groups.append(len(group))

        self.sentences = [
            prompt["text"] for prompt in self.ioi_prompts
        ]  # a list of strings. Renamed as this should NOT be forward passed

        self.templates_by_prompt = []  # for each prompt if it's ABBA or BABA
        for i in range(N):
            if self.sentences[i].index(self.ioi_prompts[i]["IO"]) < self.sentences[
                i
            ].index(self.ioi_prompts[i]["S"]):
                self.templates_by_prompt.append("ABBA")
            else:
                self.templates_by_prompt.append("BABA")

        texts = [
            (self.tokenizer.bos_token if prepend_bos else "") + prompt["text"]
            for prompt in self.ioi_prompts
        ]
        self.toks = t.Tensor(self.tokenizer(texts, padding=True).input_ids).long()

        self.word_idx = get_idx_dict(
            self.ioi_prompts,
            self.tokenizer,
            prepend_bos=prepend_bos,
            toks=self.toks,
        )
        self.prepend_bos = prepend_bos
        if manual_word_idx is not None:
            self.word_idx = manual_word_idx

        self.N = N
        self.max_len = max(
            [
                len(self.tokenizer(prompt["text"]).input_ids)
                for prompt in self.ioi_prompts
            ]
        )

        self.io_tokenIDs = [
            self.tokenizer.encode(" " + prompt["IO"])[0] for prompt in self.ioi_prompts
        ]
        self.s_tokenIDs = [
            self.tokenizer.encode(" " + prompt["S"])[0] for prompt in self.ioi_prompts
        ]

        self.tokenized_prompts = []

        for i in range(self.N):
            self.tokenized_prompts.append(
                "|".join([self.tokenizer.decode(tok) for tok in self.toks[i]])
            )

        self.device = device
        # self.to(device)

    def gen_flipped_prompts(self, flip):
        # Check if it's already been flipped (shouldn't string 2 flips together)
        if self.has_been_flipped:
            warnings.warn("This dataset has already been flipped. Generally, you should try and apply flips in one step, because this can lead to errors.")

        # Redefine seed (so it's different depending on what the flip is, e.g. we don't want (IO, RAND) then (S, RAND) to give us the same rand names)
        seed = self.seed + sum(map(ord, list("".join(flip))))

        # Get flipped prompts
        flipped_prompts = gen_flipped_prompts(self.ioi_prompts, self.templates_by_prompt, flip, NAMES, seed)

        flipped_ioi_dataset = IOIData(
            prompt_type=self.prompt_type,
            N=self.N,
            tokenizer=self.tokenizer,
            prompts=flipped_prompts,
            prefixes=self.prefixes,
            prepend_bos=self.prepend_bos,
            manual_word_idx=self.word_idx,
            has_been_flipped=True,
            seed=seed
        )
        return flipped_ioi_dataset

    def copy(self):
        copy_ioi_dataset = IOIData(
            prompt_type=self.prompt_type,
            N=self.N,
            tokenizer=self.tokenizer,
            prompts=self.ioi_prompts.copy(),
            prefixes=self.prefixes.copy() if self.prefixes is not None else self.prefixes,
        )
        return copy_ioi_dataset

    def __getitem__(self, key):
        sliced_prompts = self.ioi_prompts[key]
        sliced_dataset = IOIData(
            prompt_type=self.prompt_type,
            N=len(sliced_prompts),
            tokenizer=self.tokenizer,
            prompts=sliced_prompts,
            prefixes=self.prefixes,
            prepend_bos=self.prepend_bos,
        )
        return sliced_dataset

    def __setitem__(self, key, value):
        raise NotImplementedError()

    def __delitem__(self, key):
        raise NotImplementedError()

    def __len__(self):
        return self.N

    def tokenized_prompts(self):
        return self.toks

    def to(self, device):
        self.toks = self.toks.to(device)
        return self

class IOIDataset(Dataset, IOIData):
    def __init__(self, corrupt_format = ['ABC->XYZ', 'BAB->XYZ'], **kwargs):
        self.clean_data = IOIData(
            **kwargs
        )
        self.corr_data = self.clean_data.gen_flipped_prompts(corrupt_format)

        # self.clean_logit_diff = self.ave_logit_diff(
        #     model(self.clean_data.toks),
        #     self.clean_data
        # ).item()
        # self.corr_logit_diff = self.ave_logit_diff(
        #     model(self.corr_data.toks),
        #     self.corr_data
        # ).item()

    def __len__(self):
        return len(self.clean_data)

    def __getitem__(self, idx):
        clean_tok = self.clean_data.toks[idx]
        clean_io_id = self.clean_data.io_tokenIDs[idx]
        clean_s_id = self.clean_data.s_tokenIDs[idx]
        clean_word_end = self.clean_data.word_idx['end'][idx]

        corr_tok = self.corr_data.toks[idx]
        corr_io_id = self.corr_data.io_tokenIDs[idx]
        corr_s_id = self.corr_data.s_tokenIDs[idx]
        corr_word_end = self.corr_data.word_idx['end'][idx]
        # Maybe you only need the clean and corrupt token?
        return {
            'clean_tok': clean_tok,
            'clean_io_id': clean_io_id,
            'clean_s_id': clean_s_id,
            'corr_tok': corr_tok,
            'corr_io_id': corr_io_id,
            'corr_s_id': corr_s_id,
            'clean_word_end': clean_word_end,
            'corr_word_end': corr_word_end
        }

    def get_loss(
        clean_logits: Float[Tensor, 'batch seq d_vocab'],
        corr_logits: Float[Tensor, 'batch seq d_vocab'],
        batch: Dict,
        per_prompt: bool = False
    ):
        '''
            Return average logit difference between correct and incorrect answers
        '''
        # Get logits for indirect objects
        clean_io_logits = clean_logits[range(clean_logits.size(0)), batch['clean_word_end'], batch['clean_io_id']]
        clean_s_logits = clean_logits[range(clean_logits.size(0)), batch['clean_word_end'], batch['clean_s_id']]

        corr_io_logits = corr_logits[range(corr_logits.size(0)), batch['corr_word_end'], batch['corr_io_id']]
        corr_s_logits = corr_logits[range(corr_logits.size(0)), batch['corr_word_end'], batch['corr_s_id']]
        # Get logits for subject
        clean_logit_diff = clean_s_logits - clean_io_logits
        corr_logit_diff =  corr_s_logits - corr_io_logits

        if per_prompt:
            return (clean_logit_diff, corr_logit_diff)
        else:
            return (clean_logit_diff.mean(), corr_logit_diff.mean())



import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle
from tasks.inference_utils import get_final_logits

class IOITask_old(Task):

    class IOIPromptsDataset(Dataset):
        def __init__(self, ioi_prompts, tokenizer):
            self.ioi_prompts = ioi_prompts
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.ioi_prompts)

        def __getitem__(self, idx):
            prompt = self.ioi_prompts[idx]
            text = prompt['text']
            text = " ".join(text.split(" ")[:-1])
            label = prompt['IO']
            return {
                'text': text,
                'IO': label,
                'S': prompt['S']
            }

    def collate_batch(self, batch):

        texts = [item['text'] for item in batch]
        ios = [item['IO'] for item in batch]
        subjects = [item['S'] for item in batch]
        return {
            'text': texts,
            'IO': ios,
            'S': subjects
        }

    def __init__(self, batch_size, tokenizer, template_type="single",
                 handle_multitoken_labels=False, device='cuda'):
        """
        handle_multitoken_labels is for tokenizers other than gpt2, which might tokenize names with multiple tokens. In this case, we'll just analyze the first token.
        """
        with open(f"tasks/ioi/data/ioi_prompts_{template_type}_template_train.pkl", "rb") as f:
            ioi_prompts_train = pickle.load(f)
        with open(f"tasks/ioi/data/ioi_prompts_{template_type}_template_test.pkl", "rb") as f:
            ioi_test_prompts = pickle.load(f)
        self.ioi_prompts_train_dataset = self.IOIPromptsDataset(ioi_prompts_train, tokenizer)
        self.ioi_prompts_test_dataset = self.IOIPromptsDataset(ioi_test_prompts, tokenizer)

        self.train_loader = DataLoader(self.ioi_prompts_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_batch)
        self.test_loader = DataLoader(self.ioi_prompts_test_dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_batch)

        self.train_iter = iter(self.train_loader)
        self.test_iter = iter(self.test_loader)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.tokenizer = tokenizer
        self.handle_multitoken_labels = handle_multitoken_labels
        self.device = device
    
    def tokenize_names(self, names):
        """
        Tokenize names but handle possible multitoken labels
        """
        if self.handle_multitoken_labels:
            try:
                tokenized_names = self.tokenizer(names, return_tensors='pt').input_ids
            except ValueError:
                # tokenizes as multiple tokens
                tokenized_names = self.tokenizer(names).input_ids
                # convert to tensor, but only take first token of each
                tokenized_names = torch.tensor([[t[0]] for t in tokenized_names])
        else:
            tokenized_names = self.tokenizer(names, return_tensors='pt').input_ids
        return tokenized_names

    def calculate_loss(self, model, batch):
        """
        Batch is not tokens
        """

        last_logits = get_final_logits(model, self.tokenizer, batch['text'])
        labels = [' ' + io for io in batch['IO']]
        tokenized_labels = self.tokenize_names(labels)

        assert tokenized_labels.shape[0] == last_logits.shape[0]
        # labels should be 1 token
        assert tokenized_labels.shape[1] == 1, tokenized_labels

        return self.criterion(last_logits, tokenized_labels[:, 0].to(self.device))

    def get_train_loss(self, model):
        batch = self.get_batch(train=True)        
        return self.calculate_loss(model, batch)
    
    def get_test_loss(self, model):
        batch = self.get_batch(train=False)
        with torch.no_grad():
            return self.calculate_loss(model, batch)
    
    def get_test_accuracy(self, model, use_test_data=True, check_all_logits=False):
        """
        Accuracy of model assigning the largest logit to the indirect object. If check_all_logits is True, then checks argmax over all possible tokens. If false, checks argmax over subject token vs indirect object token.
        """
        with torch.no_grad():
            batch = self.get_batch(train=not use_test_data)
            prompts = batch['text']
            ios = batch['IO']
            subjects = batch['S']

            # io_tokens = self.tokenizer([' ' + io for io in ios], return_tensors='pt').input_ids
            io_tokens = self.tokenize_names([' ' + io for io in ios])

            assert io_tokens.shape[1] == 1
            io_tokens = io_tokens[:, 0].to(self.device)
            last_logits = get_final_logits(model, self.tokenizer, prompts)

            if check_all_logits:
                num_correct = (torch.argmax(last_logits, dim=1) == io_tokens).sum().item()

            else:
                # subject_tokens = self.tokenizer([' ' + s for s in subjects], return_tensors='pt').input_ids
                subject_tokens = self.tokenize_names([' ' + s for s in subjects])
                assert subject_tokens.shape[1] == 1
                subject_tokens = subject_tokens[:, 0].to(self.device)
                num_correct = 0
                for idx in range(len(io_tokens)):
                    subject_logit = last_logits[idx, subject_tokens[idx]]
                    io_logit = last_logits[idx, io_tokens[idx]]
                    num_correct += 1 if (io_logit.item() > subject_logit.item()) else 0
            return num_correct / len(io_tokens)
                    
    def get_tokens(self, num_batches, use_train_data=True):
        # get tokens for num_batches, return in a tensor of shape (num_batches * self.batch_size, prompt_len)
        tokens = []
        for i in range(num_batches):
            if use_train_data:
                try:
                    batch = next(self.train_iter)
                except StopIteration:
                    self.train_iter = iter(self.train_loader)
                    batch = next(self.train_iter)
            else:
                try:
                    batch = next(self.test_iter)
                except StopIteration:
                    self.test_iter = iter(self.test_loader)
                    batch = next(self.test_iter)

            prompts = batch['text']
            tokens.append(self.tokenizer(prompts, return_tensors='pt', padding=True).input_ids)
        return torch.cat(tokens, dim=0)


    # def calculate_loss(self, tokens, last_token_positions, model):
    #     outputs = model(tokens)
    #     logits = outputs[0]

    #     # get logits at the last token position
        
    #     logits_last_token = []
    #     labels = []

    #     for i in range(last_token_positions.shape[0]):
    #         logits_last_token.append(logits[i, last_token_positions[i]-1])
    #         labels.append(tokens[i, last_token_positions[i]])
    #     logits_last_token = torch.stack(logits_last_token)
    #     labels = torch.stack(labels).to(logits_last_token.device)
    #     print(f"{logits_last_token.shape=}, {labels.shape=}")
    #     print(f"{logits_last_token.argmax(dim=1)=}, {labels=}")
    
    #     loss = self.criterion(logits_last_token, labels)
        
    #     return loss

    # def get_train_loss(self, model):
    #     try:
    #         batch = next(self.train_iter)
    #     except StopIteration:
    #         self.train_iter = iter(self.train_loader)
    #         batch = next(self.train_iter)


    #     inputs = batch['tokens']
    #     last_token_positions = batch['last_token_pos']

    #     loss = self.calculate_loss(inputs, last_token_positions, model)
    #     return loss

    # def get_test_loss(self, model):
    #     # same as train
    #     with torch.no_grad():
    #         try:
    #             batch = next(self.test_iter)
    #         except StopIteration:
    #             self.test_iter = iter(self.test_loader)
    #             batch = next(self.test_iter)

    #         inputs = batch['tokens']
    #         last_token_positions = batch['last_token_pos']

    #         loss = self.calculate_loss(inputs, last_token_positions, model)
    #         return loss
    
    # def get_test_accuracy(self, model):
    #     with torch.no_grad():
    #         try:
    #             batch = next(self.test_iter)
    #         except StopIteration:
    #             self.test_iter = iter(self.test_loader)
    #             batch = next(self.test_iter)

    #         inputs = batch['tokens']
    #         last_token_positions = batch['last_token_pos']

    #         outputs = model(inputs)
    #         logits = outputs[0]

    #         # get logits at the last token position
    #         logits_last_token = []
    #         labels = []

    #         for i in range(last_token_positions.shape[0]):
    #             logits_last_token.append(logits[i, last_token_positions[i]-1])
    #             labels.append(inputs[i, last_token_positions[i]])
    #         logits_last_token = torch.stack(logits_last_token)
    #         labels = torch.stack(labels).to(logits_last_token.device)

    #         predictions = logits_last_token.argmax(dim=1)
    #         accuracy = (predictions == labels).sum().item() / predictions.shape[0]
    #         return accuracy


    # def get_loss(self, logits):
    #     patched_logit_diff = self.ave_logit_diff(logits, self.clean_data)
    #     return (patched_logit_diff - self.corrupted_logit_diff) / (self.clean_logit_diff - self.corrupted_logit_diff)


from tasks.task import Task
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
class IOITask(IOITask_old):
    class IOIPromptsDataset(Dataset):
        def __init__(self, data_list):
            self.data = data_list

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            return item
        

    def remove_IO(self, text, io, abc=False):
        if not abc:
            # remove second occurrence of io in text
            first_index = text.find(io)
            assert first_index != -1, f"IO word {io} not found in text {text}"
            # Find the second occurrence of the io word
            second_index = text.find(io, first_index + len(io))
            assert second_index != -1, f"IO word {io} only occurs once in text {text}"

            new_text = text[:second_index] + text[second_index:].replace(io, '', 1)
            
            # remove last space
            return new_text[:-1]

        else:
            """
            Just remove last word, since IO is not repeated
            """
            # just remove the last word and last space
            return text[:text.rfind(' ')]

    def process_prompt(self, prompt, abc=False):
        prompt['text'] = self.remove_IO(prompt['text'], prompt['IO'], abc=abc)
        return prompt

    def __init__(self, batch_size, tokenizer, handle_multitoken_labels=False, prompt_type='ABBA', num_data=1000, nb_templates=1, prep_acdcpp=False, acdcpp_N=25, device='cuda'):
        
        self.ioi_data = IOIData(
            prompt_type=prompt_type,
            N=num_data,
            tokenizer=tokenizer,
            prepend_bos=False,
            seed=1,
            nb_templates=nb_templates,
            device=device
        )
        
        # Split the data into train and test sets
        train_data, test_data = train_test_split(self.ioi_data.ioi_prompts, test_size=0.2, shuffle=False)
        train_data = [self.process_prompt(prompt) for prompt in train_data]
        test_data = [self.process_prompt(prompt) for prompt in test_data]

        # self.train_toks, self.test_toks = train_test_split(self.clean_data.toks, test_size=0.2, shuffle=False)

        self.train_loader = DataLoader(self.IOIPromptsDataset(train_data), batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.IOIPromptsDataset(test_data), batch_size=batch_size, shuffle=True)

        self.train_iter = iter(self.train_loader)
        self.test_iter = iter(self.test_loader)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.tokenizer = tokenizer
        self.handle_multitoken_labels = handle_multitoken_labels
        self.device = device


        if prep_acdcpp:
            self.clean_data = IOIData(
                prompt_type=prompt_type,
                N=acdcpp_N,
                tokenizer=tokenizer,
                prepend_bos=False,
                seed=1,
                nb_templates=nb_templates
            )
            self.corr_data = self.clean_data.gen_flipped_prompts('ABC->XYZ, BAB->XYZ')
            # no point making a train-test split, since these prompts are kind of meaningless
            corr_prompts = [self.process_prompt(prompt, abc=True) for prompt in self.corr_data.ioi_prompts]
            self.corr_loader = DataLoader(self.IOIPromptsDataset(corr_prompts), batch_size=batch_size, shuffle=True)
            self.corr_iter = iter(self.corr_loader)
            self.clean_logit_diff = None
            self.corrupt_logit_diff = None



    def ave_logit_diff(self,
        logits: Float[Tensor, 'batch seq d_vocab'],
        ioi_dataset: IOIData,
        per_prompt: bool = False,
    ):
        '''
        Return average logit difference between correct and incorrect answers
        '''
        # Get logits for indirect objects
        # print(f"{logits.shape=}, {self.clean_data.word_idx['end'].shape=}, {len(self.clean_data.io_tokenIDs)=}, {len(self.clean_data.s_tokenIDs)=}")
        io_logits = logits[range(logits.size(0)), ioi_dataset.word_idx['end'], ioi_dataset.io_tokenIDs]
        s_logits = logits[range(logits.size(0)), ioi_dataset.word_idx['end'], ioi_dataset.s_tokenIDs]
        # Get logits for subject
        logit_diff = io_logits - s_logits
        return logit_diff if per_prompt else logit_diff.mean()

    def set_logit_diffs(self, model):
        """
        Set clean_logit_diff and corrupt_logit_diff if they have not been set yet
        """
        clean_logits = model(self.clean_data.toks)
        corrupt_logits = model(self.corr_data.toks)
        self.clean_logit_diff = self.ave_logit_diff(clean_logits, self.clean_data).item()
        self.corrupted_logit_diff = self.ave_logit_diff(corrupt_logits, self.corr_data).item()

    def get_acdcpp_metric(self, model=None):
        '''
        Higher order function that returns the ioi_metric, using the baseline logit diffs previously calculated (or can be calculated within the function if model is passed in). Return function signature:
            - logits is output of some kind of altered model
            - 

        '''

        if self.clean_logit_diff is None or self.corrupted_logit_diff is None:
            assert model is not None, "Need to pass in model to get logit diffs, or call set_logit_diffs(model) elsewhere"
            self.set_logit_diffs(model)

        def ioi_metric(
                logits: Float[Tensor, "batch seq_len d_vocab"], 
                corrupted_logit_diff: float=self.corrupted_logit_diff,
                clean_logit_diff: float=self.clean_logit_diff,
                ioi_dataset: IOIData=self.clean_data):
            patched_logit_diff = self.ave_logit_diff(logits, ioi_dataset)
            return (patched_logit_diff - corrupted_logit_diff) / (clean_logit_diff - corrupted_logit_diff)
        return ioi_metric

    # def abs_ioi_metric(logits: Float[Tensor, "batch seq_len d_vocab"]):
    #     return abs(ioi_metric(logits))

    # def negative_ioi_metric(logits: Float[Tensor, "batch seq_len d_vocab"]):
    #     return -ioi_metric(logits)

    # def negative_abs_ioi_metric(logits: Float[Tensor, "batch seq_len d_vocab"]):
    #     return -abs_ioi_metric(logits)

class IOITask_Uniform(IOITask):
    """
    A Class for IOI tasks where the loss is calculated based on a uniform distribution. Distribution can be over IO and S, or all names we test (not recommended), or all possible tokens.
    """
    def __init__(self, batch_size, tokenizer, handle_multitoken_labels=False, prompt_type='ABBA', num_data=1000, nb_templates=1, prep_acdcpp=False, acdcpp_N=25, device='cuda', uniform_over='IO_S'):
        """
        uniform_over can be "IO_S", "names", or "all_tokens". "IO_S" means uniform over IO and S, "names" means uniform over all names that we've written, and "all_tokens" means uniform over all tokens.
        """
        super().__init__(batch_size, tokenizer, handle_multitoken_labels=handle_multitoken_labels, prompt_type=prompt_type, num_data=num_data, nb_templates=nb_templates, prep_acdcpp=prep_acdcpp, acdcpp_N=acdcpp_N, device=device)
        self.criterion = F.cross_entropy
        self.uniform_over = uniform_over
    
    def calculate_loss(self, model, batch):
        """
        Calculate loss differently, with target distribution as a uniform distribution (type depends on uniform_over).
        """
        last_logits = get_final_logits(model, self.tokenizer, batch['text'])
        target_distribution = torch.zeros_like(last_logits)

        if self.uniform_over == 'IO_S':
            # uniform over IO and S
            # get logits for IO and S
            io_labels = [' ' + io for io in batch['IO']]
            s_labels = [' ' + s for s in batch['S']]
            io_tokens = self.tokenize_names(io_labels)
            s_tokens = self.tokenize_names(s_labels)

            # get cross entropy loss with target distribution of uniform over IO and S
            # Assign a probability of 0.5 to both io_tokens and s_tokens
            for i in range(target_distribution.shape[0]):
                target_distribution[i, io_tokens[i]] = 0.5
                target_distribution[i, s_tokens[i]] = 0.5 
                # print(f"{io_labels[i]=}, {s_labels[i]=}, {io_tokens[i]=}, {s_tokens[i]=}, {last_logits[i][io_tokens[i]]=}, {last_logits[i][s_tokens[i]]=}, {last_logits[i].mean()=}")

        elif self.uniform_over == 'names':
            # uniform over all names
            # get logits for all names
            all_names = NAMES
            all_tokens = self.tokenize_names(all_names)
            # get cross entropy loss with target distribution of uniform over all names
            for i in range(target_distribution.shape[0]):
                target_distribution[i, all_tokens] = 1 / len(all_tokens[i])
        elif self.uniform_over == 'all_tokens':
            # uniform over all tokens
            # get cross entropy loss with target distribution of uniform over all tokens
            for i in range(target_distribution.shape[0]):
                target_distribution[i] = 1 / target_distribution.shape[1]
        else:
            raise ValueError(f"uniform_over {self.uniform_over} not supported")
        return self.criterion(last_logits, target_distribution)
        # tokenized_labels = self.tokenize_names(labels)
        # assert tokenized_labels.shape[0] == last_logits.shape[0]
        # # labels should be 1 token
        # assert tokenized_labels.shape[1] == 1, tokenized_labels

            # tokenized_labels = tokenized_labels[:, 0].to(self.device)
            # return self.criterion(last_logits, tokenized_labels)
