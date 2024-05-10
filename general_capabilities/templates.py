DEFAULT_4_QUESTION_FORMAT = """
{question}

Choices:
(A) {choice_A}
(B) {choice_B}
(C) {choice_C}
(D) {choice_D}

Answer:
("""

NO_ANSWER_4_QUESTION_FORMAT = """
{question}

Choices:
(A) {choice_A}
(B) {choice_B}
(C) {choice_C}
(D) {choice_D}
"""

DEFAULT_2_QUESTION_FORMAT = """
{question}

Choices:
(A) {choice_A}
(B) {choice_B}

Answer:
("""

DEFAULT_HELLASWAG_QUESTION_FORMAT = """
Given the context, which choice is the most likely completion to the sentence?

Context:
{question}

Choices:
(A) {choice_A}
(B) {choice_B}
(C) {choice_C}
(D) {choice_D}

Answer:
("""

DEFAULT_LAMBADA_QUESTION_FORMAT = """
Complete the sentence:

{question} """

DEFAULT_WINOGRANDE_QUESTION_FORMAT = """
For the given sentence, which choice best fills in the blank '_' position?

{question}

Choices:
(A) {choice_A}
(B) {choice_B}

Answer:
("""


DEFAULT_WMDP_QUESTION_FORMAT = """
The following are multiple choice questions (with answers) about {topic}.
{question}

Choices:
(A) {a1}
(B) {a2}
(C) {a3}
(D) {a4}

Answer:
("""