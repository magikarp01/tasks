
"""
HPSAQ templates
"""


SPANISH_QA_TEMPLATE = "\nPregunta:\n{question}\n\nRespuesta:\n{answer}\n"

SPANISH_Q_TEMPLATE = "\nPregunta:\n{question}\n\nRespuesta:\n"

SPANISH_SAQ_SYSTEM_PROMPT = "<<SYS>>\nQuiero que respondas la siguiente pregunta sobre Harry Potter y respondas con una respuesta. Te proporcionaré la pregunta, y tú responderás con tu respuesta. Tu respuesta debe ser una sola oración. Ahora te proporcionaré la pregunta.\n<</SYS>>\n\n"

SPANISH_ZERO_SHOT_TEMPLATE = "\nPregunta:\n{question}\n\nRespuesta:\n"

SPANISH_FEW_SHOT_TEMPLATE = "\nPregunta:\n{0}\n\nRespuesta:\n{1}\n\nPregunta:\n{2}\n\nRespuesta:\n{3}\n\nPregunta:\n{4}\n\nRespuesta:\n{5}\n\nPregunta:\n{{question}}\n\nRespuesta:\n"

SPANISH_UNRELATED_FEW_SHOT_TEMPLATE = "\nPregunta:\n¿Cuál planeta de nuestro sistema solar es conocido como el Planeta Rojo?\n\nRespuesta:\nMarte\n\nPregunta:\n¿Quién escribió la obra \"Romeo y Julieta\"?\n\nRespuesta:\nWilliam Shakespeare\n\nPregunta:\nNombra los Grandes Lagos de América del Norte.\n\nRespuesta:\nHuron, Ontario, Míchigan, Erie, Superior\n\nPregunta:\n{question}\n\nRespuesta:\n"

RUSSIAN_QA_TEMPLATE = "\nВопрос:\n{question}\n\nОтвет:\n{answer}\n"

RUSSIAN_Q_TEMPLATE = "\nВопрос:\n{question}\n\nОтвет:\n"

RUSSIAN_SAQ_SYSTEM_PROMPT = "<<SYS>>\nЯ хочу, чтобы ты ответил на следующий вопрос о Гарри Поттере, ответив одним предложением. Я предоставлю вопрос, а ты должен ответить на него. Твой ответ должен быть одним предложением. Теперь я предоставлю вопрос.\n<</SYS>>\n\nВопрос:\n"

RUSSIAN_ZERO_SHOT_TEMPLATE = "\nВопрос:\n{question}\n\nОтвет:\n"

RUSSIAN_FEW_SHOT_TEMPLATE = "\nВопрос:\n{0}\n\nОтвет:\n{1}\n\nВопрос:\n{2}\n\nОтвет:\n{3}\n\nВопрос:\n{4}\n\nОтвет:\n{5}\n\nВопрос:\n{{question}}\n\nОтвет:\n"

RUSSIAN_UNRELATED_FEW_SHOT_TEMPLATE = "\nВопрос:\nКакая планета нашей солнечной системы известна как Красная планета?\n\nОтвет:\nМарс\n\nВопрос:\nКто написал произведение \"Ромео и Джульетта\"?\n\nОтвет:\nУильям Шекспир\n\nВопрос:\nНазовите Великие озёра Северной Америки.\n\nОтвет:\nГурон, Онтарио, Мичиган, Эри, Верхнее\n\nВопрос:\n{question}\n\nОтвет:\n"


CHINESE_QA_TEMPLATE = "\n问题：\n{question}\n\n回答：\n{answer}\n"

CHINESE_Q_TEMPLATE = "\n问题：\n{question}\n\n回答：\n"

CHINESE_SAQ_SYSTEM_PROMPT = "<<SYS>>\n我希望你回答以下关于哈利波特的问题，并给出答案。我会提供问题，你需要给出答案。你的回答应该是一句话。我现在会提供问题。\n<</SYS>>\n\n"

CHINESE_ZERO_SHOT_TEMPLATE = "\n问题：\n{question}\n\n回答：\n"

CHINESE_FEW_SHOT_TEMPLATE = "\n问题：\n{0}\n\n回答：\n{1}\n\n问题：\n{2}\n\n回答：\n{3}\n\n问题：\n{4}\n\n回答：\n{5}\n\n问题：\n{{question}}\n\n回答：\n"

CHINESE_UNRELATED_FEW_SHOT_TEMPLATE = "\n问题：\n我们太阳系中哪个行星被称为红色行星？\n\n回答：\n火星\n\n问题：\n谁写了《罗密欧与朱丽叶》这部剧本？\n\n回答：\n威廉·莎士比亚\n\n问题：\n说出北美洲的五大湖。\n\n回答：\n休伦湖，安大略湖，密歇根湖，伊利湖，苏必利尔湖\n\n问题：\n{question}\n\n回答：\n"


"""
HPTrivia templates
"""

SPANISH_TRIVIA_SYSTEM_PROMPT = "Eres un asistente útil, respetuoso y honesto. Dada la siguiente pregunta de trivia, responde con la letra de la opción correcta, ya sea A o B."

RUSSIAN_TRIVIA_SYSTEM_PROMPT = "Ты полезный, уважительный и честный помощник. Для следующего вопроса викторины ответь буквой правильного варианта ответа, будь то A или B."

CHINESE_TRIVIA_SYSTEM_PROMPT = "你是一个有帮助、尊重和诚实的助手。针对以下的小问题，请用正确选项的字母回答，可以是A或者B。" 