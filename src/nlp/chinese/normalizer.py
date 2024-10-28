import re

import cn2an

from style_bert_vits2.nlp.symbols import PUNCTUATIONS


def normalize_text(text: str) -> str:
    numbers = re.findall(r"\d+(?:\.?\d+)?", text)
    for number in numbers:
        text = text.replace(number, cn2an.an2cn(number), 1)
    text = replace_punctuation(text)
    return text


def replace_punctuation(text: str) -> str:

    REPLACE_MAP = {
        "：": ",",
        "；": ",",
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "\n": ".",
        "·": ",",
        "、": ",",
        "...": "…",
        "$": ".",
        "“": "'",
        "”": "'",
        '"': "'",
        "‘": "'",
        "’": "'",
        "（": "'",
        "）": "'",
        "(": "'",
        ")": "'",
        "《": "'",
        "》": "'",
        "【": "'",
        "】": "'",
        "[": "'",
        "]": "'",
        "—": "-",
        "～": "-",
        "~": "-",
        "「": "'",
        "」": "'",
    }

    text = text.replace("嗯", "恩").replace("呣", "母")
    pattern = re.compile("|".join(re.escape(p) for p in REPLACE_MAP.keys()))

    replaced_text = pattern.sub(lambda x: REPLACE_MAP[x.group()], text)

    replaced_text = re.sub(
        r"[^\u4e00-\u9fa5" + "".join(PUNCTUATIONS) + r"]+", "", replaced_text
    )

    return replaced_text
