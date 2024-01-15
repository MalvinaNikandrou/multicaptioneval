from functools import lru_cache

import unicodedata
from typing import Literal


@lru_cache(maxsize=2**16)
def normalize_unicode(input_str: str, form: Literal["NFC", "NFD", "NFKC", "NFKD"] = "NFKC") -> str:
    return unicodedata.normalize(form, input_str)


PUNCTUATIONS = [
    ".",
    "...",
    "·",
    ",",
    "。",
    "、",
    ",",
    ":",
    ";",
    "?",
    "!",
    "''",
    "'",
    "``",
    "`",
    '"',
    "-",
    "--",
    "_",
    "/",
    "\\",
    "《",
    "》",
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    ">",
    "<",
    "=",
    "+",
    "@",
    "#",
    "%",
    "&",
    "*",
]


@lru_cache(maxsize=2**16)
def remove_punctuation(input_str: str) -> str:
    return " ".join([c for c in input_str.split() if c not in PUNCTUATIONS])
