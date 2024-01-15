from spacy.lang.ja import Japanese
import MeCab
import ipadic
from typing import Optional
from multicaptioneval.processing.tokenizer_base import BaseTokenizer


class JapaneseTokenizer(BaseTokenizer):
    def __init__(self, word_segmenter: Optional[str] = "sudachi", **kwargs) -> None:
        if word_segmenter is None or word_segmenter == "sudachi":
            self._tokenizer = Japanese()
        elif word_segmenter == "mecab":
            self._tokenizer = MeCab.Tagger(ipadic.MECAB_ARGS + " -Owakati").parse

    def tokenize(self, text: str) -> str:
        tokenized_text = self._tokenizer(text)
        if isinstance(tokenized_text, str):
            return tokenized_text.strip()
        return " ".join([token.text for token in tokenized_text])
