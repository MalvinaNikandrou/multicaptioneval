import spacy
from typing import Optional
from multicaptioneval.processing.tokenizer_base import BaseTokenizer


class ThaiTokenizer(BaseTokenizer):
    def __init__(self, word_segmenter: Optional[str] = None, **kwargs) -> None:
        if word_segmenter is None or word_segmenter == "spacy":
            self._tokenizer = spacy.blank("th")
        elif word_segmenter == "char":
            self._tokenizer = self._char_level

    def _char_level(self, text: str) -> str:
        return " ".join(list(text))

    def tokenize(self, text: str) -> str:
        tokenized_text = self._tokenizer(text)
        if isinstance(tokenized_text, str):
            return tokenized_text.strip()
        return " ".join([token.text for token in tokenized_text])
