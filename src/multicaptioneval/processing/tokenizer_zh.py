from typing import Optional
from spacy.lang.zh import Chinese
from multicaptioneval.processing.tokenizer_base import BaseTokenizer


class ChineseTokenizer(BaseTokenizer):
    def __init__(self, word_segmenter: Optional[str] = None, **kwargs) -> None:
        if word_segmenter is None or word_segmenter == "char":
            self._tokenizer = Chinese()
        elif word_segmenter == "jieba":
            cfg = {"segmenter": "jieba"}
            self._tokenizer = Chinese.from_config({"nlp": {"tokenizer": cfg}})
        elif word_segmenter == "pkuseg":
            cfg = {"segmenter": "pkuseg"}
            self._tokenizer = Chinese.from_config({"nlp": {"tokenizer": cfg}})
            self._tokenizer.tokenizer.initialize(pkuseg_model="mixed")

    def tokenize(self, text: str) -> str:
        return " ".join([token.text for token in self._tokenizer(text)])
