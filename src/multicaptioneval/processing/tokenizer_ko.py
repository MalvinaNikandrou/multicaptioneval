import spacy
import mecab_ko as MeCab
import mecab_ko_dic
from typing import Optional
from multicaptioneval.processing.tokenizer_base import BaseTokenizer


class KoreanTokenizer(BaseTokenizer):
    def __init__(self, word_segmenter: Optional[str] = None, **kwargs) -> None:
        if word_segmenter is None or word_segmenter == "mecab":
            self._tokenizer = MeCab.Tagger(mecab_ko_dic.MECAB_ARGS + " -Owakati").parse
        elif word_segmenter == "rule-based":
            self._tokenizer = spacy.blank(
                "ko",
                config={"nlp": {"tokenizer": {"@tokenizers": "spacy.Tokenizer.v1"}}},
            )

    def tokenize(self, text: str) -> str:
        tokenized_text = self._tokenizer(text)
        if isinstance(tokenized_text, str):
            return tokenized_text.strip()
        return " ".join([token.text for token in tokenized_text])
