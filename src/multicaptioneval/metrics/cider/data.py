#!/usr/bin/env python
# Tsung-Yi Lin <tl483@cornell.edu>
# Ramakrishna Vedantam <vrama91@vt.edu>

from collections import defaultdict
from typing import Optional, Union

NgramType = tuple[str]
NgramCountType = dict[NgramType, int]


class CiderNgramCounter:
    def __init__(self, max_ngram: int = 4) -> None:
        self.max_ngram = max_ngram

    def __call__(self, text: Union[str, list[str]]) -> Union[NgramCountType, list[NgramCountType]]:
        if isinstance(text, str):
            return self.cook_text(text)
        elif isinstance(text, list):
            return self.cook_text_list(text)
        else:
            raise TypeError(f"must be str or list[str]: {type(text)}")

    def cook_text_list(
        self, refs: list[str], max_ngram=4
    ) -> list[NgramCountType]:  # lhuang: oracle will call with "average"
        """Takes a list of reference sentences for a single segment
        and returns an object that encapsulates everything that BLEU
        needs to know about them.
        :param refs: list of string : reference sentences for some image
        :param max_ngram: int : number of ngrams for which (ngram) representation is calculated
        :return: result (list of dict)
        """
        return [self._precook(ref, max_ngram) for ref in refs]

    def cook_text(self, text: str, max_ngram=4) -> NgramCountType:
        """Takes a sentence and returns an object that
        encapsulates everything that BLEU needs to know about it.
        :param text: list of string : hypothesis sentence for some image
        :param max_ngram: int : number of ngrams for which (ngram) representation is calculated
        :return: result (dict)
        """
        return self._precook(text, max_ngram)

    def _precook(self, text: str, max_ngram: int = 4) -> NgramCountType:
        """
        Takes a string as input and returns an object that can be given to
        either cook_text_list or cook_text. This is optional: cook_text_list and cook_text
        can take string arguments as well.
        :param text: string : sentence to be converted into ngrams
        :param max_ngram: int    : number of ngrams for which representation is calculated
        :return: term frequency vector for occuring ngrams
        """
        # TODO: fix for other languages
        words = text.split()
        counts = defaultdict(int)
        for ngram_n in range(1, max_ngram + 1):
            for ngram_start_index in range(len(words) - ngram_n + 1):
                ngram = tuple(words[ngram_start_index : ngram_start_index + ngram_n])
                counts[ngram] += 1
        return counts


class CiderData:
    """Data to compute the CIDEr metric.

    The data are preprocessed to count the ngrams in the hypotheses and references.
    """

    def __init__(self, ngram_n: int = 4) -> None:
        self._ngram_counter = CiderNgramCounter(ngram_n)
        self.references: list[list[NgramCountType]] = []
        self.hypotheses: list[Optional[NgramCountType]] = []

    def add(self, new_hypothesis: str, new_references: list[str]) -> None:
        """Add the hypotheses and references for a single image."""
        if not isinstance(new_hypothesis, str):
            raise TypeError(f"must be str: {type(new_hypothesis)}")
        if not all([isinstance(ref, str) for ref in new_references]):
            raise TypeError(f"must be list[str]: {type(new_references)}")

        self.cook_append(hypothesis=new_hypothesis, references=new_references)

    def add_data(self, hypotheses: list[str], references: list[list[str]]) -> None:
        """Add the hypotheses and references for a multiple images."""
        if not isinstance(hypotheses, list):
            raise TypeError(f"must be list: {type(hypotheses)}")
        assert len(hypotheses) == len(references), f"{len(hypotheses)}<>{len(references)}"
        for hypothesis, refs in zip(hypotheses, references):
            self.cook_append(hypothesis=hypothesis, references=refs)

    @property
    def size(self):
        if len(self.references) != len(self.hypotheses):
            raise AssertionError(f"refs/test mismatch! {len(self.references)}<>{len(self.hypotheses)}")
        return len(self.references)

    def cook_append(self, hypothesis: str, references: list[str]) -> None:
        """called by constructor and __iadd__ to avoid creating new instances."""
        if references is not None:
            self.references.append(self._ngram_counter(references))
            if hypothesis is not None:
                self.hypotheses.append(self._ngram_counter(hypothesis))
            else:
                self.hypotheses.append(None)
