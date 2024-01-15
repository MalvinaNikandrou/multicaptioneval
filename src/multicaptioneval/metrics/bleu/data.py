from collections import defaultdict
from typing import Optional
from pydantic import (
    BaseModel,
    Field,
    ValidationInfo,
    field_validator,
)
from typing import Annotated

NgramType = tuple[str, ...]
NgramCountType = dict[NgramType, int]


class BleuNgramCounts(BaseModel):
    length: int
    max_ngram_counts: NgramCountType


class BleuReferences(BaseModel):
    lengths: list[int] = []
    max_ngram_counts: NgramCountType = {}

    def update(self, other: BleuNgramCounts) -> None:
        self.lengths.append(other.length)
        for ngram, count in other.max_ngram_counts.items():
            self.max_ngram_counts[ngram] = max(self.max_ngram_counts.get(ngram, 0), count)


class BleuHypothesisStats(BaseModel):
    referene_lengths: list[int]
    length: Annotated[int, Field(ge=0)]
    total_ngrams: list[int]
    correct_ngrams: list[int]

    @field_validator("total_ngrams", "correct_ngrams")
    @classmethod
    def validate_ngram(cls, ngrams_list: str, info: ValidationInfo):
        context = info.context
        if context:
            assert len(ngrams_list) == context.get("ngram_n", 4)
        return ngrams_list


class BleuStatsCounter:
    def __init__(self, max_ngram: int = 4) -> None:
        self.max_ngram = max_ngram

    def cook_references(self, references: list[str]) -> BleuReferences:  # lhuang: oracle will call with "average"
        """Takes a list of reference sentences for a single segment
        and returns an object that encapsulates everything that BLEU
        needs to know about them."""
        processed_references = BleuReferences()
        for refrence in references:
            processed_references.update(self._precook(refrence))
        return processed_references

    def cook_test(self, test: str, references: BleuReferences) -> BleuHypothesisStats:
        """Takes a test sentence and returns an object that
        encapsulates everything that BLEU needs to know about it."""
        hypothesis = self._precook(test)

        total_ngrams = [max(0, hypothesis.length - k + 1) for k in range(1, self.max_ngram + 1)]
        stats = BleuHypothesisStats(
            length=hypothesis.length,
            referene_lengths=references.lengths,
            total_ngrams=total_ngrams,
            correct_ngrams=[0] * self.max_ngram,
        )
        ref_counts = references.max_ngram_counts
        for ngram, count in hypothesis.max_ngram_counts.items():
            stats.correct_ngrams[len(ngram) - 1] += min(ref_counts.get(ngram, 0), count)

        try:
            BleuHypothesisStats.model_validate(stats.model_dump(), context={"ngram_n": self.max_ngram})
        except Exception as e:
            breakpoint()

        return stats

    def _precook(self, text: str) -> BleuNgramCounts:
        """Takes a string as input and returns an object that can be given to
        either cook_refs or cook_test. This is optional: cook_refs and cook_test
        can take string arguments as well."""
        words = text.split()
        counts = defaultdict(int)
        for ngram_n in range(1, self.max_ngram + 1):
            for ngram_start_index in range(len(words) - ngram_n + 1):
                ngram = tuple(words[ngram_start_index : ngram_start_index + ngram_n])
                counts[ngram] += 1
        return BleuNgramCounts(length=len(words), max_ngram_counts=counts)


class BleuData:
    """Data to compute the BLEU metric.

    The data are preprocessed to count the ngrams in the hypotheses and references.
    """

    def __init__(self, max_ngram: int = 4) -> None:
        self._ngram_counter = BleuStatsCounter(max_ngram)
        self.references: list[BleuReferences] = []
        self.hypotheses: list[Optional[BleuHypothesisStats]] = []

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
    def size(self) -> int:
        if len(self.references) != len(self.hypotheses):
            raise AssertionError(f"refs/test mismatch! {len(self.references)}<>{len(self.hypotheses)}")
        return len(self.references)

    def cook_append(self, hypothesis: str, references: list[str]) -> None:
        """called by constructor and __iadd__ to avoid creating new instances."""
        if references is not None:
            self.references.append(self._ngram_counter.cook_references(references))
            if hypothesis is not None:
                self.hypotheses.append(self._ngram_counter.cook_test(hypothesis, self.references[-1]))
            else:
                self.hypotheses.append(None)
