#!/usr/bin/env python

# bleu_scorer.py
# David Chiang <chiang@isi.edu>

# Copyright (c) 2004-2006 University of Maryland. All rights
# reserved. Do not redistribute without permission from the
# author. Not for commercial use.

# Modified by:
# Hao Fang <hfang@uw.edu>
# Tsung-Yi Lin <tl483@cornell.edu>

import math
from typing import Any, Literal, Union
from multicaptioneval.metrics.bleu.data import BleuData

SMALL_EPS = 1e-9
TINY_EPS = 1e-15
OPTIONS = Literal["closest", "average", "shortest"]


class BleuScorer(object):
    """Bleu scorer."""

    __slots__ = (
        "max_ngram",
        "data",
        "_score",
        "_brevity_penalty",
        "_testlen",
        "_reflen",
        "special_reflen",
    )
    # special_reflen is used in oracle (proportional effective ref len for a node).

    def __init__(self, max_ngram=4, special_reflen=None):
        """singular instance"""
        self.max_ngram = max_ngram
        self.data = BleuData(max_ngram=max_ngram)
        self.special_reflen = special_reflen
        self._score = None

    def update(
        self,
        hypotheses: Union[str, list[str]],
        references: Union[list[str], list[list[str]]],
    ) -> None:
        """Update with the latest set of hypothesis and references."""
        if isinstance(hypotheses, str) and isinstance(references, list):
            self.data.add(hypotheses, references)
        else:
            self.data.add_data(hypotheses, references)

    def _single_reflen(self, reflens: list[int], option=None, testlen: int = 0) -> float:
        if option == "shortest":
            reflen = min(reflens)
        elif option == "average":
            reflen = float(sum(reflens)) / len(reflens)
        elif option == "closest":
            reflen = self._get_closest_ref_len(testlen=testlen, reflens=reflens)
        else:
            assert False, f"unsupported reflen option {option} with test length {testlen}"

        return float(reflen)

    def _get_closest_ref_len(self, testlen: int, reflens: list[int]) -> int:
        """Get the reference length that is closest to the hypothesis length."""
        diffs = [(abs(testlen - reflen), reflen) for reflen in reflens]
        min_diff = min(diffs)
        # Similar to sacrebleu if there's a tie of closest lenghts, pick the shortest one
        closest_lens = [reflen for reflen, diff in zip(reflens, diffs) if diff == min_diff]
        return min(closest_lens)

    def compute(self, option: OPTIONS = "closest"):
        if self._score is not None:
            return self._score

        max_ngram = self.max_ngram
        bleu_list = [[] for _ in range(max_ngram)]
        if option is None:
            option = "average" if len(self.data.references) == 1 else "closest"

        totalstats = {
            "testlen": 0,
            "reflen": 0,
            "total": [0] * max_ngram,
            "correct": [0] * max_ngram,
        }

        # for each sentence
        for stats in self.data.hypotheses:
            if stats is None:
                continue

            testlen = stats.length
            if self.special_reflen is None:  # need computation
                reflen = self._single_reflen(stats.referene_lengths, option, testlen)
            else:
                reflen = self.special_reflen

            # append per image bleu score
            image_bleu = self.compute_bleu(
                correct=stats.correct_ngrams,
                total=stats.total_ngrams,
                testlen=testlen,
                reflen=reflen,
            )
            for ngram_n in range(max_ngram):
                bleu_list[ngram_n].append(image_bleu[ngram_n])

            # Aggregate statistics
            totalstats["testlen"] += testlen
            totalstats["reflen"] += reflen
            for ngram_n in range(max_ngram):
                totalstats["correct"][ngram_n] += stats.correct_ngrams[ngram_n]
                totalstats["total"][ngram_n] += stats.total_ngrams[ngram_n]

        self._reflen = totalstats["reflen"]
        self._testlen = totalstats["testlen"]
        self._brevity_penalty = self.brevity_penalty
        self._score = self.aggregate_bleu_scores(totalstats)
        return self._score, bleu_list

    def aggregate_bleu_scores(self, totalstats: dict[str, Any]) -> list[float]:
        return self.compute_bleu(
            correct=totalstats["correct"],
            total=totalstats["total"],
            testlen=totalstats["testlen"],
            reflen=totalstats["testlen"],
        )

    def compute_bleu(self, correct: list[int], total: list[int], testlen: int, reflen: float) -> list[float]:
        """Compute BLEU score from collected statistics."""
        bleu = 1.0
        bleus = []
        for ngram_n in range(self.max_ngram):
            correct_n = float(correct[ngram_n]) + TINY_EPS
            total_n = float(total[ngram_n]) + SMALL_EPS
            bleu *= correct_n / total_n
            bleus.append(bleu ** (1.0 / (ngram_n + 1)))
        # Brevity Penalty
        if testlen < reflen and testlen > 0:
            for ngram_n in range(self.max_ngram):
                bleus[ngram_n] *= math.exp(1 - (reflen / testlen))
        return bleus

    @property
    def brevity_penalty(self) -> float:
        """Get brevity penalty."""
        if self._testlen == 0:
            return 0.0
        elif self._testlen < self._reflen:
            return math.exp(1 - self._reflen / self._testlen)
        return 1.0
