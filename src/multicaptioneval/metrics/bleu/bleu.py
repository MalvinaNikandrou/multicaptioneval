#!/usr/bin/env python
#
# File Name : bleu.py
#
# Description : Wrapper for BLEU scorer.
#
# Creation Date : 06-01-2015
# Last Modified : Thu 19 Mar 2015 09:13:28 PM PDT
# Authors : Hao Fang <hfang@uw.edu> and Tsung-Yi Lin <tl483@cornell.edu>

from multicaptioneval.metrics.bleu.bleu_scorer import BleuScorer


class Bleu:
    def __init__(self, ngram_n=4):
        # default compute Blue score up to 4
        self._ngram_n = ngram_n
        self._hypo_for_image = {}
        self.ref_for_image = {}

    def compute_score(self, ground_truths, results):
        assert ground_truths.keys() == results.keys()
        imgIds = ground_truths.keys()

        bleu_scorer = BleuScorer(max_ngram=self._ngram_n)
        for image_id in imgIds:
            hypothesis = results[image_id]
            references = ground_truths[image_id]

            # Sanity check.
            assert isinstance(hypothesis, list)
            assert len(hypothesis) == 1
            assert isinstance(references, list)
            assert len(references) > 0
            bleu_scorer.update(hypothesis[0], references)

        score, scores = bleu_scorer.compute(option="closest")

        return score, scores

    @property
    def method(self) -> str:
        return "Bleu"

    @property
    def score_names(self) -> list[str]:
        return [f"Bleu_{n}" for n in range(1, self._ngram_n + 1)]
