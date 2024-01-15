#!/usr/bin/env python
# Tsung-Yi Lin <tl483@cornell.edu>
# Ramakrishna Vedantam <vrama91@vt.edu>

from collections import defaultdict
import numpy as np
import math
from multicaptioneval.metrics.cider.data import CiderData, NgramType, NgramCountType
from typing import Union


VectorType = list[dict[NgramType, float]]


class CiderMetric:
    def __init__(self, ngram_n: int, sigma: float, multiplier: float = 10) -> None:
        self._ngram_n = ngram_n
        self._sigma = sigma
        self._multiplier = multiplier

    def compute_doc_freq(self, refrences: list[list[NgramCountType]]) -> None:
        """Compute term frequency for reference data.

        This will be used to compute idf (inverse document frequency later).
        """
        document_frequency = defaultdict(float)
        for refs in refrences:
            # refs, k ref captions of one image
            for ngram in set([ngram for ref in refs for ngram in ref.keys()]):
                document_frequency[ngram] += 1
        self.document_frequency = document_frequency

    def counts2vec(self, counts: NgramCountType) -> tuple[VectorType, list[float], int]:
        """Function maps counts of ngram to vector of tfidf weights.

        The function returns vec, an array of dictionary that store mapping of n-gram and tf-idf weights.
        The n-th entry of array denotes length of n-grams.
        :param counts:
        :return: vector (array of dict), norm (array of float), length (int)
        """
        vector = [defaultdict(float) for _ in range(self._ngram_n)]
        norm = [0.0 for _ in range(self._ngram_n)]
        length = 0
        for ngram, term_freq in counts.items():
            # give word count 1 if it doesn't appear in reference corpus
            df = np.log(max(1.0, self.document_frequency[ngram]))
            # ngram index
            ngram_n = len(ngram) - 1
            # tf (term_freq) * idf (precomputed idf) for n-grams
            vector[ngram_n][ngram] = float(term_freq) * (self.ref_len - df)
            # compute norm for the vector. the norm will be used for computing similarity.
            norm[ngram_n] += pow(vector[ngram_n][ngram], 2)

            if ngram_n == 1:
                length += term_freq
        norm = [np.sqrt(n) for n in norm]
        return vector, norm, length

    def sim(
        self,
        vec_hyp: VectorType,
        vec_ref: VectorType,
        norm_hyp: list[float],
        norm_ref: list[float],
        length_hyp: int,
        length_ref: int,
    ) -> np.ndarray:
        """
        Compute the cosine similarity of two vectors.
        :param vec_hyp: array of dictionary for vector corresponding to hypothesis
        :param vec_ref: array of dictionary for vector corresponding to reference
        :param norm_hyp: array of float for vector corresponding to hypothesis
        :param norm_ref: array of float for vector corresponding to reference
        :param length_hyp: int containing length of hypothesis
        :param length_ref: int containing length of reference
        :return: array of score for each n-grams cosine similarity
        """
        delta = float(length_hyp - length_ref)
        # measure consine similarity
        val = np.zeros(self._ngram_n, dtype=np.float32)
        for ngram_n in range(self._ngram_n):
            # ngram
            for ngram in vec_hyp[ngram_n].keys():
                # vrama91 : added clipping
                val[ngram_n] += min(vec_hyp[ngram_n][ngram], vec_ref[ngram_n][ngram]) * vec_ref[ngram_n][ngram]

            if (norm_hyp[ngram_n] != 0) and (norm_ref[ngram_n] != 0):
                val[ngram_n] /= norm_hyp[ngram_n] * norm_ref[ngram_n]

            assert not math.isnan(val[ngram_n])
            # vrama91: added a length based gaussian penalty
            val[ngram_n] *= np.e ** (-(delta**2) / (2 * self._sigma**2))
        return val

    def __call__(self, crefs, ctest) -> list[float]:
        # compute idf
        self.compute_doc_freq(crefs)
        # compute log reference length
        self.ref_len = np.log(float(len(crefs)))
        # assert to check document frequency
        assert len(ctest) >= max(self.document_frequency.values())
        scores = []
        for test, refs in zip(ctest, crefs):
            # append score of an image to the score list
            scores.append(self._compute_score_for_image(test, refs))
        return scores

    def _compute_score_for_image(self, test, refs) -> float:
        # compute vector for test captions
        vec, norm, length = self.counts2vec(test)
        # compute vector for ref captions
        score = np.zeros(self._ngram_n, dtype=np.float32)
        for ref in refs:
            vec_ref, norm_ref, length_ref = self.counts2vec(ref)
            score += self.sim(vec, vec_ref, norm, norm_ref, length, length_ref)
        # change by vrama91 - mean of ngram scores, instead of sum
        score_avg = np.mean(score)
        # divide by number of references
        score_avg /= len(refs)
        # multiply score by the multiplier(10)
        return score_avg * self._multiplier


class CiderScorer:
    """CIDEr scorer."""

    def __init__(self, ngram_n=4, sigma=6.0):
        self._ngram_n = ngram_n
        self.sigma = sigma
        self.data = CiderData(ngram_n=ngram_n)
        self.cider = CiderMetric(ngram_n=ngram_n, sigma=sigma)

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

    def compute(self) -> tuple[float, np.ndarray]:
        # compute cider scores
        scores = self.cider(crefs=self.data.references, ctest=self.data.hypotheses)
        return np.mean(np.array(scores)), np.array(scores)
