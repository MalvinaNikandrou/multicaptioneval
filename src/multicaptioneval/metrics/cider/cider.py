# Filename: cider.py
#
# Description: Describes the class to compute the CIDEr (Consensus-Based Image Description Evaluation) Metric
#               by Vedantam, Zitnick, and Parikh (http://arxiv.org/abs/1411.5726)
#
# Creation Date: Sun Feb  8 14:16:54 2015
#
# Authors: Ramakrishna Vedantam <vrama91@vt.edu> and Tsung-Yi Lin <tl483@cornell.edu>

from multicaptioneval.metrics.cider.cider_scorer import CiderScorer


class Cider:
    """
    Main Class to compute the CIDEr metric

    """

    def __init__(self, ngram_n: int = 4, sigma: float = 6.0) -> None:
        # set cider to sum over 1 to 4-grams
        self._ngram_n = ngram_n
        # set the standard deviation parameter for gaussian penalty
        self._sigma = sigma

    def compute_score(self, ground_truths, results):
        """
        Main function to compute CIDEr score
        :param  hypo_for_image (dict) : dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
                ref_for_image (dict)  : dictionary with key <image> and value <tokenized reference sentence>
        :return: cider (float) : computed CIDEr score for the corpus
        """

        assert ground_truths.keys() == results.keys()
        imgIds = ground_truths.keys()

        cider_scorer = CiderScorer(ngram_n=self._ngram_n, sigma=self._sigma)

        for image_id in imgIds:
            hypothesis = results[image_id]
            references = ground_truths[image_id]

            # Sanity check.
            assert isinstance(hypothesis, list)
            assert len(hypothesis) == 1
            assert isinstance(references, list)
            assert len(references) > 0

            cider_scorer.update(hypotheses=hypothesis[0], references=references)

        (score, scores) = cider_scorer.compute()

        return score, scores

    @property
    def method(self) -> str:
        return "CIDEr"

    @property
    def score_names(self) -> list[str]:
        return "CIDEr"
