import sacrebleu
import numpy as np
from typing import Optional

from pycocoevalcap.cider.cider import Cider as PyCOCOCider
from pycocoevalcap.bleu.bleu import Bleu as PyCOCOBLEU

from multicaptioneval.metrics.cider.cider import Cider as MultiCaptionCider
from multicaptioneval.metrics.bleu.bleu import Bleu as MultiCaptionBLEU


def test_cider(results: dict[str, list[str]], references: dict[str, list[list[str]]]) -> None:
    """Verify CIDEr results against pycocoevalcap."""
    pycococider = PyCOCOCider()
    pycoco_score, pycoco_scores = pycococider.compute_score(gts=references, res=results)
    cider = MultiCaptionCider()
    cider_score, cider_scores = cider.compute_score(ground_truths=references, results=results)
    assert round(pycoco_score, 6) == round(cider_score, 6)
    assert np.allclose(pycoco_scores, cider_scores)


def test_bleu_against_scarebleu(
    results: dict[str, list[str]],
    references: dict[str, list[list[str]]],
    sacrebleu_results: list[str],
    sacrebleu_references: list[list[Optional[str]]],
) -> None:
    """Verify BLEU results against sacrebleu."""
    bleu = MultiCaptionBLEU()
    bleu_score, _ = bleu.compute_score(ground_truths=references, results=results)
    for ngram_n in range(bleu._ngram_n):
        metric = sacrebleu.BLEU(
            tokenize="none",
            max_ngram_order=ngram_n + 1,
        )
        sacrebleu_score = metric.corpus_score(sacrebleu_results, sacrebleu_references).score
        multicap_score = 100 * bleu_score[ngram_n]
        # Check the corpus-level BLEU score.
        assert round(sacrebleu_score, 6) == round(multicap_score, 6)


def test_bleu_against_pycocobleu(results: dict[str, list[str]], references: dict[str, list[list[str]]]) -> None:
    """Verify BLEU results against pycocoevalcap."""
    pycococbleu = PyCOCOBLEU()
    pycococbleu_score, pycococbleu_scores = pycococbleu.compute_score(gts=references, res=results)
    multicapbleu = MultiCaptionBLEU()
    multicapbleu_score, multicapbleu_scores = multicapbleu.compute_score(ground_truths=references, results=results)
    # Check the corpus-level BLEU score.
    assert np.allclose(pycococbleu_score, multicapbleu_score)
    # Check the score for each image.
    for ngram, scores in enumerate(pycococbleu_scores):
        np.allclose(scores, multicapbleu_scores[ngram])
