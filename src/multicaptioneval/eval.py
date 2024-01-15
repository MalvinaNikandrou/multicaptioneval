"""
Following the pycocoevalcap implementation, we implement the evaluation for multilingual captions.
"""
from multicaptioneval.metrics.bleu.bleu import Bleu
from multicaptioneval.metrics.cider.cider import Cider
from multicaptioneval.processing import ImageCaptionsType, ProcessingPipeline
import logging
from typing import Any, Optional, Union
from pycocotools.coco import COCO


METRICS = {
    "bleu": Bleu,
    "cider": Cider,
}


MAX_NGRAM_N = 4


class COCOEvalCap:
    def __init__(
        self,
        coco: COCO,
        cocoRes: COCO,
        metrics: Optional[list[str]] = None,
        language: str = "default",
        tokenizer_cfg: Optional[dict[str, Any]] = None,
    ) -> None:
        # image ids to evaluate
        self.evalImgs = []
        # overall evaluation metrics
        self.eval = {}
        # evaluation metrics per image
        self.imgToEval = {}
        self.coco = coco
        self.cocoRes = cocoRes
        self.params = {"image_id": coco.getImgIds()}

        self.language = language
        self._setup_metrics(metrics)
        self._setup_preprocessing(tokenizer_cfg)

    def evaluate(self) -> None:
        """Evaluate the captions."""
        ground_truths, results = self._prepare_data()
        metrics = self._initializa_metrics()
        # Compute scores
        for metric in metrics:
            logging.info(f"Computing {metric.method} score...")
            score, scores = metric.compute_score(ground_truths, results)
            self.print_scores(
                score_names=metric.score_names,
                overall_score=score,
                image_scores=scores,
                image_ids=list(ground_truths.keys()),
            )
        self.set_eval_per_image()

    def get_scores(self):
        return self.eval

    def set_eval(self, score: float, method: str) -> None:
        assert isinstance(score, float) and isinstance(method, str)
        self.eval[method] = score

    def set_image_to_eval(self, scores: list[float], imgIds: list[str], method: str) -> None:
        for imgId, score in zip(imgIds, scores):
            if imgId not in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def set_eval_per_image(self) -> None:
        self.evalImgs = [eval for _, eval in self.imgToEval.items()]

    def print_scores(
        self,
        score_names: Union[str, list[str]],
        overall_score: Union[float, list[float]],
        image_scores: Union[list[float], list[list[float]]],
        image_ids: list[str],
    ) -> None:
        """Print the scores."""
        if isinstance(overall_score, list):
            for score, scores, name in zip(overall_score, image_scores, score_names):
                self.set_eval(score, name)
                self.set_image_to_eval(scores, image_ids, name)
                logging.info(f"{name}: {score:0.3f}")
        else:
            self.set_eval(overall_score, score_names)
            self.set_image_to_eval(image_scores, image_ids, score_names)
            logging.info(f"{score_names}: {overall_score:0.3f}")

    def _setup_metrics(self, metrics: Optional[list[str]]) -> None:
        if metrics is None:
            self.metric_names = ["bleu", "cider"]
        else:
            self.metric_names = []
            for metric in metrics:
                metric = metric.lower
                if metric not in METRICS:
                    raise ValueError(f"Unknown metric: {metric}")
                self.metric_names.append(metric)

    def _setup_preprocessing(self, tokenizer_cfg) -> None:
        self.preprocessing = ProcessingPipeline(
            language=self.language,
            tokenizer_cfg=tokenizer_cfg,
        )

    def _initializa_metrics(self):
        logging.info("Initializa the metrics...")
        return [METRICS[metric](MAX_NGRAM_N) for metric in self.metric_names]

    def _prepare_data(self) -> tuple[ImageCaptionsType, ImageCaptionsType]:
        """Prepare the data for evaluation."""
        imgIds = self.params["image_id"]
        gts = {}
        res = {}
        for imgId in imgIds:
            gts[imgId] = self.coco.imgToAnns[imgId]
            res[imgId] = self.cocoRes.imgToAnns[imgId]
        # Apply the tokenizer
        logging.info("Apply the preprocessing (normalize unicode, tokenize, remove punctuation)...")
        gts = self.preprocessing(gts)
        res = self.preprocessing(res)
        return gts, res
