import pytest

from pycocotools.coco import COCO
from multicaptioneval.eval import COCOEvalCap


@pytest.mark.parametrize(
    "annotation_file,results_file,language",
    [
        (
            "tests/fixtures/en_captions_val2014.json",
            "tests/fixtures/en_captions_val2014_fakecap_results.json",
            "en",
        ),
        (
            "tests/fixtures/ar_captions_val2014.json",
            "tests/fixtures/ar_captions_val2014_fakecap_results.json",
            "ar",
        ),
        (
            "tests/fixtures/el_captions_val2014.json",
            "tests/fixtures/el_captions_val2014_fakecap_results.json",
            "el",
        ),
        (
            "tests/fixtures/fr_captions_val2014.json",
            "tests/fixtures/fr_captions_val2014_fakecap_results.json",
            "fr",
        ),
        (
            "tests/fixtures/ja_captions_val2014.json",
            "tests/fixtures/ja_captions_val2014_fakecap_results.json",
            "ja",
        ),
        (
            "tests/fixtures/ko_captions_val2014.json",
            "tests/fixtures/ko_captions_val2014_fakecap_results.json",
            "ko",
        ),
        (
            "tests/fixtures/th_captions_val2014.json",
            "tests/fixtures/th_captions_val2014_fakecap_results.json",
            "th",
        ),
        (
            "tests/fixtures/zh_captions_val2014.json",
            "tests/fixtures/zh_captions_val2014_fakecap_results.json",
            "zh",
        ),
    ],
)
def test_eval(annotation_file: str, results_file: str, language: str) -> None:
    """Make sure the evaluation runs for different languages."""
    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result, language=language)

    # evaluate on a subset of images by setting
    coco_eval.params["image_id"] = coco_result.getImgIds()

    # evaluate results
    coco_eval.evaluate()
    assert len(coco_eval.imgToEval) == len(coco_result.imgs)
    scores = coco_eval.eval.items()
    assert scores
