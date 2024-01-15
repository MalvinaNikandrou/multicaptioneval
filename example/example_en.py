from pycocotools.coco import COCO

from multicaptioneval.eval import COCOEvalCap


def evaluate(
    annotation_file: str = "tests/fixtures/en_captions_val2014.json",
    results_file: str = "tests/fixtures/en_captions_val2014_fakecap_results.json",
) -> dict[str, float]:
    """Compute the scores for the given annotation and results files."""
    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    coco_eval.params["image_id"] = coco_result.getImgIds()

    # evaluate results
    coco_eval.evaluate()
    return coco_eval.eval


if __name__ == "__main__":
    scores = evaluate()
    print(scores)
