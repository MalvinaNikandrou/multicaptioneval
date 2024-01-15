from pycocotools.coco import COCO
from tabulate import tabulate

from multicaptioneval.eval import COCOEvalCap


# Each case has the following format:
# (annotation_file, results_file, language, tokenizer_cfg)
cases = [
    (
        "tests/fixtures/en_captions_val2014.json",
        "tests/fixtures/en_captions_val2014_fakecap_results.json",
        "en",
        {},
    ),
    (
        "tests/fixtures/ar_captions_val2014.json",
        "tests/fixtures/ar_captions_val2014_fakecap_results.json",
        "ar",
        {},
    ),
    (
        "tests/fixtures/el_captions_val2014.json",
        "tests/fixtures/el_captions_val2014_fakecap_results.json",
        "el",
        {},
    ),
    (
        "tests/fixtures/fr_captions_val2014.json",
        "tests/fixtures/fr_captions_val2014_fakecap_results.json",
        "fr",
        {},
    ),
    (
        "tests/fixtures/ja_captions_val2014.json",
        "tests/fixtures/ja_captions_val2014_fakecap_results.json",
        "ja",
        {"word_segmenter": "sudachi"},
    ),
    (
        "tests/fixtures/ja_captions_val2014.json",
        "tests/fixtures/ja_captions_val2014_fakecap_results.json",
        "ja",
        {"word_segmenter": "mecab"},
    ),
    (
        "tests/fixtures/ko_captions_val2014.json",
        "tests/fixtures/ko_captions_val2014_fakecap_results.json",
        "ko",
        {"word_segmenter": "mecab"},
    ),
    (
        "tests/fixtures/ko_captions_val2014.json",
        "tests/fixtures/ko_captions_val2014_fakecap_results.json",
        "ko",
        {"word_segmenter": "rule-based"},
    ),
    (
        "tests/fixtures/th_captions_val2014.json",
        "tests/fixtures/th_captions_val2014_fakecap_results.json",
        "th",
        {"word_segmenter": "char"},
    ),
    (
        "tests/fixtures/th_captions_val2014.json",
        "tests/fixtures/th_captions_val2014_fakecap_results.json",
        "th",
        {"word_segmenter": "spacy"},
    ),
    (
        "tests/fixtures/zh_captions_val2014.json",
        "tests/fixtures/zh_captions_val2014_fakecap_results.json",
        "zh",
        {"word_segmenter": "char"},
    ),
    (
        "tests/fixtures/zh_captions_val2014.json",
        "tests/fixtures/zh_captions_val2014_fakecap_results.json",
        "zh",
        {"word_segmenter": "jieba"},
    ),
]


def evaluate(
    annotation_file: str,
    results_file: str,
    language: str,
    tokenizer_cfg: dict[str, str],
) -> dict[str, float]:
    """Compute the scores for the given annotation and results files."""
    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result, language=language, tokenizer_cfg=tokenizer_cfg)

    # evaluate on a subset of images by setting
    coco_eval.params["image_id"] = coco_result.getImgIds()

    # evaluate results
    coco_eval.evaluate()
    return coco_eval.eval


def get_results_table() -> str:
    """Get a table of the results for all languages.


    +----------+------------+--------+--------+--------+--------+--------+
    | Language | Tokenizer  | Bleu_1 | Bleu_2 | Bleu_3 | Bleu_4 | CIDEr  |
    +----------+------------+--------+--------+--------+--------+--------+
    |    en    |    PTB     | 0.5794 | 0.4044 | 0.2785 | 0.1908 | 0.5998 |
    |    ar    |    PTB     | 0.4207 | 0.2644 | 0.164  | 0.1028 | 0.3716 |
    |    el    |    PTB     | 0.4874 | 0.3215 | 0.2068 | 0.1347 | 0.3896 |
    |    fr    |    PTB     | 0.5241 | 0.3529 | 0.2455 | 0.1698 | 0.491  |
    |    ja    |  sudachi   | 0.619  | 0.4418 | 0.3216 | 0.2367 | 0.4801 |
    |    ja    |   mecab    | 0.6189 | 0.4431 | 0.3232 | 0.2384 | 0.4863 |
    |    ko    |   mecab    | 0.5908 | 0.4421 | 0.3337 | 0.2531 | 0.4823 |
    |    ko    | rule-based | 0.343  | 0.1955 | 0.1097 | 0.0607 | 0.2261 |
    |    th    |    char    | 0.8434 | 0.6607 | 0.5444 | 0.4633 | 0.6567 |
    |    th    |   spacy    | 0.4516 | 0.2827 | 0.1857 | 0.1248 | 0.4437 |
    |    zh    |    char    | 0.5992 | 0.4361 | 0.3122 | 0.2275 | 0.6524 |
    |    zh    |   jieba    | 0.5028 | 0.2974 | 0.1795 | 0.1085 | 0.3837 |
    +----------+------------+--------+--------+--------+--------+--------+
    """
    metrics = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "CIDEr"]
    results = [["Language", "Tokenizer"] + metrics]
    for case in cases:
        # Add the language to the results row
        row = [case[2]]
        # Add the tokenizer to the results row
        if case[-1]:
            row.append(case[-1]["word_segmenter"])
        else:
            row.append("PTB")

        # Add the scores to the results row
        scores = evaluate(*case)
        for metric in metrics:
            row.append(round(scores[metric], 4))
        results.append(row)

    return tabulate(results, headers="firstrow", tablefmt="pretty")


if __name__ == "__main__":
    table = get_results_table()
    print(table)
