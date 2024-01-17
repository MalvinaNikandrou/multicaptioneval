# multicaptioneval

# Multilingual Captioning Evaluation

This repository extends [pycocoevalcap](https://github.com/salaniz/pycocoevalcap/tree/master) for evaluation of
multilingual COCO-style datasets.

Inspired by efforts like [sacreBLEU](https://github.com/mjpost/sacrebleu) ([Post, 2018](https://arxiv.org/abs/1804.08771)), which has encouraged reproducible evaluation scores, multicaptioneval aims at facilitating standardized preprocessing and metric computation.
In particular, multicaptioneval can be used to compute BLEU ([Papineni et al., 2002](https://aclanthology.org/P02-1040.pdf)) and CIDEr ([Vedantam et al., 2014](https://arxiv.org/abs/1411.5726)) scores.

In terms of preprocessing, we modify pycocoevalcap steps as follows:
1. Unicode normalization with the [NFKC](https://unicode.org/reports/tr15/#Norm_Forms) normalization form
2. Language-based tokenization: We include language-specific tokenizers for Japanese, Korean, Thai, and Chinese. Other languages use the default tokenizer from pycocoevalcap.
3. Extension of punctuations that are removed from sentences


## Installation

Requirements:
- Python>=3.9

Clone the repository and run:
```bash
pip install -e .
```

## Example
For an example script, see: [example/example_en.py](example/example_en.py)
For results of the example data across languages, see: [example/example.py](example/example.py)

The example results are from [pycocoevalcap](https://github.com/salaniz/pycocoevalcap/tree/master), and have been translated
to 7 other languages (see `tests/fixtures`) using the NLLB-200-distilled-1.3B (NLLB Team, 2022) model. 

The results should look as follows:

| Language   | Tokenizer   |   Bleu_1 |   Bleu_2 |   Bleu_3 |   Bleu_4 |   CIDEr |
|------------|-------------|----------|----------|----------|----------|---------|
| en         | PTB         |   0.5794 |   0.4044 |   0.2785 |   0.1908 |  0.5998 |
| ar         | PTB         |   0.4207 |   0.2644 |   0.164  |   0.1028 |  0.3716 |
| el         | PTB         |   0.4874 |   0.3215 |   0.2068 |   0.1347 |  0.3896 |
| fr         | PTB         |   0.5241 |   0.3529 |   0.2455 |   0.1698 |  0.491  |
| ja         | sudachi     |   0.619  |   0.4418 |   0.3216 |   0.2367 |  0.4801 |
| ja         | mecab       |   0.6189 |   0.4431 |   0.3232 |   0.2384 |  0.4863 |
| ko         | mecab       |   0.5908 |   0.4421 |   0.3337 |   0.2531 |  0.4823 |
| ko         | rule-based  |   0.343  |   0.1955 |   0.1097 |   0.0607 |  0.2261 |
| th         | char        |   0.8434 |   0.6607 |   0.5444 |   0.4633 |  0.6567 |
| th         | spacy       |   0.4516 |   0.2827 |   0.1857 |   0.1248 |  0.4437 |
| zh         | char        |   0.5992 |   0.4361 |   0.3122 |   0.2275 |  0.6524 |
| zh         | jieba       |   0.5028 |   0.2974 |   0.1795 |   0.1085 |  0.3837 |
