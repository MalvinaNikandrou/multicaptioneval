from multicaptioneval.processing.normalization import (
    normalize_unicode,
    remove_punctuation,
)
from multicaptioneval.processing.tokenizer_ptb import PTBTokenizer
from multicaptioneval.processing.tokenizer_ko import KoreanTokenizer
from multicaptioneval.processing.tokenizer_ja import JapaneseTokenizer
from multicaptioneval.processing.tokenizer_th import ThaiTokenizer
from multicaptioneval.processing.tokenizer_zh import ChineseTokenizer
from multicaptioneval.processing.tokenizer_base import (
    BaseTokenizer,
    ImageCaptionsType,
    COCODatasetType,
    COCOSampleType,
)
from typing import Any, Optional


TOKENIZERS = {
    "ja": JapaneseTokenizer,
    "ko": KoreanTokenizer,
    "th": ThaiTokenizer,
    "zh": ChineseTokenizer,
    "ptb": PTBTokenizer,
    "none": BaseTokenizer,
}


class ProcessingPipeline:
    """Pipeline for processing image captions.

    This involves:
    1. Normalizing unicode
    2. Tokenization
    3. Removing punctuation
    """

    def __init__(
        self,
        language: str = "default",
        tokenizer_cfg: Optional[dict[str, Any]] = None,
    ) -> None:
        if tokenizer_cfg is None:
            tokenizer_cfg = {}
        self._setup_tokenizer(language, tokenizer_cfg)

    def _setup_tokenizer(self, language: str, tokenizer_cfg: dict[str, Any]) -> None:
        if language in {"zh", "ja", "ko", "th"}:
            self.tokenizer = TOKENIZERS[language](**tokenizer_cfg)
        else:
            self.tokenizer = TOKENIZERS["ptb"]()

    def normalize_captions(self, coco_captions: COCODatasetType) -> COCODatasetType:
        return {
            image_id: [self._normalize(caption) for caption in captions] for image_id, captions in coco_captions.items()
        }

    def _normalize(self, sample: COCOSampleType) -> COCOSampleType:
        sample["caption"] = normalize_unicode(sample["caption"])
        return sample

    def remove_punctuation_in_captions(self, image_captions: ImageCaptionsType) -> ImageCaptionsType:
        image_captions = {
            image_id: [remove_punctuation(caption) for caption in captions]
            for image_id, captions in image_captions.items()
        }
        return image_captions

    def __call__(self, coco_captions: COCODatasetType) -> ImageCaptionsType:
        coco_captions = self.normalize_captions(coco_captions)
        return self.remove_punctuation_in_captions(self.tokenizer(coco_captions))
