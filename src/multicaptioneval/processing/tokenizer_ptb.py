from pycocoevalcap.tokenizer.ptbtokenizer import (
    PTBTokenizer as pycocoevalcap_PTBTokenizer,
)
from multicaptioneval.processing.tokenizer_base import (
    COCODatasetType,
    ImageCaptionsType,
)


class PTBTokenizer:
    def __init__(self) -> None:
        self._tokenizer = pycocoevalcap_PTBTokenizer()

    def __call__(self, image_captions: COCODatasetType) -> ImageCaptionsType:
        return self._tokenizer.tokenize(image_captions)
