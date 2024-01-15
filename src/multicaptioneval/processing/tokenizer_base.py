ImageCaptionsType = dict[str, list[str]]
COCOSampleType = dict[str, str]
COCODatasetType = dict[str, list[COCOSampleType]]


class BaseTokenizer:
    def tokenize(self, text: str) -> str:
        return text

    def __call__(self, image_captions: COCODatasetType) -> ImageCaptionsType:
        return {
            image_id: [self.tokenize(caption["caption"]) for caption in captions]
            for image_id, captions in image_captions.items()
        }
