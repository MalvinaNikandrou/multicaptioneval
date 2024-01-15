import json
from pytest import fixture
from typing import Optional


@fixture(scope="session")
def results() -> dict[str, list[str]]:
    results = json.load(open("tests/fixtures/en_captions_val2014_results_after_tokenization.json"))
    return results


@fixture(scope="session")
def references() -> dict[str, list[list[str]]]:
    references = json.load(open("tests/fixtures/en_captions_val2014_references_after_tokenization.json"))
    return references


@fixture(scope="session")
def image_ids(results: dict[str, list[str]]) -> list[str]:
    return list(results.keys())


@fixture(scope="session")
def sacrebleu_results(results: dict[str, list[str]], image_ids: list[str]) -> list[str]:
    return [results[image_id][0] for image_id in image_ids]


@fixture(scope="session")
def sacrebleu_references(references: dict[str, list[list[str]]], image_ids: list[str]) -> list[list[Optional[str]]]:
    max_ref = max([len(references[image_id]) for image_id in image_ids])
    refs = [[] for _ in range(max_ref)]
    for image in image_ids:
        for ref_idx in range(max_ref):
            if ref_idx < len(references[image]):
                refs[ref_idx].append(references[image][ref_idx])
            else:
                refs[ref_idx].append(None)
    return refs
