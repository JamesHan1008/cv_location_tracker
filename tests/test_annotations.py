import json

import pytest

from train.convert_annotations_supervisely import convert_polygons_to_masks


@pytest.fixture
def supervisely_input():
    with open("tests/fixtures/convert_annotations_supervisely_input.json") as f:
        data = json.load(f)
    return data


@pytest.fixture
def supervisely_output():
    with open("tests/fixtures/convert_annotations_supervisely_output.json") as f:
        data = json.load(f)
    return data


def test_convert_annotations_supervisely(supervisely_input, supervisely_output):
    converted_annotations = convert_polygons_to_masks(supervisely_input, "file_name")

    assert converted_annotations["class_ids"] == supervisely_output["class_ids"]
    assert converted_annotations["masks"] == supervisely_output["masks"]
