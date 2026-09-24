from PIL import Image

from campusvision.imaging import clean_label, normalize_image, render_result
from campusvision.tasks import get_task


def test_normalize_applies_orientation_and_converts_to_rgb() -> None:
    image = Image.new("RGBA", (10, 20), (255, 0, 0, 128))
    image.getexif()[274] = 6

    normalized = normalize_image(image)

    assert normalized.original_size == (20, 10)
    assert normalized.processed.size == (20, 10)
    assert normalized.processed.mode == "RGB"
    assert normalized.resized is False


def test_normalize_converts_palette_and_grayscale_images() -> None:
    palette = Image.new("P", (8, 4))
    grayscale = Image.new("L", (7, 3), 128)

    assert normalize_image(palette).processed.mode == "RGB"
    assert normalize_image(grayscale).processed.mode == "RGB"


def test_normalize_limits_long_side_without_enlarging() -> None:
    large = normalize_image(Image.new("RGB", (3200, 1600)), max_side=1600)
    small = normalize_image(Image.new("RGB", (200, 100)), max_side=1600)

    assert large.processed.size == (1600, 800)
    assert large.resized is True
    assert small.processed.size == (200, 100)
    assert small.resized is False


def test_boxes_are_clamped_labeled_and_source_is_unchanged() -> None:
    source = Image.new("RGB", (100, 80), "white")
    before = source.tobytes()
    parsed = {"<OD>": {"bboxes": [[-10, 5, 120, 75]], "labels": []}}

    rendered = render_result(source, get_task("Object Detection"), parsed)

    assert source.tobytes() == before
    assert rendered.tobytes() != before
    assert rendered.getpixel((0, 5)) != (255, 255, 255)


def test_ocr_quadrilateral_is_drawn() -> None:
    source = Image.new("RGB", (100, 80), "white")
    parsed = {
        "<OCR_WITH_REGION>": {
            "quad_boxes": [[10, 10, 70, 10, 70, 30, 10, 30]],
            "labels": ["LIBRARY"],
        }
    }

    rendered = render_result(source, get_task("OCR with Regions"), parsed)

    assert rendered.getpixel((10, 10)) != (255, 255, 255)
    assert rendered.getpixel((70, 30)) != (255, 255, 255)


def test_nested_polygons_are_filled() -> None:
    source = Image.new("RGB", (100, 80), "white")
    parsed = {
        "<REFERRING_EXPRESSION_SEGMENTATION>": {
            "polygons": [[[[10, 10, 70, 10, 70, 50, 10, 50]]]],
            "labels": ["door"],
        }
    }

    rendered = render_result(
        source,
        get_task("Referring Expression Segmentation"),
        parsed,
    )

    assert rendered.getpixel((30, 30)) != (255, 255, 255)


def test_invalid_coordinates_and_empty_results_are_safe() -> None:
    source = Image.new("RGB", (20, 20), "white")
    parsed = {"<OD>": {"bboxes": [[float("nan"), 0, 10, 10]]}}

    invalid = render_result(source, get_task("Object Detection"), parsed)
    empty = render_result(source, get_task("Object Detection"), {})

    assert invalid.tobytes() == source.tobytes()
    assert empty.tobytes() == source.tobytes()
    assert invalid is not source
    assert empty is not source


def test_open_vocabulary_uses_its_specific_box_labels() -> None:
    source = Image.new("RGB", (160, 80), "white")
    parsed = {
        "<OPEN_VOCABULARY_DETECTION>": {
            "bboxes": [[0, 15, 100, 60]],
            "bboxes_labels": ["directional arrow"],
            "polygons": [],
            "polygons_labels": [],
        }
    }

    rendered = render_result(source, get_task("Open Vocabulary Detection"), parsed)

    assert rendered.getpixel((70, 16)) != (255, 255, 255)


def test_display_labels_remove_model_control_tokens() -> None:
    assert clean_label("</s><s>CAMPUS LIBRARY</s>") == "CAMPUS LIBRARY"
