from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import pytest
from PIL import Image

from campusvision.app import build_app, run_from_ui
from campusvision.contracts import TaskInputError
from campusvision.tasks import TASKS


class SuccessfulService:
    def __init__(self, root: Path) -> None:
        self.root = root

    def run(self, image, task_name, text_input, max_new_tokens, num_beams):
        image_path = self.root / "annotated.png"
        json_path = self.root / "result.json"
        image.save(image_path)
        json_path.write_text("{}", encoding="utf-8")
        return SimpleNamespace(
            annotated_image=image,
            generated_text="a campus library",
            parsed={"<CAPTION>": "a campus library"},
            artifacts=SimpleNamespace(image_path=image_path, json_path=json_path),
            record=SimpleNamespace(
                device="cpu",
                dtype="float32",
                cold_start=True,
                model_load_seconds=1.25,
                inference_seconds=0.5,
                total_seconds=2.0,
                rss_before_mb=100.0,
                rss_after_mb=120.0,
            ),
        )


class InvalidService:
    def run(self, image, task_name, text_input, max_new_tokens, num_beams):
        raise TaskInputError("Phrase Grounding requires text describing what to locate.")


def test_ui_adapter_returns_visual_structured_downloads_and_metrics(tmp_path) -> None:
    result = run_from_ui(
        Image.new("RGB", (80, 60), "white"),
        "Caption",
        "",
        128,
        3,
        service=SuccessfulService(tmp_path),
    )

    annotated, text, parsed, image_file, json_file, status = result
    assert annotated.size == (80, 60)
    assert text == "a campus library"
    assert parsed == {"<CAPTION>": "a campus library"}
    assert image_file == str(tmp_path / "annotated.png")
    assert json_file == str(tmp_path / "result.json")
    assert "CPU" in status
    assert "Inference: 0.500 s" in status
    assert "Process RSS change: +20.0 MiB" in status


def test_ui_adapter_reports_missing_image_and_input_errors(tmp_path) -> None:
    with pytest.raises(gr.Error, match="Upload, paste, or capture"):
        run_from_ui(None, "Caption", "", 128, 3, service=SuccessfulService(tmp_path))

    with pytest.raises(gr.Error, match="requires text"):
        run_from_ui(
            Image.new("RGB", (20, 20)),
            "Phrase Grounding",
            "",
            128,
            3,
            service=InvalidService(),
        )


def test_app_exposes_requested_inputs_tasks_and_outputs(tmp_path) -> None:
    demo = build_app(service=SuccessfulService(tmp_path))
    config = demo.get_config_file()
    components = config["components"]

    image_inputs = [
        item
        for item in components
        if item["type"] == "image" and item["props"].get("label") == "Campus image"
    ]
    assert len(image_inputs) == 1
    assert set(image_inputs[0]["props"]["sources"]) == {"upload", "webcam", "clipboard"}

    dropdown = next(
        item
        for item in components
        if item["type"] == "dropdown" and item["props"].get("label") == "Vision task"
    )
    choices = {
        choice[1] if isinstance(choice, (list, tuple)) else choice
        for choice in dropdown["props"]["choices"]
    }
    assert choices == set(TASKS)

    labels = {item["props"].get("label") for item in components}
    assert {
        "Phrase or object to locate",
        "Maximum generated tokens",
        "Beam count",
        "Annotated result",
        "Generated text",
        "Structured result",
        "Download annotated PNG",
        "Download result JSON",
        "Run details",
    } <= labels
