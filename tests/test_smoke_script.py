import runpy
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
from PIL import Image

from scripts.smoke_test import main


class SmokeService:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.calls = []

    def run(self, image, task, text, max_new_tokens, num_beams):
        self.calls.append((image.size, task, text, max_new_tokens, num_beams))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        image_path = self.output_dir / "annotated.png"
        json_path = self.output_dir / "result.json"
        image.save(image_path)
        json_path.write_text("{}", encoding="utf-8")
        return SimpleNamespace(
            generated_text="CAMPUS LIBRARY",
            parsed={"<OCR>": "CAMPUS LIBRARY"},
            artifacts=SimpleNamespace(image_path=image_path, json_path=json_path),
            record=SimpleNamespace(
                device="cpu",
                dtype="float32",
                model_load_seconds=1.0,
                inference_seconds=2.0,
                total_seconds=3.5,
                rss_before_mb=100.0,
                rss_after_mb=125.0,
            ),
        )


def test_smoke_command_runs_service_and_prints_measurements(tmp_path, capsys) -> None:
    image_path = tmp_path / "notice.png"
    Image.new("RGB", (120, 80), "white").save(image_path)
    service = SmokeService(tmp_path / "outputs")

    exit_code = main(
        [
            "--image",
            str(image_path),
            "--task",
            "OCR",
            "--max-new-tokens",
            "64",
            "--num-beams",
            "1",
        ],
        service=service,
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert service.calls == [((120, 80), "OCR", "", 64, 1)]
    assert "CAMPUS LIBRARY" in output
    assert "Device: cpu (float32)" in output
    assert "Inference: 2.000 s" in output
    assert "Process RSS change: +25.0 MiB" in output
    assert str(tmp_path / "outputs" / "result.json") in output


def test_hugging_face_entrypoint_exposes_gradio_demo() -> None:
    namespace = runpy.run_path("app.py")

    assert isinstance(namespace["demo"], gr.Blocks)
