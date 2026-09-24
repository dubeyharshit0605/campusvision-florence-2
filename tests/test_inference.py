import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest
import torch
from PIL import Image

from campusvision.artifacts import ArtifactStore
from campusvision.contracts import TaskInputError
from campusvision.inference import (
    BackendResult,
    DeviceUnavailableError,
    InferenceError,
    InferenceService,
    choose_device,
)


class ControlledBackend:
    def __init__(self) -> None:
        self.load_count = 0
        self.calls: list[tuple[str, str, tuple[int, int], int, int]] = []
        self.active = 0
        self.max_active = 0
        self.guard = threading.Lock()

    def load(self, choice) -> None:
        self.load_count += 1

    def generate(self, image, task, prompt, max_new_tokens, num_beams) -> BackendResult:
        with self.guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.01)
            self.calls.append((task.token, prompt, image.size, max_new_tokens, num_beams))
            if task.result_kind == "text":
                return BackendResult("a campus library", {task.token: "a campus library"})
            return BackendResult(
                "object<loc_100><loc_100><loc_800><loc_800>",
                {task.token: {"bboxes": [[5, 5, 40, 40]], "labels": ["object"]}},
            )
        finally:
            with self.guard:
                self.active -= 1


class FailingBackend(ControlledBackend):
    def generate(self, image, task, prompt, max_new_tokens, num_beams) -> BackendResult:
        raise RuntimeError("controlled backend failure")


def test_auto_device_priority_and_cpu_dtype(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", Mock(return_value=True))
    monkeypatch.setattr(torch.backends.mps, "is_available", Mock(return_value=True))
    assert choose_device().device.type == "cuda"

    monkeypatch.setattr(torch.cuda, "is_available", Mock(return_value=False))
    assert choose_device().device.type == "mps"

    monkeypatch.setattr(torch.backends.mps, "is_available", Mock(return_value=False))
    cpu = choose_device()
    assert cpu.device.type == "cpu"
    assert cpu.dtype == torch.float32


def test_explicit_unavailable_or_unknown_device_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", Mock(return_value=False))

    with pytest.raises(DeviceUnavailableError, match="CUDA"):
        choose_device("cuda")
    with pytest.raises(DeviceUnavailableError, match="auto, cpu, cuda, or mps"):
        choose_device("tpu")


@pytest.mark.parametrize(
    ("task", "text", "tokens", "beams"),
    [
        ("Phrase Grounding", " ", 128, 3),
        ("Caption", None, 15, 3),
        ("Caption", None, 2049, 3),
        ("Caption", None, 128, 0),
        ("Caption", None, 128, 6),
    ],
)
def test_validation_happens_before_model_load(tmp_path, task, text, tokens, beams) -> None:
    backend = ControlledBackend()
    service = InferenceService(backend, ArtifactStore(tmp_path), requested_device="cpu")

    with pytest.raises(TaskInputError):
        service.run(Image.new("RGB", (64, 32)), task, text, tokens, beams)

    assert backend.load_count == 0


def test_service_loads_once_and_passes_exact_prompt_and_processed_size(tmp_path) -> None:
    backend = ControlledBackend()
    service = InferenceService(backend, ArtifactStore(tmp_path), requested_device="cpu")
    image = Image.new("RGB", (3200, 1600), "white")

    first = service.run(image, "Open Vocabulary Detection", "  bicycle  ", 128, 2)
    second = service.run(image, "Caption", "ignored", 64, 1)

    assert backend.load_count == 1
    assert backend.calls[0] == (
        "<OPEN_VOCABULARY_DETECTION>",
        "<OPEN_VOCABULARY_DETECTION>bicycle",
        (1600, 800),
        128,
        2,
    )
    assert backend.calls[1][0:3] == ("<CAPTION>", "<CAPTION>", (1600, 800))
    assert first.artifacts.json_path.exists()
    assert second.record.cold_start is False
    assert first.record.cold_start is True
    assert first.record.library_versions["python"].startswith("3.11")


def test_concurrent_runs_do_not_overlap_generation(tmp_path) -> None:
    backend = ControlledBackend()
    service = InferenceService(backend, ArtifactStore(tmp_path), requested_device="cpu")

    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(
            pool.map(
                lambda _: service.run(Image.new("RGB", (80, 60)), "Caption"),
                range(6),
            )
        )

    assert backend.load_count == 1
    assert backend.max_active == 1
    assert len({outcome.record.run_id for outcome in outcomes}) == 6


def test_backend_failure_is_saved_as_failed_and_wrapped(tmp_path) -> None:
    service = InferenceService(
        FailingBackend(),
        ArtifactStore(tmp_path),
        requested_device="cpu",
    )

    with pytest.raises(InferenceError, match="controlled backend failure"):
        service.run(Image.new("RGB", (80, 60)), "Caption")

    rows = (tmp_path / "experiments.csv").read_text(encoding="utf-8")
    assert ",failed," in rows
    result_files = list(tmp_path.glob("*/result.json"))
    assert len(result_files) == 1
    assert json.loads(result_files[0].read_text(encoding="utf-8"))["record"]["status"] == "failed"


def test_render_failure_preserves_generated_result(tmp_path, monkeypatch) -> None:
    backend = ControlledBackend()
    service = InferenceService(backend, ArtifactStore(tmp_path), requested_device="cpu")
    monkeypatch.setattr(
        "campusvision.inference.render_result",
        Mock(side_effect=ValueError("cannot render")),
    )

    with pytest.raises(InferenceError, match="cannot render"):
        service.run(Image.new("RGB", (80, 60)), "Object Detection")

    result_path = next(tmp_path.glob("*/result.json"))
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["generated_text"].startswith("object")
    assert payload["parsed"]["<OD>"]["labels"] == ["object"]
