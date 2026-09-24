from __future__ import annotations

import importlib.metadata
import platform
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from PIL import Image

from campusvision.artifacts import ArtifactStore, RunArtifacts, RunRecord
from campusvision.contracts import TaskInputError, TaskSpec
from campusvision.imaging import normalize_image, render_result
from campusvision.metrics import accelerator_memory_megabytes, rss_megabytes, synchronize_device
from campusvision.tasks import build_prompt

CHECKPOINT = "microsoft/Florence-2-base-ft"
CHECKPOINT_REVISION = "f6c1a25888ffc1d945ee8a1a77ac833c7303d46e"


class DeviceUnavailableError(RuntimeError):
    """Raised when a requested accelerator cannot be used."""


class InferenceError(RuntimeError):
    """Raised when a Florence inference run cannot be completed."""


@dataclass(frozen=True, slots=True)
class DeviceChoice:
    device: torch.device
    dtype: torch.dtype
    label: str


@dataclass(frozen=True, slots=True)
class BackendResult:
    generated_text: str
    parsed: dict[str, object]


@dataclass(frozen=True, slots=True)
class InferenceOutcome:
    annotated_image: Image.Image
    generated_text: str
    parsed: dict[str, object]
    artifacts: RunArtifacts
    record: RunRecord


class ModelBackend(Protocol):
    def load(self, choice: DeviceChoice) -> None: ...

    def generate(
        self,
        image: Image.Image,
        task: TaskSpec,
        prompt: str,
        max_new_tokens: int,
        num_beams: int,
    ) -> BackendResult: ...


def _mps_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def choose_device(requested: str = "auto") -> DeviceChoice:
    normalized = requested.strip().lower()
    if normalized not in {"auto", "cpu", "cuda", "mps"}:
        raise DeviceUnavailableError("Device must be auto, cpu, cuda, or mps.")

    if normalized == "auto":
        if torch.cuda.is_available():
            normalized = "cuda"
        elif _mps_available():
            normalized = "mps"
        else:
            normalized = "cpu"
    elif normalized == "cuda" and not torch.cuda.is_available():
        raise DeviceUnavailableError("CUDA was requested but is not available on this machine.")
    elif normalized == "mps" and not _mps_available():
        raise DeviceUnavailableError("MPS was requested but is not available on this machine.")

    dtype = torch.float32 if normalized == "cpu" else torch.float16
    return DeviceChoice(torch.device(normalized), dtype, normalized.upper())


class FlorenceBackend:
    def __init__(
        self,
        checkpoint: str = CHECKPOINT,
        revision: str = CHECKPOINT_REVISION,
    ) -> None:
        self.checkpoint = checkpoint
        self.revision = revision
        self.processor = None
        self.model = None
        self.choice: DeviceChoice | None = None

    def load(self, choice: DeviceChoice) -> None:
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            self.checkpoint,
            revision=self.revision,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint,
            revision=self.revision,
            torch_dtype=choice.dtype,
            trust_remote_code=True,
        ).to(choice.device)
        self.model.eval()
        self.choice = choice

    def generate(
        self,
        image: Image.Image,
        task: TaskSpec,
        prompt: str,
        max_new_tokens: int,
        num_beams: int,
    ) -> BackendResult:
        if self.processor is None or self.model is None or self.choice is None:
            raise RuntimeError("Florence backend has not been loaded.")

        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.choice.device)
        pixel_values = inputs["pixel_values"].to(self.choice.device, self.choice.dtype)
        with torch.inference_mode():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=num_beams,
            )
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=False,
        )[0]
        parsed = self.processor.post_process_generation(
            generated_text,
            task=task.token,
            image_size=image.size,
        )
        return BackendResult(generated_text, parsed)


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


class InferenceService:
    def __init__(
        self,
        backend: ModelBackend,
        artifact_store: ArtifactStore,
        *,
        requested_device: str = "auto",
    ) -> None:
        self.backend = backend
        self.artifact_store = artifact_store
        self.requested_device = requested_device
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._loaded = False

    def _ensure_loaded(self, choice: DeviceChoice) -> tuple[bool, float]:
        if self._loaded:
            return False, 0.0
        with self._load_lock:
            if self._loaded:
                return False, 0.0
            started = time.perf_counter()
            self.backend.load(choice)
            elapsed = time.perf_counter() - started
            self._loaded = True
            return True, elapsed

    def run(
        self,
        image: Image.Image,
        task_name: str,
        text_input: str | None = None,
        max_new_tokens: int = 512,
        num_beams: int = 3,
    ) -> InferenceOutcome:
        task, prompt = build_prompt(task_name, text_input)
        if not 16 <= int(max_new_tokens) <= 2048:
            raise TaskInputError("max_new_tokens must be between 16 and 2048.")
        if not 1 <= int(num_beams) <= 5:
            raise TaskInputError("num_beams must be between 1 and 5.")
        if not isinstance(image, Image.Image):
            raise TaskInputError("Upload, paste, or capture an image before running a task.")

        started = time.perf_counter()
        normalized = normalize_image(image)
        choice = choose_device(self.requested_device)
        record = RunRecord(
            checkpoint=CHECKPOINT,
            revision=CHECKPOINT_REVISION,
            library_versions={
                "python": platform.python_version(),
                "torch": _version("torch"),
                "transformers": _version("transformers"),
            },
            device=choice.device.type,
            dtype=str(choice.dtype).removeprefix("torch."),
            task=task.name,
            phrase=(text_input or "").strip() if task.requires_text else "",
            original_size=normalized.original_size,
            processed_size=normalized.processed.size,
            resized=normalized.resized,
            generation_settings={
                "max_new_tokens": int(max_new_tokens),
                "num_beams": int(num_beams),
            },
        )
        generated_text = ""
        parsed: dict[str, object] = {}
        record.rss_before_mb = rss_megabytes()
        try:
            cold_start, load_seconds = self._ensure_loaded(choice)
            record.cold_start = cold_start
            record.model_load_seconds = load_seconds
            with self._inference_lock:
                synchronize_device(choice.device.type)
                inference_started = time.perf_counter()
                result = self.backend.generate(
                    normalized.processed,
                    task,
                    prompt,
                    int(max_new_tokens),
                    int(num_beams),
                )
                synchronize_device(choice.device.type)
                record.inference_seconds = time.perf_counter() - inference_started
            generated_text = result.generated_text
            parsed = result.parsed
            annotated = render_result(normalized.processed, task, parsed)
            record.rss_after_mb = rss_megabytes()
            record.accelerator_memory_mb = accelerator_memory_megabytes(choice.device.type)
            record.total_seconds = time.perf_counter() - started
        except Exception as exc:
            record.status = "failed"
            record.error = f"{type(exc).__name__}: {exc}"
            record.rss_after_mb = rss_megabytes()
            record.total_seconds = time.perf_counter() - started
            self.artifact_store.save_failure(
                record,
                generated_text=generated_text,
                parsed=parsed,
            )
            message = (
                f"Inference failed on {choice.device.type}: {exc}. "
                "Check the model download, available memory, and selected device."
            )
            raise InferenceError(message) from exc

        try:
            artifacts = self.artifact_store.save_success(
                record,
                generated_text=generated_text,
                parsed=parsed,
                annotated_image=annotated,
            )
        except Exception as exc:
            message = f"Could not save inference artifacts: {exc}"
            raise InferenceError(message) from exc
        return InferenceOutcome(annotated, generated_text, parsed, artifacts, record)


_SERVICE: InferenceService | None = None
_SERVICE_LOCK = threading.Lock()


def get_inference_service(
    output_dir: str | Path = "outputs",
    requested_device: str = "auto",
) -> InferenceService:
    global _SERVICE
    if _SERVICE is None:
        with _SERVICE_LOCK:
            if _SERVICE is None:
                _SERVICE = InferenceService(
                    FlorenceBackend(),
                    ArtifactStore(output_dir),
                    requested_device=requested_device,
                )
    return _SERVICE
