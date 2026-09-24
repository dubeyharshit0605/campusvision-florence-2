from __future__ import annotations

import csv
import json
import threading
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image

_CSV_LOCK = threading.Lock()


@dataclass(slots=True)
class RunRecord:
    run_id: str = field(default_factory=lambda: uuid4().hex)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    status: str = "success"
    error: str | None = None
    checkpoint: str = "microsoft/Florence-2-base-ft"
    revision: str = "main"
    library_versions: dict[str, str] = field(default_factory=dict)
    device: str = "cpu"
    dtype: str = "float32"
    task: str = ""
    phrase: str = ""
    original_size: tuple[int, int] = (0, 0)
    processed_size: tuple[int, int] = (0, 0)
    resized: bool = False
    generation_settings: dict[str, int] = field(default_factory=dict)
    cold_start: bool = False
    model_load_seconds: float = 0.0
    inference_seconds: float = 0.0
    total_seconds: float = 0.0
    rss_before_mb: float = 0.0
    rss_after_mb: float = 0.0
    accelerator_memory_mb: float | None = None
    output_paths: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunArtifacts:
    json_path: Path
    image_path: Path | None
    csv_path: Path


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


class ArtifactStore:
    def __init__(self, root: str | Path = "outputs") -> None:
        self.root = Path(root)
        self.csv_path = self.root / "experiments.csv"

    def save_success(
        self,
        record: RunRecord,
        *,
        generated_text: str,
        parsed: dict[str, object],
        annotated_image: Image.Image,
    ) -> RunArtifacts:
        record.status = "success"
        run_dir = self._run_directory(record.run_id)
        image_path = run_dir / "annotated.png"
        json_path = run_dir / "result.json"
        record.output_paths = {"json": str(json_path), "image": str(image_path)}

        image_temp = image_path.with_suffix(".png.tmp")
        annotated_image.save(image_temp, format="PNG")
        image_temp.replace(image_path)
        self._write_json(
            json_path,
            {
                "record": record,
                "generated_text": generated_text,
                "parsed": parsed,
            },
        )
        self._append_csv(record)
        return RunArtifacts(json_path, image_path, self.csv_path)

    def save_failure(
        self,
        record: RunRecord,
        *,
        generated_text: str = "",
        parsed: dict[str, object] | None = None,
    ) -> RunArtifacts:
        record.status = "failed"
        run_dir = self._run_directory(record.run_id)
        json_path = run_dir / "result.json"
        record.output_paths = {"json": str(json_path)}
        payload: dict[str, object] = {"record": record}
        if generated_text or parsed:
            payload.update({"generated_text": generated_text, "parsed": parsed or {}})
        self._write_json(json_path, payload)
        self._append_csv(record)
        return RunArtifacts(json_path, None, self.csv_path)

    def _run_directory(self, run_id: str) -> Path:
        if not run_id or any(char in run_id for char in "\\/:"):
            raise ValueError("run_id must be a nonempty filename-safe identifier.")
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    @staticmethod
    def _write_json(path: Path, payload: dict[str, object]) -> None:
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)

    def _append_csv(self, record: RunRecord) -> None:
        row = to_jsonable(record)
        for key in ("library_versions", "generation_settings", "output_paths"):
            row[key] = json.dumps(row[key], ensure_ascii=False, sort_keys=True)
        for key, value in row.items():
            if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
                row[key] = f"'{value}"
        fieldnames = list(row)
        self.root.mkdir(parents=True, exist_ok=True)
        with _CSV_LOCK:
            write_header = not self.csv_path.exists()
            with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
