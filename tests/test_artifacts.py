import csv
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

from campusvision.artifacts import ArtifactStore, RunRecord


def make_record(run_id: str, status: str = "success") -> RunRecord:
    return RunRecord(
        run_id=run_id,
        status=status,
        task="Caption",
        original_size=(100, 80),
        processed_size=(100, 80),
    )


def test_success_saves_unique_json_png_and_csv(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    image = Image.new("RGB", (100, 80), "white")

    first = store.save_success(
        make_record("run-one"),
        generated_text="a library",
        parsed={"score": np.float32(0.5), "size": (100, 80)},
        annotated_image=image,
    )
    second = store.save_success(
        make_record("run-two"),
        generated_text="a noticeboard",
        parsed={},
        annotated_image=image,
    )

    assert first.json_path.parent != second.json_path.parent
    assert first.image_path is not None and first.image_path.exists()
    assert second.image_path is not None and second.image_path.exists()
    payload = json.loads(first.json_path.read_text(encoding="utf-8"))
    assert payload["parsed"] == {"score": 0.5, "size": [100, 80]}
    with first.csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["run_id"] for row in rows] == ["run-one", "run-two"]


def test_failure_saves_json_without_png(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    record = make_record("failed-run", status="failed")
    record.error = "model unavailable"

    result = store.save_failure(record)

    assert result.json_path.exists()
    assert result.image_path is None
    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert payload["record"]["status"] == "failed"
    assert payload["record"]["error"] == "model unavailable"


def test_concurrent_csv_appends_are_complete(tmp_path) -> None:
    store = ArtifactStore(tmp_path)

    def save(index: int) -> None:
        record = make_record(f"run-{index}", status="failed")
        record.error = "controlled failure"
        store.save_failure(record)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(save, range(20)))

    with (tmp_path / "experiments.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 20
    assert {row["run_id"] for row in rows} == {f"run-{index}" for index in range(20)}
    assert all(row["status"] == "failed" for row in rows)
