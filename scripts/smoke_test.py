from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from campusvision.artifacts import to_jsonable
from campusvision.inference import InferenceService, get_inference_service
from campusvision.tasks import TASKS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a real Florence-2 CampusVision smoke test.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--task", choices=list(TASKS), default="Caption")
    parser.add_argument("--text", default="")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=3)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    service: InferenceService | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        with Image.open(args.image) as opened:
            image = opened.copy()
        selected_service = service or get_inference_service(args.output_dir, args.device)
        outcome = selected_service.run(
            image,
            args.task,
            args.text,
            args.max_new_tokens,
            args.num_beams,
        )
    except Exception as exc:
        print(f"CampusVision smoke test failed: {exc}", file=sys.stderr)
        return 1

    record = outcome.record
    print(f"Generated text: {outcome.generated_text}")
    print("Parsed result:")
    print(json.dumps(to_jsonable(outcome.parsed), ensure_ascii=False, indent=2))
    print(f"Device: {record.device} ({record.dtype})")
    print(f"Model load: {record.model_load_seconds:.3f} s")
    print(f"Inference: {record.inference_seconds:.3f} s")
    print(f"Total: {record.total_seconds:.3f} s")
    print(f"Process RSS change: {record.rss_after_mb - record.rss_before_mb:+.1f} MiB")
    print(f"Annotated image: {outcome.artifacts.image_path or 'none'}")
    print(f"Result JSON: {outcome.artifacts.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
