from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

from PIL import Image, ImageDraw, ImageOps

from campusvision.contracts import TaskSpec

COLORS = (
    (0, 102, 204),
    (220, 38, 38),
    (5, 150, 105),
    (147, 51, 234),
    (234, 88, 12),
    (8, 145, 178),
)


@dataclass(frozen=True, slots=True)
class NormalizedImage:
    original_size: tuple[int, int]
    processed: Image.Image
    resized: bool


def normalize_image(image: Image.Image, max_side: int = 1600) -> NormalizedImage:
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a Pillow image.")
    if max_side < 1:
        raise ValueError("max_side must be positive.")

    oriented = ImageOps.exif_transpose(image)
    original_size = oriented.size
    processed = oriented.convert("RGB")
    resized = max(processed.size) > max_side
    if resized:
        scale = max_side / max(processed.size)
        target = (
            max(1, round(processed.width * scale)),
            max(1, round(processed.height * scale)),
        )
        processed = processed.resize(target, Image.Resampling.LANCZOS)
    return NormalizedImage(original_size, processed, resized)


def _result_body(task: TaskSpec, parsed: Mapping[str, Any]) -> Mapping[str, Any]:
    candidate = parsed.get(task.token, parsed)
    return candidate if isinstance(candidate, Mapping) else {}


def _finite_numbers(values: object, count: int | None = None) -> list[float] | None:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return None
    if count is not None and len(values) != count:
        return None
    numbers: list[float] = []
    for value in values:
        if not isinstance(value, Real) or not math.isfinite(float(value)):
            return None
        numbers.append(float(value))
    return numbers


def _clamp(value: float, maximum: int) -> int:
    return round(min(max(value, 0.0), float(maximum - 1)))


def _label(labels: object, index: int) -> str:
    if (
        isinstance(labels, Sequence)
        and not isinstance(labels, (str, bytes))
        and index < len(labels)
        and str(labels[index]).strip()
    ):
        return str(labels[index]).strip()
    return f"region {index + 1}"


def _draw_label(draw: ImageDraw.ImageDraw, point: tuple[int, int], text: str, color: tuple) -> None:
    x, y = point
    left, top, right, bottom = draw.textbbox((x, y), text)
    draw.rectangle((left - 2, top - 2, right + 2, bottom + 2), fill=color)
    draw.text((x, y), text, fill=(255, 255, 255, 255))


def _polygon_sequences(value: object) -> Iterable[list[float]]:
    numbers = _finite_numbers(value)
    if numbers is not None:
        if len(numbers) >= 6 and len(numbers) % 2 == 0:
            yield numbers
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            yield from _polygon_sequences(child)


def render_result(
    image: Image.Image,
    task: TaskSpec,
    parsed: dict[str, object],
) -> Image.Image:
    base = image.convert("RGB").copy()
    body = _result_body(task, parsed)
    if not body:
        return base

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    labels = body.get("labels", [])

    boxes = body.get("bboxes", [])
    if isinstance(boxes, Sequence):
        for index, box in enumerate(boxes):
            coords = _finite_numbers(box, 4)
            if coords is None:
                continue
            x0, y0, x1, y1 = coords
            points = (
                _clamp(min(x0, x1), base.width),
                _clamp(min(y0, y1), base.height),
                _clamp(max(x0, x1), base.width),
                _clamp(max(y0, y1), base.height),
            )
            color = (*COLORS[index % len(COLORS)], 255)
            draw.rectangle(points, outline=color, width=3)
            _draw_label(draw, (points[0], points[1]), _label(labels, index), color)

    quadrilaterals = body.get("quad_boxes", [])
    if isinstance(quadrilaterals, Sequence):
        for index, quad in enumerate(quadrilaterals):
            coords = _finite_numbers(quad, 8)
            if coords is None:
                continue
            points = [
                (_clamp(coords[offset], base.width), _clamp(coords[offset + 1], base.height))
                for offset in range(0, 8, 2)
            ]
            color = (*COLORS[index % len(COLORS)], 255)
            draw.line([*points, points[0]], fill=color, width=3, joint="curve")
            _draw_label(draw, points[0], _label(labels, index), color)

    polygons = list(_polygon_sequences(body.get("polygons", [])))
    for index, polygon in enumerate(polygons):
        points = [
            (_clamp(polygon[offset], base.width), _clamp(polygon[offset + 1], base.height))
            for offset in range(0, len(polygon), 2)
        ]
        rgb = COLORS[index % len(COLORS)]
        draw.polygon(points, fill=(*rgb, 72), outline=(*rgb, 255))
        _draw_label(draw, points[0], _label(labels, index), (*rgb, 255))

    return Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
