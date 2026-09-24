from dataclasses import dataclass
from typing import Literal

ResultKind = Literal["text", "boxes", "ocr", "polygons", "mixed"]


@dataclass(frozen=True, slots=True)
class TaskSpec:
    name: str
    token: str
    requires_text: bool
    result_kind: ResultKind


class TaskInputError(ValueError):
    """Raised when a task selection or its text input is invalid."""
