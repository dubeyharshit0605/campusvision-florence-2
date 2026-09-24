import pytest

from campusvision.contracts import TaskInputError
from campusvision.tasks import TASKS, build_prompt, get_task

EXPECTED_TASKS = {
    "Caption",
    "Detailed Caption",
    "More Detailed Caption",
    "Object Detection",
    "OCR",
    "OCR with Regions",
    "Dense Region Captioning",
    "Phrase Grounding",
    "Referring Expression Segmentation",
    "Open Vocabulary Detection",
}


def test_registry_exposes_all_supported_tasks() -> None:
    assert set(TASKS) == EXPECTED_TASKS
    assert get_task("Object Detection").token == "<OD>"
    assert (
        get_task("Referring Expression Segmentation").token
        == "<REFERRING_EXPRESSION_SEGMENTATION>"
    )


def test_phrase_prompt_is_trimmed_and_joined_to_token() -> None:
    task, prompt = build_prompt("Open Vocabulary Detection", "  bicycle  ")

    assert task.name == "Open Vocabulary Detection"
    assert prompt == "<OPEN_VOCABULARY_DETECTION>bicycle"


@pytest.mark.parametrize("text", [None, "", "   "])
def test_phrase_tasks_reject_empty_text(text: str | None) -> None:
    with pytest.raises(TaskInputError, match="requires text"):
        build_prompt("Phrase Grounding", text)


def test_fixed_prompt_ignores_stale_phrase_text() -> None:
    _, prompt = build_prompt("Caption", "stale browser value")

    assert prompt == "<CAPTION>"


def test_phrase_is_bounded() -> None:
    with pytest.raises(TaskInputError, match="500"):
        build_prompt("Phrase Grounding", "x" * 501)


def test_unknown_task_is_rejected() -> None:
    with pytest.raises(TaskInputError, match="Unsupported task"):
        get_task("Face Recognition")
