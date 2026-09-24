from campusvision.contracts import TaskInputError, TaskSpec

_TASK_LIST = (
    TaskSpec("Caption", "<CAPTION>", False, "text"),
    TaskSpec("Detailed Caption", "<DETAILED_CAPTION>", False, "text"),
    TaskSpec("More Detailed Caption", "<MORE_DETAILED_CAPTION>", False, "text"),
    TaskSpec("Object Detection", "<OD>", False, "boxes"),
    TaskSpec("OCR", "<OCR>", False, "text"),
    TaskSpec("OCR with Regions", "<OCR_WITH_REGION>", False, "ocr"),
    TaskSpec("Dense Region Captioning", "<DENSE_REGION_CAPTION>", False, "boxes"),
    TaskSpec("Phrase Grounding", "<CAPTION_TO_PHRASE_GROUNDING>", True, "boxes"),
    TaskSpec(
        "Referring Expression Segmentation",
        "<REFERRING_EXPRESSION_SEGMENTATION>",
        True,
        "polygons",
    ),
    TaskSpec("Open Vocabulary Detection", "<OPEN_VOCABULARY_DETECTION>", True, "mixed"),
)

TASKS: dict[str, TaskSpec] = {task.name: task for task in _TASK_LIST}


def get_task(task_name: str) -> TaskSpec:
    try:
        return TASKS[task_name]
    except KeyError as exc:
        supported = ", ".join(TASKS)
        message = f"Unsupported task: {task_name!r}. Choose one of: {supported}."
        raise TaskInputError(message) from exc


def build_prompt(task_name: str, text_input: str | None = None) -> tuple[TaskSpec, str]:
    task = get_task(task_name)
    if not task.requires_text:
        return task, task.token

    text = (text_input or "").strip()
    if not text:
        raise TaskInputError(f"{task.name} requires text describing what to locate.")
    if len(text) > 500:
        raise TaskInputError("Text input must be 500 characters or fewer.")
    return task, f"{task.token}{text}"
