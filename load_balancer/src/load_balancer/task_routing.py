from typing import Any
from .task_classification import TaskClass, classify_task
from . import config


def model_candidates_for_task(payload: dict[str, Any]) -> tuple[TaskClass, list[str]]:
    task_class = classify_task(payload)
    classes = getattr(config, "MODEL_CLASSES", {}) or {}

    class_cfg = classes.get(task_class.value)
    if not class_cfg:
        return task_class, []

    return task_class, list(class_cfg.get("candidates", []))
