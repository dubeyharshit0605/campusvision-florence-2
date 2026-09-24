import psutil
import torch


def rss_megabytes() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def synchronize_device(device_type: str) -> None:
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def accelerator_memory_megabytes(device_type: str) -> float | None:
    if device_type == "cuda":
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return None
