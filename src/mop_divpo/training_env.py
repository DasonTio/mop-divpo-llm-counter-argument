"""Training environment checks for Colab and local GPU runtimes."""
from __future__ import annotations

import re
from importlib import metadata

MIN_UNSUPPORTED_TORCHAO_VERSION = (0, 16, 0)


def _version_tuple(version: str) -> tuple[int, int, int]:
    parts = [int(part) for part in re.findall(r"\d+", version)[:3]]
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def check_torchao_compatibility() -> None:
    """Fail early when Colab ships an old torchao build that breaks PEFT.

    This project does not use torchao quantization. If an incompatible torchao
    package is installed, PEFT still detects it during LoRA injection and raises
    after the base model has already been downloaded. Catch it before training.
    """
    try:
        torchao_version = metadata.version("torchao")
    except metadata.PackageNotFoundError:
        return

    if _version_tuple(torchao_version) > MIN_UNSUPPORTED_TORCHAO_VERSION:
        return

    minimum = ".".join(str(part) for part in MIN_UNSUPPORTED_TORCHAO_VERSION)
    raise RuntimeError(
        "Incompatible torchao package detected. "
        f"Found torchao=={torchao_version}, but PEFT requires torchao>{minimum} "
        "when torchao is installed. This project does not need torchao; in Colab, "
        "run `!pip uninstall -y torchao` and restart the runtime before training."
    )
