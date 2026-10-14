"""Entry-point reporting and compatibility for portable temporary storage."""
from __future__ import annotations

import shutil
import sys

from .temp_storage import initialize_temp_storage, preview_temp_storage


def initialize_runtime_storage() -> None:
    storage = initialize_temp_storage()
    free = shutil.disk_usage(storage.base).free
    print(
        f"Temporary storage: {storage.temporary} "
        f"(base={storage.base}, free_bytes={free})",
        file=sys.stderr,
    )


def preview_runtime_storage() -> None:
    print(f"Temporary storage preview: {preview_temp_storage()}", file=sys.stderr)
