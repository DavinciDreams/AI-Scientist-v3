# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub>=0.30"]
# ///
"""Download the Qwen2.5-Math-1.5B weights the GRPO evaluator loads offline.

Run once before evaluating: ``uv run examples/grpo/download_model.py``.
The snapshot (about 3 GB) lands in ``examples/grpo/resources/Qwen2.5-Math-1.5B``.
"""

import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
MODEL_DIR = Path(__file__).resolve().parent / "resources" / MODEL_NAME.split("/")[-1]


def main() -> None:
    snapshot_download(
        MODEL_NAME,
        local_dir=MODEL_DIR,
        allow_patterns=["*.json", "*.safetensors", "*.txt"],
    )
    # Drop the hub's per-file download metadata; evaluations only read the snapshot.
    shutil.rmtree(MODEL_DIR / ".cache", ignore_errors=True)
    print(f"Model snapshot ready: {MODEL_DIR}")


if __name__ == "__main__":
    main()
