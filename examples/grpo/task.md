# Goal
- Improve Group Relative Policy Optimization (GRPO) for training a language model on math reasoning tasks.
- The base model is Qwen2.5-Math-1.5B, trained on the MATH dataset using GRPO.
- Your goal is to maximize the final validation accuracy (mean reward) of the model.
- Evaluate candidate programs via `evaluate.py` and maximize `combined_score` from `metrics.json`.
- Respect correctness: `correct.json` must report `correct: true`.
- Focus on implementing ONE specific variant of your idea with ONE fixed set of hyperparameters. Do NOT perform hyperparameter tuning, grid searches, ablation studies, multiple variants, comparative experiments, or additional baseline comparisons.
- The evaluator runs your program on the GPUs visible to the container (two: vLLM sampling on `cuda:0`, training on `cuda:1`) with the local model snapshot passed as `GRPO_MODEL_PATH`; keep the model and the `grpo_utils` interfaces unchanged.
- IMPORTANT: Each evaluation runs a full GRPO training loop (20 epochs) and takes approximately 1–2 hours. When calling `evaluate.py`, you MUST set a bash timeout of at least 7200000 ms (2 hours) to avoid the process being killed prematurely.

# Rubrics
- `accuracy_target`: final validation accuracy should improve over the baseline.
- `training_stability`: training should complete without NaN losses, OOM errors, or crashes.
- `code_safety`: program must remain executable with the provided evaluator and `grpo_utils` library.
- `evidence`: every evaluated candidate must produce `metrics.json`, `correct.json`, and rubric evidence.
