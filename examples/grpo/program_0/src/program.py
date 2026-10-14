import os
import json
import subprocess

os.environ["VLLM_USE_V1"] = "0"

import torch
import random
from typing import Literal
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from vllm import LLM, SamplingParams

from grpo_utils.sample import (
    load_prompt_template,
    sample_rollout,
    load_dataset_with_prompt_template_eval,
    load_dataset_with_prompt_template_train,
)
from grpo_utils.drgrpo_grader import grade, extract_answer
from grpo_utils.evaluate import evaluate_vllm, r1_zero_reward_fn_eval
from grpo_utils.data import create_data_loader


# =============================================================================
# UTILITIES
# =============================================================================


def r1_zero_reward_fn_train(response, ground_truth, fast=True):
    # We are strict about format to evaluate our models.
    # Note: "reward" fields are for GRPO optimization and may be modified for training purposes.
    # "is_answer_correct" is for logging only and should NOT be modified.
    if "</think> <answer>" in response and "</answer>" in response:
        model_answer = response.split("<answer>")[-1].replace("</answer>", "")
        if "\\boxed" in model_answer:
            model_answer = extract_answer(model_answer)
            if model_answer is None:
                return {
                    "format_reward": 1.0,
                    "answer_reward": 0.0,
                    "reward": 0.0,
                    "is_answer_correct": 0,
                }
        if isinstance(ground_truth, float) or isinstance(ground_truth, int):
            ground_truth = str(ground_truth)
        if isinstance(ground_truth, str):
            is_correct = grade(model_answer, ground_truth, fast)
        elif isinstance(ground_truth, list):
            is_correct = False
            for gt in ground_truth:
                is_correct |= grade(model_answer, gt, fast)
        if is_correct:
            return {
                "format_reward": 1.0,
                "answer_reward": 1.0,
                "reward": 1.0,
                "is_answer_correct": 1,
            }
        else:
            # Formatted but wrong answer; no format reward to avoid hacking.
            return {
                "format_reward": 1.0,
                "answer_reward": 0.0,
                "reward": 0.0,
                "is_answer_correct": 0,
            }
    else:
        # Unformatted.
        return {
            "format_reward": 0.0,
            "answer_reward": 0.0,
            "reward": 0.0,
            "is_answer_correct": 0,
        }


def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    # Tokenize prompts and outputs separately
    prompt_tokens = [
        tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompt_strs
    ]
    output_tokens = [
        tokenizer.encode(output, add_special_tokens=False) for output in output_strs
    ]

    # Concatenate prompt and output tokens
    prompt_and_output_tokens = [p + o for p, o in zip(prompt_tokens, output_tokens)]
    prompt_and_output_lens = [len(tokens) for tokens in prompt_and_output_tokens]
    max_len = max(prompt_and_output_lens)

    # Pad sequences to max length
    padded_tokens = [
        tokens + [tokenizer.pad_token_id] * (max_len - len(tokens))
        for tokens in prompt_and_output_tokens
    ]

    # Convert to tensors
    input_ids = torch.tensor(padded_tokens)
    labels = input_ids[:, 1:]
    input_ids = input_ids[:, :-1]

    # Create response mask
    response_mask = torch.zeros_like(input_ids)
    for i, (p_len, o_len) in enumerate(
        zip([len(p) for p in prompt_tokens], [len(o) for o in output_tokens])
    ):
        # Set mask to 1 for output tokens (after prompt)
        response_mask[i, (p_len - 1) : (p_len + o_len - 1)] = 1

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}


def compute_entropy(logits):
    # Compute the entropy of the logits using log_softmax for numerical stability
    log_probs = torch.log_softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)
    probs = torch.exp(log_probs)
    entropy = -torch.sum(probs * log_probs, dim=-1)  # (batch_size, seq_len)
    return entropy


def get_response_log_probs(
    model, input_ids, labels, return_token_entropy=False, no_grad=True
):
    if no_grad:
        with torch.no_grad():
            outputs = model(input_ids, labels=labels)
            logits = outputs.logits  # (batch_size, seq_len, vocab_size)
            log_probs = torch.log_softmax(
                logits, dim=-1
            )  # (batch_size, seq_len, vocab_size)
            # Get log probs of the actual label tokens
            batch_size, seq_len = labels.shape  # (batch_size, seq_len)
            log_probs = torch.gather(
                log_probs, dim=-1, index=labels.unsqueeze(-1)
            ).squeeze(-1)
            if return_token_entropy:
                entropy = compute_entropy(logits)
            else:
                entropy = None
    else:
        outputs = model(input_ids, labels=labels)
        logits = outputs.logits  # (batch_size, seq_len, vocab_size)
        log_probs = torch.log_softmax(
            logits, dim=-1
        )  # (batch_size, seq_len, vocab_size)
        # Get log probs of the actual label tokens
        batch_size, seq_len = labels.shape  # (batch_size, seq_len)
        log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(
            -1
        )
        if return_token_entropy:
            entropy = compute_entropy(logits)
        else:
            entropy = None

    return {"log_probs": log_probs, "token_entropy": entropy}


def masked_normalize(tensor, mask, normalize_constant, dim):
    # Apply mask to tensor (set masked elements to 0)
    masked_tensor = tensor * mask

    # Sum along specified dimension
    sum_tensor = torch.sum(masked_tensor, dim=dim)

    # Normalize by constant
    normalized = sum_tensor / normalize_constant

    return normalized


def sft_microbatch_train_step(
    policy_log_probs, response_mask, gradient_accumulation_steps, normalize_constant=1.0
):
    # Compute negative log likelihood loss
    # Note: policy_log_probs are already log probabilities, so we just need to negate them
    nll = -policy_log_probs

    # Normalize the loss using the response mask
    # This ensures we only consider loss for response tokens and properly average
    loss = masked_normalize(
        nll,
        response_mask,
        normalize_constant=normalize_constant,
        dim=-1,  # Sum over sequence length
    )

    # Scale loss by gradient accumulation steps
    loss = loss / gradient_accumulation_steps

    # Take mean across batch to get scalar loss
    loss = loss.mean()

    # Now backward() will work since loss is a scalar
    loss.backward()

    metadata = {
        "per_sample_loss": loss.item()  # Store the scalar loss value
    }

    return loss, metadata


# =============================================================================
# GRPO UTILITIES
# =============================================================================


def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    group_size,
    advantage_eps,
    normalize_by_std,
):
    """
    reward_fn: Callable[[str, str], dict[str, float]] Scores the rollout responses against the ground truths, producing a dict with keys "reward", "format_reward", and "answer_reward".
    rollout_responses: list[str] Rollouts from the policy. The length of this list is rollout_batch_size = n_prompts_per_rollout_batch * group_size.
    repeated_ground_truths: list[str] The ground truths for the examples. The length of this list is rollout_batch_size, because the ground truth for each example is repeated group_size times.
    group_size: int Number of responses per question (group).
    advantage_eps: float Small constant to avoid division by zero in normalization.
    normalize_by_std: bool If True, divide by the per-group standard deviation; otherwise subtract only the group mean.

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]].
            - advantages shape (rollout_batch_size,). Group-normalized rewards for each rollout response.
            - raw_rewards shape (rollout_batch_size,). Unnormalized rewards for each rollout response.
            - metadata your choice of other statistics to log (e.g. mean, std, max/min of rewards).
    """
    # Compute raw rewards for each response
    raw_rewards = []
    is_answer_correct = []
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        reward_dict = reward_fn(response, ground_truth)
        raw_rewards.append(reward_dict["reward"])
        is_answer_correct.append(reward_dict["is_answer_correct"])
    raw_rewards = torch.tensor(raw_rewards)
    is_answer_correct = torch.tensor(is_answer_correct, dtype=torch.float)
    # Reshape rewards into groups
    n_groups = len(raw_rewards) // group_size
    grouped_rewards = raw_rewards.view(n_groups, group_size)

    # Compute group statistics
    group_means = grouped_rewards.mean(dim=1, keepdim=True)
    if normalize_by_std:
        group_stds = grouped_rewards.std(dim=1, keepdim=True) + advantage_eps
        advantages = (grouped_rewards - group_means) / group_stds
    else:
        advantages = grouped_rewards - group_means

    # Flatten advantages back to original shape
    advantages = advantages.view(-1)

    # Compute metadata statistics
    metadata = {
        "mean_reward": raw_rewards.mean().item(),
        "accuracy": is_answer_correct.mean().item(),
        "std_reward": raw_rewards.std().item(),
        "max_reward": raw_rewards.max().item(),
        "min_reward": raw_rewards.min().item(),
        "mean_advantage": advantages.mean().item(),
        "std_advantage": advantages.std().item(),
    }

    return advantages, raw_rewards, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    raw_rewards_or_advantages: torch.Tensor, shape (batch_size, 1).
    policy_log_probs: torch.Tensor, shape (batch_size, sequence_length).

    Returns:
    torch.Tensor Shape (batch_size, sequence_length), the per-token policy-gradient loss (to be aggregated across the batch and sequence dimensions in the training loop).
    """
    return -raw_rewards_or_advantages * policy_log_probs


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    advantages: torch.Tensor, shape (batch_size, 1).
    policy_log_probs: torch.Tensor, shape (batch_size, sequence_length).
    old_log_probs: torch.Tensor, shape (batch_size, sequence_length).
    cliprange: float, the clip range for the ratio.

    Returns:
    tuple[torch.Tensor, dict[str, torch.Tensor]].
    loss torch.Tensor of shape (batch_size, sequence_length), the per-token clipped loss.
    metadata dict containing whether each token was clipped or not.
    """
    # Calculate probability ratio r = π_θ(a|s) / π_θ_old(a|s)
    ratio = torch.exp(
        policy_log_probs - old_log_probs
    )  # shape: (batch_size, sequence_length)

    # Calculate surrogate objectives
    surr1 = ratio * advantages  # Unclipped surrogate
    surr2 = (
        torch.clamp(ratio, 1 - cliprange, 1 + cliprange) * advantages
    )  # Clipped surrogate

    # Take the minimum of the surrogates (negative since we want to maximize reward)
    loss = -torch.min(surr1, surr2)

    # Track which tokens were clipped (where surr2 < surr1)
    was_clipped = surr2 < surr1

    metadata = {
        "clipped_tokens": was_clipped,
        "clip_fraction": was_clipped.float().mean(),
    }

    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Select and compute the desired policy-gradient loss.
    policy_log_probs (batch_size, sequence_length)
    raw_rewards Required if loss_type == "no_baseline"; shape (batch_size, 1).
    advantages Required for "reinforce_with_baseline" and "grpo_clip"; shape (batch_size, 1).
    old_log_probs Required for "grpo_clip"; shape (batch_size, sequence_length).
    cliprange Required for "grpo_clip"; float.

    Returns:
    tuple[torch.Tensor, dict[str, torch.Tensor]].
    loss (batch_size, sequence_length), per-token loss.
    metadata dict, statistics from the underlying routine (e.g., clip fraction for GRPO-Clip).
    """
    # Assert input shapes
    assert (
        policy_log_probs.dim() == 2
    ), f"Expected policy_log_probs to have 2 dimensions, got {policy_log_probs.dim()}"

    if loss_type == "no_baseline":
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        return loss, {"loss_type": "no_baseline"}

    elif loss_type == "reinforce_with_baseline":
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        return loss, {"loss_type": "reinforce_with_baseline"}

    elif loss_type == "grpo_clip":
        assert cliprange is not None
        return compute_grpo_clip_loss(
            advantages, policy_log_probs, old_log_probs, cliprange
        )

    else:
        raise ValueError(f"Invalid loss type: {loss_type}")


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """
    Compute the mean of tensor along a given dimension, considering only those elements where mask == 1.
    tensor: torch.Tensor The data to be averaged.
    mask: torch.Tensor Same shape as tensor; positions with 1 are included in the mean.
    dim: int | None Dimension over which to average. If None, compute the mean over all masked elements.

    Returns:
    torch.Tensor The masked mean; shape matches tensor.mean(dim) semantics.
    """
    return torch.sum(tensor * mask, dim=dim) / torch.sum(mask, dim=dim)


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Return:
    tuple[torch.Tensor, dict[str, torch.Tensor]].
    loss scalar tensor. The microbatch loss, adjusted for gradient accumulation. We return this so we can log it.
    metadata Dict with metadata from the underlying loss call, and any other statistics you might want to log.

    You should call loss.backward() in this function. Make sure to adjust for gradient accumulation.
    """
    loss, metadata = compute_policy_gradient_loss(
        policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange
    )  # (batch_size, sequence_length)
    loss = masked_mean(loss, response_mask)
    loss = loss / gradient_accumulation_steps
    loss.backward()
    return loss, metadata


# =============================================================================
# MAIN TRAINING CODE
# =============================================================================

def load_policy_into_vllm_instance(policy, llm):
    # Move state dict to CPU to avoid doubling GPU memory on the training device
    state_dict = {k: v.cpu() for k, v in policy.state_dict().items()}
    llm.apply_model(lambda model, sd=state_dict: model.load_weights(sd.items()))
    del state_dict
    torch.cuda.empty_cache()


def load_model_and_tokenizer(
    model_path="Qwen/Qwen2.5-Math-1.5B", tokenizer_path="Qwen/Qwen2.5-Math-1.5B"
):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer


def evaluate_model(
    policy_model,
    vllm_model,
    eval_prompts,
    eval_answers,
    eval_sampling_params,
    output_path=None,
):
    load_policy_into_vllm_instance(policy_model, vllm_model)
    metrics = evaluate_vllm(
        vllm_model,
        r1_zero_reward_fn_eval,
        eval_prompts,
        eval_answers,
        eval_sampling_params,
        output_path=output_path,
    )
    return metrics


def train_loop(
    model,
    tokenizer,
    vllm_model,
    train_prompts,
    train_answers,
    learning_rate,
    grpo_steps,
    train_steps_per_rollout,
    batch_size,
    gradient_accumulation_steps=4,
    group_size=2,
    rollout_subset_size=256,
    device="cuda",
    eval_epochs=5,
    eval_prompts=None,
    eval_answers=None,
    sampling_params=None,
    cliprange=0.2,
    loss_type="reinforce_with_baseline",
) -> list[dict[str, float | int]]:
    """
    Train the model using GRPO.
    """
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.0, betas=(0.9, 0.95)
    )
    global_step = 0  # Initialize global step counter
    logs: list[dict[str, float | int]] = []

    for epoch in range(grpo_steps):
        model.train()
        epoch_log: dict[str, float | int] = {"epoch": epoch}

        # Evaluate on validation set every eval_epochs epochs
        if (
            epoch % eval_epochs == 0
            and eval_prompts is not None
            and eval_answers is not None
        ):
            model.eval()
            print("\nEvaluating on validation set at epoch: ", epoch)
            eval_results = evaluate_model(
                model, vllm_model, eval_prompts, eval_answers, sampling_params
            )
            eval_mean_reward = sum(
                result["reward"]["reward"] for result in eval_results
            ) / len(eval_results)
            epoch_log["eval/mean_reward"] = eval_mean_reward
            epoch_log["eval/accuracy"] = sum(
                result["reward"]["is_answer_correct"] for result in eval_results
            ) / len(eval_results)
            model.train()

        ## load the current policy model to vllm for sampling rollouts
        load_policy_into_vllm_instance(model, vllm_model)

        ## sample rollouts
        print("Sampling rollouts for epoch: ", epoch)
        rollout_prompts, rollout_answers, rollout_responses, rollout_rewards = (
            sample_rollout(
                vllm_model,
                r1_zero_reward_fn_train,
                train_prompts,
                train_answers,
                G=group_size,
                eval_sampling_params=sampling_params,
                subset_size=rollout_subset_size,
                return_rewards=True,
                batch_size=512,
            )
        )
        # Randomly sample 2 rollouts to print
        indices = random.sample(range(len(rollout_prompts)), 2)
        print("Example rollouts:")
        for idx in indices:
            print(f"\nRollout {idx}:")
            print(f"Prompt: {rollout_prompts[idx]}")
            print(f"Response: {rollout_responses[idx]}")
            print(f"Reward: {rollout_rewards[idx]}")
            print(f"Ground truth: {rollout_answers[idx]}")
        rollout_tokenized = tokenize_prompt_and_output(
            rollout_prompts, rollout_responses, tokenizer
        )
        rollout_data_loader = create_data_loader(
            rollout_tokenized, batch_size=batch_size, shuffle=False
        )

        # Get old policy log probs batch by batch to avoid OOM
        old_log_probs_list = []
        with torch.no_grad():
            for batch in rollout_data_loader:
                input_ids, labels, response_mask = [t.to(device) for t in batch]
                old_response_log_probs = get_response_log_probs(
                    model, input_ids, labels, return_token_entropy=False, no_grad=True
                )
                old_log_probs_list.append(old_response_log_probs["log_probs"])

                # Clean up memory after each batch
                del old_response_log_probs
                torch.cuda.empty_cache()

            # Concatenate all batches
            old_log_probs = torch.cat(old_log_probs_list, dim=0)
            del old_log_probs_list
            torch.cuda.empty_cache()

        # Compute advantages using group normalization - no gradients needed
        with torch.no_grad():
            advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
                reward_fn=r1_zero_reward_fn_train,
                rollout_responses=rollout_responses,
                repeated_ground_truths=rollout_answers,
                group_size=group_size,
                advantage_eps=1e-6,
                normalize_by_std=True,
            )
            advantages = advantages.to(device)

        # Log raw rewards statistics
        print("\nGRPO epoch: ", epoch)
        print(f"Mean reward: {reward_metadata['mean_reward']:.4f}")

        # Add reward statistics to epoch log
        epoch_log["train/mean_reward"] = reward_metadata["mean_reward"]
        epoch_log["train/accuracy"] = reward_metadata["accuracy"]
        epoch_log["train/std_reward"] = reward_metadata["std_reward"]
        epoch_log["train/min_reward"] = reward_metadata["min_reward"]
        epoch_log["train/max_reward"] = reward_metadata["max_reward"]
        epoch_log["train/mean_advantage"] = reward_metadata["mean_advantage"]
        epoch_log["train/std_advantage"] = reward_metadata["std_advantage"]

        ## train on this rollout batch for train_steps_per_rollout steps
        epoch_losses = []
        epoch_entropies = []
        epoch_clip_fractions = []

        for train_step in range(train_steps_per_rollout):
            # Process each batch
            for batch_idx, batch in tqdm(enumerate(rollout_data_loader)):
                global_step += 1
                input_ids, labels, response_mask = [t.to(device) for t in batch]

                # Get current policy log probs (with gradients)
                response_log_probs = get_response_log_probs(
                    model, input_ids, labels, return_token_entropy=True, no_grad=False
                )
                policy_log_probs = response_log_probs["log_probs"]
                token_entropy = response_log_probs["token_entropy"]

                # Compute masked mean entropy for this batch
                batch_entropy = (
                    token_entropy * response_mask
                ).sum() / response_mask.sum()
                epoch_entropies.append(batch_entropy.item())

                # Calculate data index for advantages/old_log_probs
                batch_idx_total = batch_idx * batch_size
                batch_advantages = advantages[
                    batch_idx_total : batch_idx_total + batch_size
                ].unsqueeze(-1)  # Add dimension to get (batch_size, 1)
                batch_old_log_probs = old_log_probs[
                    batch_idx_total : batch_idx_total + batch_size
                ]

                loss, loss_metadata = grpo_microbatch_train_step(
                    policy_log_probs=policy_log_probs,
                    response_mask=response_mask,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    loss_type=loss_type,
                    advantages=batch_advantages,
                    old_log_probs=batch_old_log_probs,
                    cliprange=cliprange,
                )

                # Track clip fraction if available (for grpo_clip loss type)
                if "clip_fraction" in loss_metadata:
                    epoch_clip_fractions.append(loss_metadata["clip_fraction"].item())

                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    epoch_losses.append(loss.item())

        # Record average metrics for this epoch
        if epoch_losses:
            epoch_log["train/loss"] = sum(epoch_losses) / len(epoch_losses)
        if epoch_entropies:
            epoch_log["train/entropy"] = sum(epoch_entropies) / len(epoch_entropies)
        if epoch_clip_fractions:
            epoch_log["train/clip_fraction"] = sum(epoch_clip_fractions) / len(
                epoch_clip_fractions
            )

        logs.append(epoch_log)

        # Clean up epoch-level tensors to prevent memory accumulation across epochs
        del old_log_probs, advantages, rollout_tokenized, rollout_data_loader
        del rollout_prompts, rollout_answers, rollout_responses, rollout_rewards
        torch.cuda.empty_cache()

    return logs


def run_experiment(
    seed: int = 42,
) -> tuple[float, list[dict[str, float | int]]]:
    """
    Run the GRPO training experiment.

    Args:
        seed: Random seed for reproducibility.

    Returns:
        tuple[float, list[dict[str, float]]]: A tuple of (final_val_accuracy, logs).
            - final_val_accuracy: The last recorded eval/mean_reward, or 0.0 if no eval was run.
            - logs: Training logs from each epoch.
    """

    # Display GPU information
    print("=== GPU Information ===")
    subprocess.run(["nvidia-smi"])
    print("=" * 50 + "\n")

    # Set random seeds for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Hyperparameters that should not be changed whatsoever
    # The evaluator points GRPO_MODEL_PATH at the local Qwen2.5-Math-1.5B snapshot.
    model_path = os.environ.get("GRPO_MODEL_PATH", "Qwen/Qwen2.5-Math-1.5B")
    tokenizer_path = model_path

    grpo_steps = 20
    rollout_subset_size = 128
    eval_epochs = 2

    group_size = 8
    train_steps_per_rollout = 1
    gradient_accumulation_steps = 16
    batch_size = 4

    # Hyperparameters that can be changed
    learning_rate = 1e-5
    cliprange = 0.2
    loss_type = "grpo_clip"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script")
    else:
        print("CUDA is available")

    prompt_template = load_prompt_template()
    n_gpus = torch.cuda.device_count()
    print(f"Found {n_gpus} GPUs")
    if n_gpus >= 2:
        vllm_device = "cuda:0"
        train_device = "cuda:1"
        vllm_gpu_util = 0.85
    else:
        vllm_device = "cuda:0"
        train_device = "cuda:0"
        vllm_gpu_util = 0.40

    print(f"Loading vLLM model on {vllm_device}...")
    vllm_model = LLM(
        model=model_path,
        tokenizer=tokenizer_path,
        gpu_memory_utilization=vllm_gpu_util,
    )

    print("Loading evaluation dataset...")
    eval_prompts, eval_answers = load_dataset_with_prompt_template_eval(prompt_template)
    eval_sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    # Initialize training model on separate GPU if available
    print(f"Loading training model on {train_device}...")
    model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path)
    train_prompts, train_answers = load_dataset_with_prompt_template_train(
        prompt_template
    )
    logs = train_loop(
        model,
        tokenizer,
        vllm_model,
        train_prompts,
        train_answers,
        learning_rate,
        grpo_steps,
        train_steps_per_rollout,
        batch_size,
        gradient_accumulation_steps,
        group_size,
        rollout_subset_size,
        eval_epochs=eval_epochs,
        eval_prompts=eval_prompts,
        eval_answers=eval_answers,
        sampling_params=eval_sampling_params,
        cliprange=cliprange,
        loss_type=loss_type,
        device=train_device,
    )

    # Final validation evaluation
    print("\n=== Final Validation Evaluation ===")
    model.eval()
    final_eval_output_path = os.environ.get("GRPO_FINAL_EVAL_OUTPUT_PATH")
    final_eval_results = evaluate_model(
        model,
        vllm_model,
        eval_prompts,
        eval_answers,
        eval_sampling_params,
        output_path=final_eval_output_path,
    )
    final_eval_accuracy = sum(
        result["reward"]["is_answer_correct"] for result in final_eval_results
    ) / len(final_eval_results)
    final_eval_mean_reward = sum(
        result["reward"]["reward"] for result in final_eval_results
    ) / len(final_eval_results)
    print(f"Final validation accuracy: {final_eval_accuracy:.4f}")
    print(f"Final validation mean reward: {final_eval_mean_reward:.4f}")

    # Cleanup vLLM engine (prevents subprocess hang on exit)
    del vllm_model

    # Cleanup distributed resources
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    # Clean up CUDA memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final_eval_accuracy, logs


if __name__ == "__main__":
    val_accuracy, logs = run_experiment(seed=42)
    # Write structured results for evaluate.py
    results = {"final_accuracy": float(val_accuracy), "logs": logs}
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print("\n=== Training Logs ===")
    for log in logs:
        print(log)
    print(f"\nFinal validation accuracy: {val_accuracy}")
