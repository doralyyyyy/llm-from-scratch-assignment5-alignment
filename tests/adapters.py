from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


def _alpaca_sft_text(prompt: str, response: str) -> str:
    """Same structure as cs336_alignment/prompts/alpaca_sft.prompt (used by golden fixtures)."""
    return (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"{prompt}\n\n"
        "### Response:\n"
        f"{response}"
    )


def run_tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id
    assert pad_id is not None

    # Concatenate prompt/output *token ids* so the response span matches subword boundaries
    # (tokenizing the concatenated string can split tokens differently at the boundary).
    full_sequences: list[list[int]] = []
    prompt_lens: list[int] = []
    output_lens: list[int] = []
    for p, o in zip(prompt_strs, output_strs):
        p_ids = tokenizer(p, add_special_tokens=False)["input_ids"]
        o_ids = tokenizer(o, add_special_tokens=False)["input_ids"]
        full_sequences.append(p_ids + o_ids)
        prompt_lens.append(len(p_ids))
        output_lens.append(len(o_ids))

    max_full_len = max(len(s) for s in full_sequences)
    batch = len(full_sequences)
    input_ids_full = torch.full((batch, max_full_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch, max_full_len), dtype=torch.long)
    for i, seq in enumerate(full_sequences):
        L = len(seq)
        input_ids_full[i, :L] = torch.tensor(seq, dtype=torch.long)
        attention_mask[i, :L] = 1

    input_ids = input_ids_full[:, :-1]
    labels = input_ids_full[:, 1:]
    response_mask = torch.zeros_like(labels, dtype=torch.bool)
    max_label_len = labels.shape[1]

    for i, (plen, olen) in enumerate(zip(prompt_lens, output_lens)):
        seq_len = int(attention_mask[i].sum().item())
        start = max(plen - 1, 0)
        end = min(start + olen, max(seq_len - 1, 0), max_label_len)
        if end > start:
            response_mask[i, start:end] = True

    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }


def run_compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute rewards for each group of rollout responses, 
    normalized by the group size.

    For more on GRPO, see:
        DeepSeekMath: https://arxiv.org/abs/2402.03300
        DeepSeek-R1: https://arxiv.org/abs/2501.12948

    Args:
        reward_fn: Callable[[str, str], dict[str, float]], 
            scores the rollout responses against the ground truths, 
            producing a dict with keys 
            "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str], rollouts from the policy. 
            The length of this list is 
            `rollout_batch_size = n_prompts_per_rollout_batch * group_size`.
        repeated_ground_truths: list[str], the ground truths for the examples. 
            The length of this list is `rollout_batch_size`, 
            because the ground truth for each example is repeated `group_size` times.
        group_size: int, number of rollouts per group.
        advantage_eps: float, epsilon to avoid division by zero
            during group normalization.
        normalize_by_std: bool, whether to normalize the rewards by
            std(rewards).

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
            torch.Tensor of shape (rollout_batch_size,): 
                group-normalized rewards for each rollout response.
            torch.Tensor of shape (rollout_batch_size,): 
                raw rewards for each rollout response.
            dict[str, float]: metadata for the rewards of the rollout batch.
                You may choose what you wish to log here
                (some statistics of the rewards, etc.).
    """
    reward_dicts = [
        reward_fn(response, ground_truth)
        for response, ground_truth in zip(rollout_responses, repeated_ground_truths)
    ]
    raw_rewards = torch.tensor(
        [reward_dict["reward"] for reward_dict in reward_dicts], dtype=torch.float32
    )

    grouped_raw_rewards = raw_rewards.view(-1, group_size)
    group_means = grouped_raw_rewards.mean(dim=-1, keepdim=True)
    centered = grouped_raw_rewards - group_means

    if normalize_by_std:
        # Match snapshot: use Bessel-corrected std (unbiased=True), same as numpy.std(ddof=1).
        group_stds = centered.std(dim=-1, keepdim=True, unbiased=True)
        normalized = centered / (group_stds + advantage_eps)
    else:
        normalized = centered

    normalized_rewards = normalized.reshape(-1)
    metadata = {
        "reward_mean": raw_rewards.mean().item(),
        "reward_std": raw_rewards.std(unbiased=False).item(),
        "normalized_reward_mean": normalized_rewards.mean().item(),
        "normalized_reward_std": normalized_rewards.std(unbiased=False).item(),
    }
    return normalized_rewards, raw_rewards, metadata


def run_compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the logits (i.e., entropy of the final dimension)."""
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def run_get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> torch.Tensor:
    """Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
                Note that we have not masked out the token indices corresponding
                to the prompt or padding; that is done in the train loop.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions. As with the log-probs,
                we have not masked out the token indices corresponding to the prompt
                or padding; that is done in the train loop.
    """
    logits = model(input_ids).logits
    log_probs_vocab = F.log_softmax(logits, dim=-1)
    log_probs = torch.gather(log_probs_vocab, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    result: dict[str, torch.Tensor] = {"log_probs": log_probs}
    if return_token_entropy:
        result["token_entropy"] = run_compute_entropy(logits)
    return result


def run_compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute policy gradient loss using either raw rewards or advantages.

    Args:
        raw_rewards_or_advantages: torch.Tensor of shape (batch_size, 1): 
            the raw rewards or advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length): 
            the policy gradient per-token loss.
    """
    return -(raw_rewards_or_advantages * policy_log_probs)


def run_compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip loss.

    Args:
        advantages: torch.Tensor of shape (batch_size, 1): 
            the advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the old policy.
        cliprange: float, the clip range for the ratio.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            torch.Tensor of shape (batch_size, sequence_length): 
                the GRPO-Clip per-token loss.
            dict[str, torch.Tensor]: metadata for the GRPO-Clip loss 
                (used to compute clip fraction).
    """
    ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    unclipped_loss = -(advantages * ratio)
    clipped_loss = -(advantages * clipped_ratio)
    loss = torch.maximum(unclipped_loss, clipped_loss)
    metadata = {"is_clipped": (ratio != clipped_ratio)}
    return loss, metadata


def run_compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Wrapper that delegates to the appropriate policy gradient loss function above.
    """
    if loss_type == "no_baseline":
        loss = run_compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        return loss, {}
    if loss_type == "reinforce_with_baseline":
        loss = run_compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        return loss, {}
    if loss_type == "grpo_clip":
        return run_compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
        )
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def run_masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Compute the mean of the tensor along a dimension,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to compute the mean of.
        mask: torch.Tensor, the mask. We only take the mean over
            the elements with mask value 1.
        dim: int | None, the dimension to compute the mean along.
            If None, sum over all non-masked elements and average
            by their total count.

    Returns:
        torch.Tensor, the mean of the tensor along the specified
            dimension, considering only the elements with mask value 1.
    """
    masked_tensor = tensor * mask.to(dtype=tensor.dtype)
    if dim is None:
        denom = mask.to(dtype=tensor.dtype).sum()
        return masked_tensor.sum() / denom
    denom = mask.to(dtype=tensor.dtype).sum(dim=dim)
    return masked_tensor.sum(dim=dim) / denom

def run_sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.
    """
    per_token_loss = -policy_log_probs
    if normalize_constant is None:
        loss = run_masked_mean(per_token_loss, response_mask)
    else:
        per_example_loss = run_masked_normalize(
            tensor=per_token_loss,
            mask=response_mask,
            dim=-1,
            normalize_constant=normalize_constant,
        )
        loss = per_example_loss.mean()
    loss = loss / gradient_accumulation_steps
    loss.backward()
    return loss, {}

    
def run_grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        response_mask: torch.Tensor of shape (batch_size, sequence_length): 
            the mask for the response.
        gradient_accumulation_steps: int, the number of gradient accumulation steps.
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"], 
            the type of loss function to use.
        raw_rewards: torch.Tensor | None, the raw rewards for each rollout response.
            Needed for loss_type="no_baseline".
        advantages: torch.Tensor | None, the advantages for each rollout response.
            Needed for loss_type in {"reinforce_with_baseline", "grpo_clip"}.
        old_log_probs: torch.Tensor | None, the log-probs of the old policy.
            Needed for loss_type="grpo_clip".
        cliprange: float | None, the clip range for the ratio. 
            Needed for loss_type="grpo_clip".
        constant_normalize_factor: int | None, provided if we want to sum over 
            the sequence dimension and normalize by this constant factor
            (as in Dr. GRPO).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: 
            the policy gradient loss and its metadata.
    """
    per_token_loss, metadata = run_compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    loss = run_masked_mean(per_token_loss, response_mask)
    loss = loss / gradient_accumulation_steps
    loss.backward()
    return loss, metadata


def run_masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, the mask. We only consider elements
            with mask value 1.
        dim: int | None, the dimension to sum along before
            normalization. If None, sum over all dimensions.
        normalize_constant: float, the constant to divide by
            for normalization.

    Returns:
        torch.Tensor, the normalized sum, where masked elements
            (mask=0) don't contribute to the sum.
    """
    masked_tensor = tensor * mask.to(dtype=tensor.dtype)
    if dim is None:
        return masked_tensor.sum() / normalize_constant
    return masked_tensor.sum(dim=dim) / normalize_constant


"""
The below adapters are used in the optional 
RLHF / safety part of the Alignment assignment.
"""


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    """
    Given a tokenizer and a path to a dataset with instruction-tuning examples,
    construct a PyTorch Dataset for language modeling. The examples should be
    packed, i.e., all sequences in the dataset are of a constant length (`seq_length`).

    Args:
        tokenizer: transformers.PreTrainedTokenizerBase
            Transformers tokenizer to use in tokenizing and encoding text.
        dataset_path: str
            Path to file with instruction-tuning examples.
        seq_length: int
            Number of tokens to include in each example.
        shuffle: bool
            If true, shuffle the documents before packing them into examples.

    Returns:
        PyTorch Dataset for language modeling. Each example in this dataset is a dictionary of
        with keys "input_ids" and "labels" (both tensors of shape (seq_length, )).
        "input_ids" contains the token IDs for the language modeling inputs, and "labels" contains
        the token IDs for the language modeling labels.
    """
    class PackedSFTDataset(Dataset):
        def __init__(self, examples: list[dict[str, Tensor]]):
            self.examples = examples

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, idx: int) -> dict[str, Tensor]:
            return self.examples[idx]

    with open(dataset_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if shuffle:
        perm = torch.randperm(len(records)).tolist()
        records = [records[i] for i in perm]

    all_tokens: list[int] = []
    for i, record in enumerate(records):
        text = _alpaca_sft_text(record["prompt"], record["response"])
        if i == 0:
            token_ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        else:
            # Llama 3: <|end_of_text|> then <|begin_of_text|> between packed documents
            # (matches tests/fixtures/tokenized_sft_sample.json).
            token_ids = [128001, 128000] + tokenizer(text, add_special_tokens=False)["input_ids"]
        all_tokens.extend(token_ids)

    chunk_len = seq_length + 1
    examples: list[dict[str, Tensor]] = []
    for i in range(0, len(all_tokens) - chunk_len + 1, seq_length):
        chunk = all_tokens[i : i + chunk_len]
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels = torch.tensor(chunk[1:], dtype=torch.long)
        examples.append({"input_ids": input_ids, "labels": labels})

    return PackedSFTDataset(examples)


def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    """
    Given a PyTorch Dataset, return an iterable over batches of size `batch_size`.
    Iterating through the returned iterable should constitute one epoch over the Dataset.

    Args:
        dataset: Dataset
            Dataset to emit batches from.
        batch_size: int
            Number of examples to include per batch.
        shuffle: bool
            If true, shuffle examples before batching them.

    Returns:
        Iterable over batches, where each batch has size `batch_size`.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """
    Given an MMLU example and a model output, parse the model output into a
    predicted option letter (i.e., 'A', 'B', 'C', or 'D'). If the model output
    cannot be parsed into a prediction option letter, return None.

    mmlu_example: dict[str, Any]
        Dictionary with an MMLU example. Contains the following keys:
        - "subject": str with the subject of the question.
        - "question": str with the text of the question.
        - "options": list[str] with the four answer options (in order).
                     The first option refers to letter "A", the second to "B", etc.
        - "answer": str with the option of the correct answer (e.g., "A")
    model_output: str
        str with the model's output to the MMLU example.

    Returns:
        str (one of "A", "B", "C", or "D") if the model output can be parsed into a prediction,
        else None.
    """
    text = model_output.strip()
    letter_matches = re.findall(r"\b([ABCD])\b", text, flags=re.IGNORECASE)
    if letter_matches:
        return letter_matches[-1].upper()

    options = mmlu_example.get("options", [])
    normalized_text = text.lower()
    for idx, option in enumerate(options[:4]):
        opt = option.strip()
        if not opt:
            continue
        if opt.isdigit():
            if re.search(rf"(?<![0-9]){re.escape(opt)}(?![0-9])", normalized_text):
                return chr(ord("A") + idx)
        elif opt.lower() in normalized_text:
            return chr(ord("A") + idx)
    return None


def run_parse_gsm8k_response(
    model_output: str,
) -> str | None:
    """
    Given a GSM8K model output, parse the model output into a predicted numeric answer by
    taking the last number that occurs in the output.

    model_output: str
        str with the model's output to a GSM8K example.

    Returns:
        str with the predicted numeric answer if the model output can be parsed into a prediction,
        else None.
    """
    number_matches = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", model_output)
    if not number_matches:
        return None
    return number_matches[-1].replace(",", "")


def run_compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.

    lm: torch.nn.Module
        Language model being trained.
    lm_ref: torch.nn.Module
        Reference language model.
    tokenizer: PreTrainedTokenizerBase
        Tokenizer for both language models.
    beta: float
        DPO beta hyperparameter.
    prompt: str
        Prompt for this instance of preference pair.
    response_chosen: str
        Preferred response to the prompt.
    response_rejected: str
        Rejected response to the prompt.

    Returns:
        torch.Tensor with the DPO loss for this example.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenized = run_tokenize_prompt_and_output(
        prompt_strs=[prompt, prompt],
        output_strs=[response_chosen, response_rejected],
        tokenizer=tokenizer,
    )
    input_ids = tokenized["input_ids"]
    labels = tokenized["labels"]
    response_mask = tokenized["response_mask"].to(dtype=torch.float32)

    pi_log_probs = run_get_response_log_probs(
        model=lm,
        input_ids=input_ids,
        labels=labels,
        return_token_entropy=False,
    )["log_probs"]
    ref_log_probs = run_get_response_log_probs(
        model=lm_ref,
        input_ids=input_ids,
        labels=labels,
        return_token_entropy=False,
    )["log_probs"]

    pi_seq_logp = (pi_log_probs * response_mask).sum(dim=-1)
    ref_seq_logp = (ref_log_probs * response_mask).sum(dim=-1)

    chosen_lograt = pi_seq_logp[0] - ref_seq_logp[0]
    rejected_lograt = pi_seq_logp[1] - ref_seq_logp[1]
    logits = beta * (chosen_lograt - rejected_lograt)
    return -F.logsigmoid(logits)
