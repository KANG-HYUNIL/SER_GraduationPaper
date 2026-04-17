import math

import torch


def _confidence_weights(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    confidence = probs.max(dim=-1).values
    return confidence / confidence.sum().clamp_min(1e-8)


def aggregate_chunk_logits(
    logits: torch.Tensor,
    mode: str = "mean_logit",
    topk_ratio: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 2:
        raise ValueError(f"Expected chunk logits with shape [num_chunks, num_classes], got {tuple(logits.shape)}.")
    if logits.size(0) == 1:
        weights = logits.new_ones(1)
        return logits[0], weights

    if mode == "mean_logit":
        weights = logits.new_full((logits.size(0),), 1.0 / logits.size(0))
        return logits.mean(dim=0), weights

    if mode == "confidence_weighted_logit":
        weights = _confidence_weights(logits)
        return torch.sum(logits * weights.unsqueeze(-1), dim=0), weights

    if mode == "topk_logit":
        k = max(1, math.ceil(logits.size(0) * float(topk_ratio)))
        confidence = torch.softmax(logits, dim=-1).max(dim=-1).values
        topk_indices = torch.topk(confidence, k=k, largest=True).indices
        selected = logits[topk_indices]
        weights = logits.new_zeros(logits.size(0))
        weights[topk_indices] = 1.0 / k
        return selected.mean(dim=0), weights

    raise ValueError(f"Unsupported utterance aggregation mode: {mode}")


def aggregate_chunk_embeddings(embeddings: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected chunk embeddings with shape [num_chunks, embed_dim], got {tuple(embeddings.shape)}."
        )
    if embeddings.size(0) == 1:
        return embeddings[0]
    if weights is None:
        return embeddings.mean(dim=0)
    weights = weights / weights.sum().clamp_min(1e-8)
    return torch.sum(embeddings * weights.unsqueeze(-1), dim=0)
