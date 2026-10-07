# SPDX-License-Identifier: Apache-2.0
"""Top-k-first top-k/top-p sampling for HPU.

Without Triton, the upstream PyTorch path (``apply_top_k_top_p_pytorch``) sorts
the whole vocabulary of every row whenever top-p is set, even when every request
also sets a small top-k. On HPU that sort dominates sampling for large
vocabularies and batches.

When the host knows that every row's top-k is at most ``max_top_k`` (a small
static bucket), only the ``max_top_k`` largest logits of each row can survive.
This path finds them without sorting the vocabulary: it splits each row into
fixed-size groups, keeps the ``max_top_k`` groups with the largest maxima and
takes the top-k of those groups only. It then applies the same top-k and top-p
masks as upstream to the candidates in ascending order and samples among the
candidates. The kept set and its probabilities are the same as upstream, except
that tokens exactly at the top-p boundary may differ by float rounding of the
cumulative sum.

The bound comes from the host-side ``top_k`` values of the scheduled requests
(``InputBatch.top_k_cpu``), so no device-to-host sync is needed and every bucket
is a static shape.
"""
from __future__ import annotations

from typing import Optional

import torch
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler, random_sample

# Static max_top_k buckets. A batch whose largest top-k exceeds the last bucket
# (or a batch with any row without top-k) uses the upstream full-vocab sort.
TOP_K_BUCKETS = (32, 64, 128, 256)
# Vocabulary group length for the candidate search.
_GROUP_LEN = 128


def top_k_bucket(max_top_k: int) -> Optional[int]:
    """Smallest static bucket >= max_top_k, or None if none fits."""
    for bucket in TOP_K_BUCKETS:
        if max_top_k <= bucket:
            return bucket
    return None


def _grouped_topk(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k values (descending) and indices of each row of ``logits``.

    The maxima of the k groups with the largest maxima are k values that are >=
    every value outside those groups, so the top-k of those groups is a top-k of
    the row. Indices may point past the vocabulary only for -inf values (the
    padding), when a row has fewer than k finite logits.
    """
    batch, vocab = logits.shape
    groups = -(-vocab // _GROUP_LEN)
    if groups <= k:
        return logits.topk(k, dim=-1)
    if groups * _GROUP_LEN != vocab:
        logits = torch.nn.functional.pad(logits, (0, groups * _GROUP_LEN - vocab), value=-float("inf"))
    grouped = logits.view(batch * groups, _GROUP_LEN)
    # Eager HPU gather does not support int64 values; vocab ids fit in int32.
    group_ids = logits.view(batch, groups, _GROUP_LEN).amax(dim=-1).topk(k, dim=-1)[1].to(torch.int32)
    rows = group_ids + torch.arange(batch, device=logits.device, dtype=torch.int32).unsqueeze(1) * groups
    candidates = grouped.index_select(0, rows.reshape(-1)).view(batch, k * _GROUP_LEN)
    vals, pos = candidates.topk(k, dim=-1)
    pos = pos.to(torch.int32)
    idx = group_ids.gather(1, pos // _GROUP_LEN) * _GROUP_LEN + pos % _GROUP_LEN
    return vals, idx.to(torch.long)


def top_k_top_p_candidates(
    logits: torch.Tensor,
    k: torch.Tensor,
    p: Optional[torch.Tensor],
    max_top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Masked candidate logits and their vocab indices, both [batch, max_top_k].

    Every row must have k <= max_top_k. The candidates are in ascending order;
    masked ones are -inf. Same masks as ``apply_top_k_top_p_pytorch`` (top-k
    keeps ties of the k-th value that fall inside the first ``max_top_k``).
    """
    vals, idx = _grouped_topk(logits, max_top_k)
    # Ascending order, as in the upstream sort, so the top-p cumulative sum is
    # computed the same way.
    vals, idx = vals.flip(-1), idx.flip(-1)
    kth = vals.gather(1, (max_top_k - k.to(torch.long)).unsqueeze(1))
    vals = vals.masked_fill(vals < kth, -float("inf"))
    if p is not None:
        probs_sum = vals.softmax(dim=-1).cumsum(dim=-1)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        # at least one
        top_p_mask[:, -1] = False
        vals = vals.masked_fill(top_p_mask, -float("inf"))
    return vals, idx


def apply_top_k_top_p_topk_first(
    logits: torch.Tensor,
    k: torch.Tensor,
    p: Optional[torch.Tensor],
    max_top_k: int,
) -> torch.Tensor:
    """Full-vocab logits with the top-k and top-p masks applied.

    Every row must have k <= max_top_k. Returns a new tensor.
    """
    vals, idx = top_k_top_p_candidates(logits, k, p, max_top_k)
    batch, vocab = logits.shape
    # Room for indices of -inf padding past the vocabulary.
    padded = -(-vocab // _GROUP_LEN) * _GROUP_LEN
    out = torch.full((batch, padded), -float("inf"), dtype=logits.dtype, device=logits.device)
    return out.scatter_(-1, idx, vals)[:, :vocab]


class HPUTopKTopPSampler(TopKTopPSampler):
    """TopKTopPSampler with a top-k-first path for small, host-known top-k.

    The model runner sets ``max_top_k`` before each sampling call to a static
    bucket (``top_k_bucket``) that bounds every row's top-k, or to None.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_top_k: Optional[int] = None
        self.forward = self.forward_hpu

    def forward_hpu(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: Optional[torch.Tensor],
        p: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        max_top_k = self.max_top_k
        if k is None or max_top_k is None or max_top_k >= logits.shape[-1]:
            return self.forward_native(logits, generators, k, p)
        if generators or self.logprobs_mode in ("processed_logits", "processed_logprobs"):
            # Full-vocab logits are needed, or seeded requests must draw the
            # same noise as without the bound.
            logits = apply_top_k_top_p_topk_first(logits, k, p, max_top_k)
            logits_to_return = None
            if self.logprobs_mode == "processed_logits":
                logits_to_return = logits
            elif self.logprobs_mode == "processed_logprobs":
                logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)
            probs = logits.softmax(dim=-1, dtype=torch.float32)
            return random_sample(probs, generators, self.use_fp64_gumbel), logits_to_return
        # Sample among the candidates: masked candidates have probability 0,
        # like every logit outside them.
        vals, idx = top_k_top_p_candidates(logits, k, p, max_top_k)
        probs = vals.softmax(dim=-1, dtype=torch.float32)
        chosen = random_sample(probs, generators, self.use_fp64_gumbel)
        return idx.to(torch.int32).gather(1, chosen.unsqueeze(1)).view(-1).to(torch.long), None
