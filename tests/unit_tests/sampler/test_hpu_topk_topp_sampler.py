# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import pytest
import torch
import habana_frameworks.torch  # noqa: F401

from vllm.sampling_params import SamplingParams
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

from vllm_gaudi.v1.sample.hpu_topk_topp_sampler import (HPUTopKTopPSampler, apply_top_k_top_p_topk_first, top_k_bucket,
                                                        top_k_top_p_candidates)
from vllm_gaudi.v1.worker.hpu_input_batch import CachedRequestState, InputBatch

DEVICE = "hpu"


def _distinct_logits(batch_size: int, vocab_size: int) -> torch.Tensor:
    # Distinct, well separated logits: the top-p boundary never falls within
    # float rounding of a cumulative sum.
    rows = [torch.randperm(vocab_size).float() * 0.05 for _ in range(batch_size)]
    return torch.stack(rows).to(DEVICE)


def test_top_k_bucket():
    assert top_k_bucket(1) == 32
    assert top_k_bucket(20) == 32
    assert top_k_bucket(33) == 64
    assert top_k_bucket(256) == 256
    assert top_k_bucket(257) is None


@pytest.mark.parametrize("vocab_size", [300, 1000, 4097, 50000])
@pytest.mark.parametrize("use_top_p", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_topk_first_matches_upstream(vocab_size: int, use_top_p: bool, compiled: bool):
    torch.manual_seed(0)
    batch_size = 8
    logits = _distinct_logits(batch_size, vocab_size)
    k = torch.tensor([1, 2, 5, 20, 31, 32, 7, 20], dtype=torch.int32, device=DEVICE)
    p = torch.tensor([0.1, 0.5, 0.9, 0.95, 1.0, 0.8, 0.99, 0.3], device=DEVICE) if use_top_p else None
    fn = apply_top_k_top_p_topk_first
    if compiled:
        fn = torch.compile(fn, backend="hpu_backend", dynamic=False)
    out = fn(logits.clone(), k, p, 32).cpu()
    ref = apply_top_k_top_p_pytorch(logits.clone().cpu(), k.cpu(), None if p is None else p.cpu())
    assert torch.equal(torch.isfinite(out), torch.isfinite(ref))
    assert torch.equal(out[torch.isfinite(out)], ref[torch.isfinite(ref)])
    if not use_top_p:
        assert torch.equal(torch.isfinite(out).sum(-1), k.cpu().long())


def test_topk_first_keeps_ties_like_upstream():
    # Ties of the k-th value are kept, as in the upstream threshold compare.
    logits = torch.tensor([[5.0, 4.0, 4.0, 4.0, 1.0, 0.0] + [-1.0] * 40], device=DEVICE)
    k = torch.tensor([2], dtype=torch.int32, device=DEVICE)
    out = apply_top_k_top_p_topk_first(logits.clone(), k, None, 32).cpu()
    ref = apply_top_k_top_p_pytorch(logits.clone().cpu(), k.cpu(), None)
    assert torch.equal(torch.isfinite(out), torch.isfinite(ref))
    assert torch.isfinite(out).sum().item() == 4


def test_topk_first_few_finite_logits():
    # Rows with fewer finite logits than max_top_k pick -inf padding as candidates.
    vocab_size = 5000
    logits = torch.full((2, vocab_size), -float("inf"), device=DEVICE)
    logits[0, [3, 4000, vocab_size - 1]] = torch.tensor([1.0, 2.0, 0.5], device=DEVICE)
    logits[1, 17] = 0.0
    k = torch.tensor([20, 5], dtype=torch.int32, device=DEVICE)
    p = torch.tensor([0.99, 0.9], device=DEVICE)
    out = apply_top_k_top_p_topk_first(logits.clone(), k, p, 32).cpu()
    ref = apply_top_k_top_p_pytorch(logits.clone().cpu(), k.cpu(), p.cpu())
    assert out.shape == ref.shape
    assert torch.equal(torch.isfinite(out), torch.isfinite(ref))
    sampler = HPUTopKTopPSampler("raw_logprobs")
    sampler.max_top_k = 32
    for _ in range(5):
        tokens, _ = sampler(logits.clone(), {}, k, p)
        assert torch.isfinite(ref.gather(1, tokens.cpu().unsqueeze(1))).all()


@pytest.mark.parametrize("vocab_size", [1000, 50000])
def test_candidates_probs_match_upstream(vocab_size: int):
    torch.manual_seed(0)
    logits = _distinct_logits(8, vocab_size) * 0.02
    k = torch.tensor([1, 2, 5, 20, 31, 32, 7, 20], dtype=torch.int32, device=DEVICE)
    p = torch.tensor([0.1, 0.5, 0.9, 0.95, 1.0, 0.8, 0.99, 0.3], device=DEVICE)
    vals, idx = top_k_top_p_candidates(logits.clone(), k, p, 32)
    probs = torch.zeros(8, vocab_size).scatter_(1, idx.cpu(), vals.softmax(-1, dtype=torch.float32).cpu())
    ref = apply_top_k_top_p_pytorch(logits.clone().cpu(), k.cpu(), p.cpu()).softmax(-1)
    torch.testing.assert_close(probs, ref)


@pytest.mark.parametrize("logprobs_mode", ["raw_logprobs", "processed_logprobs"])
def test_hpu_topk_topp_sampler_kept_tokens(logprobs_mode: str):
    torch.manual_seed(0)
    batch_size, vocab_size = 4, 5000
    logits = _distinct_logits(batch_size, vocab_size) * 0.2
    k = torch.tensor([20, 5, 32, 10], dtype=torch.int32, device=DEVICE)
    p = torch.tensor([0.95, 0.9, 1.0, 0.8], device=DEVICE)
    kept = torch.isfinite(apply_top_k_top_p_pytorch(logits.clone().cpu(), k.cpu(), p.cpu()))
    sampler = HPUTopKTopPSampler(logprobs_mode)
    results = []
    for max_top_k in (None, 32):
        sampler.max_top_k = max_top_k
        set_random_seed(123)
        tokens, processed = sampler(logits.clone(), {}, k, p)
        assert kept.gather(1, tokens.cpu().unsqueeze(1)).all()
        results.append((tokens.cpu(), None if processed is None else processed.cpu()))
    if logprobs_mode == "processed_logprobs":
        # The full-vocab logits are built, so the same noise picks the same tokens.
        assert torch.equal(results[0][0], results[1][0])
        torch.testing.assert_close(results[0][1], results[1][1])


def test_hpu_topk_topp_sampler_seeded_requests():
    # Seeded rows draw full-vocab noise from their generator, as without the bound.
    torch.manual_seed(0)
    logits = _distinct_logits(3, 5000) * 0.2
    k = torch.tensor([20, 5, 32], dtype=torch.int32, device=DEVICE)
    p = torch.tensor([0.95, 0.9, 1.0], device=DEVICE)
    sampler = HPUTopKTopPSampler("raw_logprobs")
    results = []
    for max_top_k in (None, 32):
        sampler.max_top_k = max_top_k
        generators = {i: torch.Generator(device=DEVICE).manual_seed(7 + i) for i in range(3)}
        results.append(sampler(logits.clone(), generators, k, p)[0].cpu())
    assert torch.equal(results[0], results[1])


def test_hpu_topk_topp_sampler_distribution():
    # Candidate sampling follows the top-k/top-p probabilities (CPU RNG).
    torch.manual_seed(0)
    logits = torch.full((4096, 3000), -5.0)
    logits[:, [10, 2000, 2999]] = torch.tensor([2.0, 1.0, 0.0])
    k = torch.full((4096, ), 2, dtype=torch.int32)
    p = torch.full((4096, ), 0.99)
    sampler = HPUTopKTopPSampler("raw_logprobs")
    sampler.max_top_k = 32
    tokens, _ = sampler(logits, {}, k, p)
    assert set(tokens.unique().tolist()) <= {10, 2000}
    expected = torch.tensor([2.0, 1.0]).softmax(-1)[0].item()
    assert abs((tokens == 10).float().mean().item() - expected) < 0.03


def _request(index: int, top_k: int) -> CachedRequestState:
    return CachedRequestState(
        req_id=f"req_{index}",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(temperature=1.0, top_k=top_k, top_p=0.9),
        pooling_params=None,
        mm_features=[],
        block_ids=([], ),
        generator=None,
        num_computed_tokens=0,
        output_token_ids=[],
    )


@pytest.mark.parametrize("top_ks,selected,expected", [
    ([20, 50, 5], [0, 2], 20),
    ([20, 50, 5], [0, 1, 2], 50),
    ([20, -1, 5], [0, 2], 20),
    ([20, -1, 5], [0, 1], None),
    ([-1, -1], [0, 1], None),
])
def test_input_batch_selected_max_top_k(top_ks: list[int], selected: list[int], expected: Optional[int]):
    input_batch = InputBatch(
        max_num_reqs=len(top_ks),
        max_model_len=64,
        max_num_batched_tokens=64,
        device=torch.device(DEVICE),
        pin_memory=is_pin_memory_available(),
        vocab_size=1024,
        block_sizes=[1],
        kernel_block_sizes=[1],
    )
    for i, top_k in enumerate(top_ks):
        input_batch.add_request(_request(i, top_k))
    input_batch.make_selective_sampling_metadata([(f"req_{i}", []) for i in selected])
    assert input_batch.selected_max_top_k == expected
