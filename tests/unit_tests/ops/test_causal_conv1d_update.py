# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the decode-time causal conv1d state update (hpu_causal_conv1d_update)."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from vllm_gaudi.ops.causal_conv1d_pytorch import (_depthwise_conv1d_tpc, _depthwise_conv1d_tpc_channels_last,
                                                  hpu_causal_conv1d_update)
from vllm.platforms import current_platform

DEVICE = current_platform.device_type


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("width,seq_len", [(2, 2), (4, 4), (4, 7)])
@pytest.mark.parametrize("with_bias", [False, True])
def test_channels_last_matches_channels_first(dtype, width, seq_len, with_bias):
    torch.manual_seed(0)
    x = torch.randn(3, 64, seq_len, dtype=dtype, device=DEVICE)
    weight = torch.randn(64, width, dtype=dtype, device=DEVICE)
    bias = torch.randn(64, dtype=dtype, device=DEVICE) if with_bias else None
    expected = _depthwise_conv1d_tpc(x, weight, bias)
    result = _depthwise_conv1d_tpc_channels_last(x.transpose(1, 2).contiguous(), weight, bias)
    assert result.shape == (3, seq_len - width + 1, 64)
    torch.testing.assert_close(result.transpose(1, 2), expected, rtol=0, atol=0)


def _reference_update(x, conv_states, weight, bias, indices):
    """Channels-first reference: gather, conv1d, silu, write back the last width - 1 inputs."""
    state_len = weight.shape[1] - 1
    safe = torch.remainder(indices.long(), conv_states.shape[0])
    init = conv_states[safe, -state_len:, :].transpose(1, 2).float()  # [N, dim, state_len]
    seq = torch.cat([init, x.float().unsqueeze(-1)], dim=2)
    out = F.silu(F.conv1d(seq, weight.float().unsqueeze(1), None if bias is None else bias.float(), groups=x.shape[1]))
    conv_states[safe, -state_len:, :] = seq[:, :, -state_len:].transpose(1, 2).to(conv_states.dtype)
    return out.squeeze(-1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("extra_state_cols", [0, 2])
@pytest.mark.parametrize("compiled", [False, True])
def test_conv1d_update_matches_reference(dtype, extra_state_cols, compiled):
    torch.manual_seed(0)
    num_seqs, dim, width, slots = 6, 128, 4, 10
    x = torch.randn(num_seqs, dim, dtype=dtype, device=DEVICE)
    weight = torch.randn(dim, width, dtype=dtype, device=DEVICE) * 0.5
    bias = torch.randn(dim, dtype=dtype, device=DEVICE) * 0.1
    conv_states = torch.randn(slots, width - 1 + extra_state_cols, dim, dtype=dtype, device=DEVICE)
    # Padded batch entries use PAD_SLOT_ID (-1), which maps to the last slot.
    indices = torch.tensor([3, 0, 7, -1, 5, -1], dtype=torch.int32, device=DEVICE)
    query_start_loc = torch.arange(num_seqs + 1, dtype=torch.int32, device=DEVICE)

    ref_states = conv_states.clone()
    expected = _reference_update(x, ref_states, weight, bias, indices)

    def run(x, conv_states, weight, bias, indices, query_start_loc):
        return hpu_causal_conv1d_update(x=x,
                                        conv_state=conv_states,
                                        weight=weight,
                                        bias=bias,
                                        activation="silu",
                                        conv_state_indices=indices,
                                        query_start_loc=query_start_loc)

    fn = torch.compile(run, backend="hpu_backend", dynamic=False) if compiled else run
    result = fn(x, conv_states, weight, bias, indices, query_start_loc)

    assert result.shape == (num_seqs, dim) and result.dtype == dtype
    tol = 2e-2 if dtype == torch.bfloat16 else 1e-5
    # Padded entries share the garbage slot; only compare real sequences.
    real = indices >= 0
    torch.testing.assert_close(result[real].float(), expected[real], rtol=tol, atol=tol)
    torch.testing.assert_close(conv_states[:slots - 1], ref_states[:slots - 1], rtol=0, atol=0)
