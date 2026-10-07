# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
import pytest
import torch
import habana_frameworks.torch as htorch
from utils import get_data_path, create_row_parallel_linear, create_fused_moe
from unittest.mock import MagicMock
from vllm_gaudi.ops.hpu_fp8 import Fp8LinearMethod, HPUFp8MoEMethod
from vllm_gaudi.utils import HPUCompileConfig
from vllm.forward_context import override_forward_context
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from safetensors import safe_open


def test_fp8_linear_method(default_vllm_config: None, dist_init, monkeypatch):
    monkeypatch.setenv("VLLM_HPU_FORCE_CHANNEL_FP8", "0")
    config = {'activation_scheme': 'dynamic', 'fmt': 'e4m3', 'quant_method': 'fp8', 'weight_block_size': [128, 128]}
    oot_quant_config = Fp8Config.from_config(config)

    # Prepare linear layer with oot Fp8LinearMethod
    oot_op = create_row_parallel_linear(input_size=256, output_size=256, quant_config=oot_quant_config).to("hpu")
    assert isinstance(oot_op.quant_method, Fp8LinearMethod)

    # Weight and weight_scale_inv were extracted from first RowParallelLinear layer of Qwen/Qwen3-8B-FP8
    # (with adjusted shapes, to make tensors smaller)
    with safe_open(get_data_path("data/fp8/linear.safetensors"), framework="pt", device="hpu") as f:
        oot_op.weight.copy_(f.get_tensor("weight"))
        oot_op.weight_scale_inv.copy_(f.get_tensor("weight_scale_inv"))
    oot_op.quant_method.process_weights_after_loading(oot_op)

    if not htorch.utils.internal.is_lazy():
        # Setting fullgraph to False, because currently there is a graph break
        compile_config = HPUCompileConfig(fullgraph=False)
        oot_op = torch.compile(oot_op, **compile_config.get_compile_args())

    # Input and expected output
    # Output tensor holds the data that was returned by cuda implementation of Fp8LinearMethod for given input
    # (Fp8LinearMethod was triggered offline with the same input as below to get the ref_output)
    with safe_open(get_data_path("data/fp8/linear.safetensors"), framework="pt", device="hpu") as f:
        input = f.get_tensor("input")
        ref_output = f.get_tensor("ref_output")

    # Execute layer
    out = oot_op(input)

    # Check correctness
    torch.testing.assert_close(ref_output, out, atol=1e-3, rtol=1e-3)


@pytest.mark.xfail(reason="Failed due upstream MOE refactor - PR's: 30627, 30825, 31036")
def test_fp8_moe_method(default_vllm_config: None, dist_init, monkeypatch):
    monkeypatch.setenv("VLLM_HPU_FORCE_CHANNEL_FP8", "0")
    config = {
        'activation_scheme': 'dynamic',
        'modules_to_not_convert': [],
        'fmt': 'e4m3',
        'quant_method': 'fp8',
        'weight_block_size': [128, 128]
    }
    oot_quant_config = Fp8Config.from_config(config)

    # Prepare FusedMoE layer with oot HPUFp8MoEMethod
    oot_op = create_fused_moe(oot_quant_config).to("hpu")
    assert isinstance(oot_op.routed_experts.quant_method, HPUFp8MoEMethod)

    # Weights were extracted from first FusedMoE layer of Qwen/Qwen3-30B-A3B-FP8
    # (with adjusted shapes, to make tensors smaller)
    with safe_open(get_data_path("data/fp8/moe.safetensors"), framework="pt", device="hpu") as f:
        w13_weight = f.get_tensor("w13_weight")
        oot_op.routed_experts.w13_weight.copy_(w13_weight.repeat(128, 1, 1))

        w13_weight_scale_inv = f.get_tensor("w13_weight_scale_inv")
        oot_op.routed_experts.w13_weight_scale_inv.copy_(w13_weight_scale_inv.repeat(128, 1, 1))

        w2_weight = f.get_tensor("w2_weight")
        oot_op.routed_experts.w2_weight.copy_(w2_weight.repeat(128, 1, 1))

        w2_weight_scale_inv = f.get_tensor("w2_weight_scale_inv")
        oot_op.routed_experts.w2_weight_scale_inv.copy_(w2_weight_scale_inv.repeat(128, 1, 1))

    oot_op.routed_experts.quant_method.process_weights_after_loading(oot_op.routed_experts)

    if not htorch.utils.internal.is_lazy():
        compile_config = HPUCompileConfig()
        oot_op = torch.compile(oot_op, **compile_config.get_compile_args())

    # Input and expected output
    # Output tensor holds the data that was returned by cuda implementation of Fp8MoEMethod for given input
    # (Fp8MoEMethod was triggered offline with the same input as below to get the ref_output)
    with safe_open(get_data_path("data/fp8/moe.safetensors"), framework="pt", device="hpu") as f:
        hidden_states = f.get_tensor("hidden_states")
        router_logits = f.get_tensor("router_logits")
        ref_output = f.get_tensor("ref_output")

    # Execute layer
    mock_ctx = MagicMock(spec=["dp_metadata"])
    mock_ctx.dp_metadata = None
    with override_forward_context(mock_ctx):
        out = oot_op.forward_impl(hidden_states, router_logits)

    # Check correctness
    torch.testing.assert_close(ref_output, out, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("ep_rank", [0, 2])
@pytest.mark.parametrize("tokens", [1, 16, 64])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("fused_down", [False, True])
def test_dense_silu_fp8_moe_matches_stock_op(ep_rank: int, tokens: int, compiled: bool, fused_down: bool):
    """The dense all-local-experts path must be as accurate as the stock per-channel FP8 op.

    Both quantize activations to FP8 with different rounding, so each is compared
    with an fp32 reference on the same FP8 weights. With fused_down the dense path
    uses the requantized [H, E*I] copy of w2.
    """
    from vllm_gaudi.extension.ops import VllmMixtureOfExpertsOpFP8PerChannel, dynamic_quant
    from vllm_gaudi.ops.hpu_moe_combine import dense_silu_fp8_moe, prepare_dense_fused_down

    torch.manual_seed(0)
    local, global_experts, hidden, inter, top_k = 16, 64, 256, 128, 4
    experts_min = ep_rank * local
    w13, s13 = dynamic_quant(torch.randn(local, 2 * inter, hidden, dtype=torch.bfloat16, device="hpu") * 0.05)
    w2, s2 = dynamic_quant(torch.randn(local, hidden, inter, dtype=torch.bfloat16, device="hpu") * 0.05)
    s13, s2 = s13.squeeze(-1), s2.squeeze(-1)
    layer = SimpleNamespace(w13_weight=w13,
                            w2_weight=w2,
                            w13_weight_scale_inv=s13,
                            w2_weight_scale_inv=s2,
                            local_num_experts=local,
                            moe_config=SimpleNamespace(ep_rank=ep_rank))
    op = VllmMixtureOfExpertsOpFP8PerChannel(global_experts, local, experts_min, experts_min + local - 1)
    for j in range(local):
        op.w13_list[j].set_weight(w13[j])
        op.w13_list[j].set_scale_inv_fp8(s13[j])
        op.w2_list[j].set_weight(w2[j])
        op.w2_list[j].set_scale_inv_fp8(s2[j])

    x = torch.randn(tokens, hidden, dtype=torch.bfloat16)
    # Distinct global experts per token; every token hits at least one local expert.
    topk_ids = torch.stack([torch.randperm(global_experts)[:top_k] for _ in range(tokens)])
    topk_ids[:, 0] = experts_min + torch.arange(tokens) % local
    for t in range(tokens):
        while (topk_ids[t, 1:] == topk_ids[t, 0]).any():
            topk_ids[t, 1:] = torch.randperm(global_experts)[:top_k - 1]
    topk_weights = torch.softmax(torch.randn(tokens, top_k), dim=-1).to(torch.bfloat16)

    w13_ref = (w13.float() * s13.unsqueeze(-1)).cpu()
    w2_ref = (w2.float() * s2.unsqueeze(-1)).cpu()
    ref = torch.zeros(tokens, hidden)
    for t in range(tokens):
        for k in range(top_k):
            j = int(topk_ids[t, k]) - experts_min
            if 0 <= j < local:
                gate, up = (w13_ref[j] @ x[t].float()).chunk(2)
                ref[t] += float(topk_weights[t, k]) * (w2_ref[j] @ (torch.nn.functional.silu(gate) * up))

    if fused_down:
        prepare_dense_fused_down(layer)
    dense = dense_silu_fp8_moe
    stock = op.forward
    if compiled:
        dense = torch.compile(dense, backend="hpu_backend", dynamic=False)
        stock = torch.compile(stock, backend="hpu_backend", dynamic=False)
    x, topk_ids, topk_weights = x.to("hpu"), topk_ids.to("hpu"), topk_weights.to("hpu")
    out = dense(layer, x, topk_ids, topk_weights).float().cpu()
    out_stock = stock(x, topk_ids, topk_weights, permuted_weights=True, activation="silu").float().cpu()
    assert out.shape == (tokens, hidden)
    err = ((out - ref).norm() / ref.norm()).item()
    err_stock = ((out_stock - ref).norm() / ref.norm()).item()
    assert err < 0.08, err
    # One token is too few samples to rank the two roundings.
    if tokens >= 16:
        assert err <= 1.15 * err_stock + 5e-3, (err, err_stock)
