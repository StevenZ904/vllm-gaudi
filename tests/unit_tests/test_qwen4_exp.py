# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the HPU Qwen4Exp n-gram embedding (PLE) building blocks."""

from types import SimpleNamespace

import pytest
import torch
import habana_frameworks.torch  # noqa: F401
from vllm.config import VllmConfig, set_current_vllm_config

import vllm_gaudi.models.qwen4_exp as qwen4_exp

VOCAB = 1000
EOS = 999


@pytest.fixture
def default_vllm_config():
    """VllmConfig with the minimal model_config the linear layers read."""
    vllm_config = VllmConfig()
    vllm_config.model_config = SimpleNamespace(dtype=torch.bfloat16, is_moe=False, hf_config=None, quantization=None)
    with set_current_vllm_config(vllm_config):
        yield


def _config(**overrides):
    cfg = dict(hidden_size=64,
               hc_count=4,
               ngram_size=3,
               heads_per_ngram=4,
               ple_conv_kernel_size=4,
               ple_embed_dim=64,
               rms_norm_eps=1e-6,
               ngram_vocab_size_base=5000,
               make_ngram_vocab_size_divisible_by=128,
               split_ngram_parts=8,
               vocab_size=VOCAB,
               eos_token_id=EOS)
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def _hash_params(emb, seed=0):
    """Checkpoint-style hash parameters for ``emb`` (layer 0)."""
    gen = torch.Generator().manual_seed(seed)
    primes, prime = [], _config().ngram_vocab_size_base - 1
    for _ in range(emb.ngram_heads):
        prime = qwen4_exp._next_prime(prime)
        primes.append(prime)
    offsets = [0]
    for p in primes[:-1]:
        offsets.append(offsets[-1] + p)
    bound = (1 << 63) // VOCAB
    mult = torch.randint(0, bound // 2, (emb.ngram_size, ), generator=gen, dtype=torch.int64) * 2 + 1
    return mult, torch.tensor(primes), torch.tensor(offsets)


def _make_embedding(seed=0, fp8_table=True):
    emb = qwen4_exp.HpuQwen4ExpNGramEmbedding(_config(),
                                              embedding_dim=64,
                                              ple_dense_layer_id=0,
                                              fp8_table=fp8_table,
                                              prefix="ple")
    mult, primes, offsets = _hash_params(emb, seed)
    emb.load_weights([("layer_multipliers", mult), ("ngram_heads_vocab_sizes", primes),
                      ("ngram_heads_offsets", offsets)])
    return emb, mult.tolist(), primes.tolist(), offsets.tolist()


def _reference_rows(history, mult, primes, offsets, heads_per_ngram):
    """Python-int reference of the upstream int64 n-gram hash."""
    rows = []
    for p in range(2, len(history)):
        cur, prev1, prev2 = history[p], history[p - 1], history[p - 2]
        if prev1 == EOS:  # the trigram does not reach across a document boundary
            prev2 = EOS
        bigram = (cur * mult[0]) ^ (prev1 * mult[1])
        trigram = bigram ^ (prev2 * mult[2])
        rows.append(
            [bigram % primes[h] + offsets[h] for h in range(heads_per_ngram)] +
            [trigram % primes[heads_per_ngram + h] + offsets[heads_per_ngram + h] for h in range(heads_per_ngram)])
    return rows


@pytest.mark.parametrize("compiled", [False, True])
def test_mod_u63_matches_int64(compiled: bool):
    gen = torch.Generator().manual_seed(0)
    values = torch.randint(0, 1 << 62, (4096, ), generator=gen, dtype=torch.int64) * 2 + 1
    values[:4] = torch.tensor([0, 1, (1 << 63) - 1, 1 << 32])
    primes = torch.tensor([2, 3, 4999, 5003, 20000003, (1 << 25) - 39], dtype=torch.int64)
    hi = (values >> 32).to(torch.int32)
    lo = values & 0xFFFFFFFF
    lo = torch.where(lo >= (1 << 31), lo - (1 << 32), lo).to(torch.int32)

    fn = qwen4_exp._mod_u63
    if compiled:
        fn = torch.compile(fn, backend="hpu_backend", dynamic=False)
    out = fn(hi.to("hpu"), lo.to("hpu"), primes.to(torch.int32).to("hpu")).cpu()

    expected = torch.tensor([[v % p for p in primes.tolist()] for v in values.tolist()], dtype=torch.int64)
    torch.testing.assert_close(out.to(torch.int64), expected, rtol=0, atol=0)


@pytest.mark.parametrize("compiled", [False, True])
def test_ngram_hash_matches_reference(default_vllm_config, dist_init, compiled: bool):
    emb, mult, primes, offsets = _make_embedding()
    emb = emb.to("hpu")
    gen = torch.Generator().manual_seed(1)
    tokens = torch.randint(0, VOCAB - 1, (2, 34), generator=gen)
    tokens[0, 0] = EOS  # document boundaries at, right after and two after the context
    tokens[0, 7] = EOS
    tokens[1, 10:12] = EOS
    tokens[1, 20] = EOS

    fn = emb.hash_ids
    if compiled:
        fn = torch.compile(fn, backend="hpu_backend", dynamic=False)
    rows = fn(tokens.to(torch.int32).to("hpu")).cpu().view(2, 32, -1)

    for b in range(2):
        expected = _reference_rows(tokens[b].tolist(), mult, primes, offsets, emb.heads_per_ngram)
        assert rows[b].tolist() == expected


def test_ngram_lookup_dequantizes_full_e4m3_range(default_vllm_config, dist_init):
    emb, *_ = _make_embedding()
    gen = torch.Generator().manual_seed(2)
    # Random FP8 bytes, including codes above the Gaudi2 e4m3 range (|x| > 240)
    # and the NaN code (0x7f), which the lookup table maps to zero.
    raw = torch.randint(0, 256, (emb.num_rows, emb.head_dim), generator=gen, dtype=torch.int32).to(torch.uint8)
    raw[:, 0] = 0x7E  # 448
    raw[:, 1] = 0x7F  # NaN
    weight = raw.view(torch.float8_e4m3fn)
    shards = torch.split(weight, emb.checkpoint_shard_rows)
    scale = torch.tensor([0.25])
    emb.load_weights([(f"ngram_embedding.shard_{i}.weight", s)
                      for i, s in enumerate(shards)] + [("ngram_embedding.weight_scale", scale)])
    emb = emb.to("hpu")

    history = torch.randint(0, VOCAB, (1, 18), generator=gen, dtype=torch.int32)
    out = emb(history.to("hpu")).cpu().float()
    rows = emb.hash_ids(history.to("hpu")).cpu().long()

    expected = torch.nan_to_num(weight.float(), nan=0.0)[rows.reshape(-1)] * scale
    expected = expected.view(rows.shape[0], -1).to(torch.bfloat16).float()
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    assert (out.view(-1, emb.head_dim)[:, 0] == 448 * 0.25).all()


def test_ngram_lookup_bf16_table(default_vllm_config, dist_init):
    emb, *_ = _make_embedding(fp8_table=False)
    gen = torch.Generator().manual_seed(3)
    weight = torch.randn(emb.num_rows, emb.head_dim, generator=gen).to(torch.bfloat16)
    shards = torch.split(weight, emb.checkpoint_shard_rows)
    emb.load_weights([(f"ngram_embedding.shard_{i}.weight", s) for i, s in enumerate(shards)])
    emb = emb.to("hpu")

    history = torch.randint(0, VOCAB, (2, 12), generator=gen, dtype=torch.int32)
    out = emb(history.to("hpu")).cpu()
    rows = emb.hash_ids(history.to("hpu")).cpu().long()

    expected = weight[rows.reshape(-1)].view(rows.shape[0], -1)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="does not match"):
        emb.load_weights([("ngram_embedding.shard_0.weight", shards[0].to(torch.float8_e4m3fn))])


def test_ngram_table_load_is_validated(default_vllm_config, dist_init):
    emb, *_ = _make_embedding(fp8_table=False)
    weight = torch.randn(emb.num_rows, emb.head_dim).to(torch.bfloat16)
    shards = torch.split(weight, emb.checkpoint_shard_rows)
    assert len(shards) == emb.split_parts
    with pytest.raises(ValueError, match="unexpected PLE table shard"):
        emb.load_weights([("ngram_embedding.shard_0.weight", shards[0][:-1])])
    with pytest.raises(ValueError, match="unexpected PLE table shard"):
        emb.load_weights([(f"ngram_embedding.shard_{emb.split_parts}.weight", shards[0])])

    # A missing shard must not silently leave zero rows behind.
    emb.load_weights([(f"ngram_embedding.shard_{i}.weight", s) for i, s in enumerate(shards) if i != 3])
    with pytest.raises(ValueError, match="shards missing"):
        emb.check_loaded()
    emb.load_weights([("ngram_embedding.shard_3.weight", shards[3])])
    emb.check_loaded()


@pytest.mark.parametrize("use_combine", [True, False])
@pytest.mark.parametrize("compiled", [False, True])
def test_gated_residual_matches_upstream(default_vllm_config, dist_init, use_combine: bool, compiled: bool):
    from vllm.models.qwen4_exp.common.hyperconnection import GatedResidual, HyperConnectionConfig

    cfg = _config(hc_lowrank=16)
    ref = GatedResidual(HyperConnectionConfig(hc_count=cfg.hc_count,
                                              hidden_size=cfg.hidden_size,
                                              params_dtype=torch.float32,
                                              hc_lowrank=cfg.hc_lowrank,
                                              rms_norm_eps=cfg.rms_norm_eps,
                                              hc_per_branch_norm=True),
                        use_combine=use_combine)
    hc = qwen4_exp.HpuGatedResidual(cfg, use_combine=use_combine, prefix="hc")
    gen = torch.Generator().manual_seed(4)
    with torch.no_grad():
        for p in ref.parameters():
            p.copy_(torch.randn(p.shape, generator=gen) * 0.1)
        hc.hc_norm.weight.copy_(ref.hc_norm.weight)
        hc.input_mix_weight_up.weight.copy_(ref.input_mix_weight_up.weight)
        if use_combine:
            # The checkpoint's down projection and injection logits share one GEMM.
            hc.input_mix_weight_down_block_inject.weight.copy_(
                torch.cat([ref.input_mix_weight_down.weight, ref.block_inject_weight.weight]))
        else:
            hc.input_mix_weight_down.weight.copy_(ref.input_mix_weight_down.weight)
    hc = hc.to("hpu")

    hyper_input = torch.randn(7, cfg.hc_count * cfg.hidden_size, generator=gen)
    block_output = torch.randn(7, cfg.hidden_size, generator=gen)
    mix = torch.compile(hc.mix, backend="hpu_backend", dynamic=False) if compiled else hc.mix
    with torch.no_grad():
        ref_mixed, residuals = ref.mix(hyper_input)
        mixed, injection = mix(hyper_input.to("hpu"))
        torch.testing.assert_close(mixed.cpu(), ref_mixed, rtol=1e-3, atol=1e-3)
        if use_combine:
            out = hc.combine(hyper_input.to("hpu"), block_output.to("hpu"), injection)
            torch.testing.assert_close(out.cpu(), ref.combine(block_output, residuals), rtol=1e-3, atol=1e-3)
        else:
            assert injection is None


class _Host:
    """Stand-in for HpuQwen4ExpPLEHostGDN's PLE state slots."""

    def __init__(self, ple, slots=4):
        self.ple_conv_state = torch.zeros(slots,
                                          ple.conv_state_len,
                                          ple.hc_hidden_size,
                                          dtype=torch.bfloat16,
                                          device="hpu")
        self.ple_ctx_state = torch.full((slots, ple.context_len), EOS, dtype=torch.int32, device="hpu")

    def _resolve_state_indices(self, attn_metadata):
        return attn_metadata.load_indices_tensor


def _prefill_md(slot, num_real, has_initial):
    return SimpleNamespace(is_prompt=True,
                           load_indices_tensor=torch.tensor([slot], device="hpu"),
                           query_start_loc_p=torch.tensor([0, num_real], dtype=torch.int32, device="hpu"),
                           has_initial_states_p=torch.tensor([has_initial], device="hpu"))


def _decode_md(slot):
    return SimpleNamespace(is_prompt=False,
                           load_indices_tensor=torch.tensor([slot], device="hpu"),
                           query_start_loc_p=None,
                           has_initial_states_p=None)


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def test_ple_layer_compiled_and_streaming_match_eager(default_vllm_config, dist_init, monkeypatch):
    torch.manual_seed(0)
    ple = qwen4_exp.HpuQwen4ExpPLELayer(_config(), 0, prefix="layers.0.ple")
    emb = ple.ple_embedding
    mult, primes, offsets = _hash_params(emb)
    with torch.no_grad():
        for p in ple.parameters():
            if p.dtype.is_floating_point:
                p.copy_(torch.randn_like(p, dtype=torch.float32) * 0.1)
        emb.table.copy_(torch.randint(-2**31, 2**31 - 1, emb.table.shape, dtype=torch.int64).to(torch.int32))
        emb.load_weights([("layer_multipliers", mult), ("ngram_heads_vocab_sizes", primes),
                          ("ngram_heads_offsets", offsets), ("ngram_embedding.weight_scale", torch.tensor([0.01]))])
    ple = ple.to("hpu", torch.bfloat16)
    for name in ("table", "hash_terms", "head_primes", "head_offsets"):
        t = getattr(emb, name)
        t.data = t.data.to(torch.int32)

    ctx = SimpleNamespace(attn_metadata=None)
    monkeypatch.setattr(qwen4_exp, "get_forward_context", lambda: ctx)

    pad, real = 16, 9
    ids = torch.randint(0, VOCAB - 1, (1, pad))
    ids[0, 4] = EOS
    hidden = torch.randn(1, pad, ple.hc_hidden_size, dtype=torch.bfloat16)

    def prefill(fn, host, slot, lo, hi, has_initial):
        n = hi - lo
        h = torch.zeros(1, pad, ple.hc_hidden_size, dtype=torch.bfloat16)
        i = torch.zeros(1, pad, dtype=torch.int64)
        h[:, :n], i[:, :n] = hidden[:, lo:hi], ids[:, lo:hi]
        ctx.attn_metadata = _prefill_md(slot, n, has_initial)
        return fn(h.to("hpu"), i.to("hpu"), host)[0, :n].cpu()

    def decode(fn, host, slot, pos):
        ctx.attn_metadata = _decode_md(slot)
        return fn(hidden[:, pos:pos + 1].to("hpu"), ids[:, pos:pos + 1].to("hpu"), host)[0].cpu()

    with torch.no_grad():
        # Whole prompt in one eager prefill: the reference.
        host = _Host(ple)
        ref = prefill(ple, host, 1, 0, real, False)
        ref_next = decode(ple, host, 1, real)

        # Same with every decoder op compiled (HPU graph fusion must not change the result).
        compiled = torch.compile(ple, backend="hpu_backend", dynamic=False)
        host = _Host(ple)
        out = prefill(compiled, host, 2, 0, real, False)
        out_next = decode(compiled, host, 2, real)
        assert _rel(out, ref) < 2e-2
        assert _rel(out_next, ref_next) < 2e-2

        # Chunked prefill followed by token-by-token decode carries the conv
        # history and the n-gram context through the state slots.
        host = _Host(ple)
        host.ple_conv_state[3].normal_()  # stale state from a previous request
        parts = [prefill(ple, host, 3, 0, 3, False), prefill(ple, host, 3, 3, 6, True)]
        parts += [decode(ple, host, 3, pos) for pos in range(6, real)]
        out = torch.cat(parts)
        assert _rel(out, ref) < 2e-2
        assert _rel(decode(ple, host, 3, real), ref_next) < 2e-2
