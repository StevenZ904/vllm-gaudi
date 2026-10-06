# SPDX-License-Identifier: Apache-2.0
"""Gaudi implementation of Qwen4Exp (e.g. Qwen3.8-Flash-Next).

Qwen4Exp extends the Qwen3.5 hybrid GDN/attention stack with:

* hyper-connections (HC): the residual stream holds ``hc_count`` parallel
  copies of the hidden state that every block mixes and re-injects;
* a per-layer n-gram embedding (PLE) that hashes the last three tokens into
  a very large table and feeds the result through a dilated short conv;
* Qwen sparse attention (QSA) on the full-attention layers.

The upstream vLLM implementation relies on CUDA/ROCm kernels (CuTe DSL HC
GEMMs, UVA lookups into a host-resident table, int64 hashing, a dedicated
short-conv KV-cache spec).  This module re-implements the model with plain
torch ops that lower well on HPU:

* HC and PLE math is written as dense torch ops that the regional
  ``torch.compile`` of each decoder layer can fuse.
* The n-gram hash is evaluated in int32 arithmetic (HPU has no native int64)
  from per-token lookup tables and a limb-wise modular reduction.
* The n-gram table is sharded across TP ranks in device memory.  FP8
  checkpoints keep it as raw bytes dequantized through a 256-entry lookup
  table (the checkpoint uses the full OCP e4m3 range, which exceeds the
  Gaudi2 FP8 format); BF16 checkpoints keep it in the model dtype.
* The PLE recurrent state (last ``ngram_size - 1`` tokens and the dilated conv
  history) rides along with the GDN state slots of the linear-attention layer
  that hosts the PLE, so no additional KV-cache group is created.
"""

import os
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import MergedColumnParallelLinear, ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    MultiModalEmbeddings,
    SupportsMRoPE,
    SupportsPP,
    _require_is_multimodal,
)
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration, Qwen3_5Model
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention, Qwen3NextSparseMoeBlock
from vllm.model_executor.models.qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
    Qwen3VLProcessingInfo,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    StageMissingLayer,
    WeightsMapper,
    _merge_multimodal_embeddings,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.v1.kv_cache_interface import MambaSpec

from vllm_gaudi.models.qwen3_5 import HPUGatedDeltaNetAttention

logger = init_logger(__name__)

# Upstream Qwen4Exp layer-type names.  Kept local so that importing this
# module never pulls in the CUDA/ROCm specific upstream implementation.
_QSA_LAYER_TYPE = "qwen_sparse_attention"
_ATTENTION_LAYER_TYPES = ("full_attention", _QSA_LAYER_TYPE)

# Checkpoint tensors that only matter for upstream kernels (or for the MTP
# draft head, which is not supported on HPU yet).
_SKIPPED_WEIGHT_SUBSTRS = (
    "hashstats_",
    "token_lookup",
    "hyper_connection_mixer.block_inject_weight",
    # TODO: QSA indexer; full attention is exact while a sequence holds at
    # most ``indexer_budget`` tokens plus one partial compression group.
    "self_attn.indexer.",
)

_QWEN4_EXP_IGNORED_MISSING_SUFFIXES = [
    ".bias",
    "_bias",
    ".k_scale",
    "_k_scale",
    ".v_scale",
    "_v_scale",
    "_weight_scale",
    "_input_scale",
]

# The checkpoint stores these projections separately; they are packed into
# adjacent shards of one replicated GEMM at runtime.
_EXTRA_WEIGHTS_MAPPER = WeightsMapper(
    orig_to_new_stacked={
        "hyper_connection.input_mix_weight_down.weight": (
            "hyper_connection.input_mix_weight_down_block_inject.weight",
            0,
        ),
        "hyper_connection.block_inject_weight.weight": (
            "hyper_connection.input_mix_weight_down_block_inject.weight",
            1,
        ),
        "ple.key_proj": ("ple.kv_proj", 0),
        "ple.value_proj": ("ple.kv_proj", 1),
    })


class GroupedGemmaRMSNorm(nn.Module):
    """RMSNorm over contiguous groups of ``group_size`` with ``(1 + w)`` scale."""

    def __init__(self, hidden_size: int, eps: float, group_size: int, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        assert hidden_size % group_size == 0
        self.variance_epsilon = eps
        self.group_size = group_size
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        x = hidden_states.float().unflatten(-1, (-1, self.group_size))
        x = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.variance_epsilon)
        return (x.flatten(-2) * (1.0 + self.weight.float())).to(input_dtype)


class HpuGatedResidual(nn.Module):
    """Hyper-connection gated residual (``mix`` before, ``combine`` after a block).

    The residual stream is ``[..., hc_count * hidden_size]`` (stream-major).
    ``mix`` returns the single-stream block input together with the
    per-stream injection weights consumed by ``combine``.
    """

    def __init__(self, config, use_combine: bool = True, prefix: str = "") -> None:
        super().__init__()
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.hc_lowrank = config.hc_lowrank
        self.use_combine = use_combine
        hc_hidden_size = self.hc_count * self.hidden_size
        self.hc_norm = GroupedGemmaRMSNorm(hc_hidden_size, config.rms_norm_eps, self.hidden_size)
        if use_combine:
            # One GEMM for the low-rank mix projection and the injection logits.
            self.input_mix_weight_down_block_inject = MergedColumnParallelLinear(
                hc_hidden_size,
                [self.hc_lowrank, self.hc_count],
                bias=False,
                quant_config=None,
                return_bias=False,
                disable_tp=True,
                prefix=f"{prefix}.input_mix_weight_down_block_inject",
            )
        else:
            self.input_mix_weight_down = ReplicatedLinear(
                hc_hidden_size,
                self.hc_lowrank,
                bias=False,
                quant_config=None,
                return_bias=False,
                prefix=f"{prefix}.input_mix_weight_down",
            )
        self.input_mix_weight_up = ReplicatedLinear(
            self.hc_lowrank,
            hc_hidden_size,
            bias=False,
            quant_config=None,
            return_bias=False,
            prefix=f"{prefix}.input_mix_weight_up",
        )

    def mix(self, hyper_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        normed = self.hc_norm(hyper_input)
        injection = None
        if self.use_combine:
            down, inject_logits = self.input_mix_weight_down_block_inject(normed).split(
                [self.hc_lowrank, self.hc_count], dim=-1)
            injection = 2.0 * torch.sigmoid(inject_logits / self.hc_count)
        else:
            down = self.input_mix_weight_down(normed)
        gate = torch.sigmoid(self.input_mix_weight_up(F.silu(down / self.hc_count)))
        gate = gate.unflatten(-1, (self.hc_count, self.hidden_size))
        mixed = (gate * normed.unflatten(-1, (self.hc_count, self.hidden_size))).mean(dim=-2)
        return mixed, injection

    def combine(self, hyper_input: torch.Tensor, block_output: torch.Tensor, injection: torch.Tensor) -> torch.Tensor:
        residual = hyper_input.unflatten(-1, (self.hc_count, self.hidden_size))
        out = residual + block_output.unsqueeze(-2) * injection.unsqueeze(-1)
        return out.flatten(-2)


def _mod_u63(hi: torch.Tensor, lo: torch.Tensor, primes: torch.Tensor) -> torch.Tensor:
    """``(hi * 2**32 + lo) mod p`` in int32 arithmetic.

    ``hi`` holds the (non-negative) upper 32 bits of a 63-bit value, ``lo`` the
    bit pattern of the lower 32 bits.  ``primes`` is broadcast over a new last
    dimension.  Horner's rule over 2/6-bit limbs keeps every intermediate
    below ``64 * p + 63`` which fits int32 for ``p < 2**25``.
    """
    hi = hi.unsqueeze(-1)
    lo = lo.unsqueeze(-1)
    r = torch.zeros_like(hi) + torch.zeros_like(primes)
    for word in (hi, lo):
        r = torch.remainder(r * 4 + ((word >> 30) & 3), primes)
        for shift in (24, 18, 12, 6, 0):
            r = torch.remainder(r * 64 + ((word >> shift) & 63), primes)
    return r


@torch._dynamo.disable
def _save_ple_state(passthrough: torch.Tensor, conv_state: torch.Tensor, ctx_state: torch.Tensor,
                    state_indices: torch.Tensor, new_conv: torch.Tensor, new_ctx: torch.Tensor) -> torch.Tensor:
    """Persist the PLE conv history / token context for the next chunk.

    Runs outside the compiled region (HPU torch.compile may drop in-place
    updates of aliased state tensors).  Padding rows carry index -1 and land
    in the trailing garbage slot.  Returns ``passthrough`` so the compiled
    graph consumes the call.
    """
    safe_idx = torch.remainder(state_indices, conv_state.shape[0]).long()
    conv_state.index_copy_(0, safe_idx, new_conv.to(conv_state.dtype))
    ctx_state.index_copy_(0, safe_idx, new_ctx.to(ctx_state.dtype))
    return passthrough


class HpuQwen4ExpNGramEmbedding(nn.Module):
    """Hashed n-gram embedding with a TP-sharded table.

    ``table`` holds this rank's contiguous slice of the table rows: raw FP8
    bytes packed into int32 words when ``fp8_table`` is set, otherwise the
    rows themselves in the model dtype.  The hash multipliers, head sizes and
    offsets come from the checkpoint and are turned into int32 lookup tables
    once they are loaded.
    """

    def __init__(self,
                 config,
                 embedding_dim: int,
                 ple_dense_layer_id: int,
                 fp8_table: bool = True,
                 dtype: torch.dtype = torch.bfloat16,
                 prefix: str = "") -> None:
        super().__init__()
        self.prefix = prefix
        self.ngram_size = config.ngram_size
        if self.ngram_size != 3:
            # ``hash_ids`` builds the bigram and trigram hashes explicitly.
            raise NotImplementedError(f"Qwen4Exp n-gram embedding on HPU supports ngram_size=3, got {self.ngram_size}")
        self.heads_per_ngram = config.heads_per_ngram
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.head_dim = embedding_dim // self.ngram_heads
        assert self.head_dim * self.ngram_heads == embedding_dim
        assert self.head_dim % 4 == 0
        self.vocab_size = config.vocab_size
        eos = config.eos_token_id
        self.eos_token_id = int(eos[0] if isinstance(eos, (list, tuple)) else eos)

        # Table geometry: one prime-sized bucket range per head (heads of
        # later PLE layers continue the prime sequence), padded like the
        # checkpoint and split into ``split_ngram_parts`` row shards.
        prime = config.ngram_vocab_size_base - 1
        for _ in range(ple_dense_layer_id * self.ngram_heads):
            prime = _next_prime(prime)
        total = 0
        for _ in range(self.ngram_heads):
            prime = _next_prime(prime)
            total += prime
        div = config.make_ngram_vocab_size_divisible_by
        self.num_rows = (total + div - 1) // div * div
        self.split_parts = int(getattr(config, "split_ngram_parts", 512))
        self.checkpoint_shard_rows = (self.num_rows + self.split_parts - 1) // self.split_parts

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.rows_per_rank = (self.num_rows + self.tp_size - 1) // self.tp_size
        self.row_start = self.tp_rank * self.rows_per_rank
        self.fp8_table = fp8_table
        if fp8_table:
            table = torch.zeros(self.rows_per_rank, self.head_dim // 4, dtype=torch.int32)
        else:
            table = torch.zeros(self.rows_per_rank, self.head_dim, dtype=dtype)
        self.table = nn.Parameter(table, requires_grad=False)
        # FP8 byte -> bf16 value (scale folded in), filled by load_weights.
        self.register_buffer("dequant_lut", torch.zeros(256, dtype=torch.bfloat16), persistent=False)
        # Per-token hash terms: columns (hi, lo) of tok * multiplier[i].
        self.register_buffer("hash_terms",
                             torch.zeros(self.vocab_size, 2 * self.ngram_size, dtype=torch.int32),
                             persistent=False)
        self.register_buffer("head_primes", torch.ones(self.ngram_heads, dtype=torch.int32), persistent=False)
        self.register_buffer("head_offsets", torch.zeros(self.ngram_heads, dtype=torch.int32), persistent=False)
        self._hash_params: dict[str, torch.Tensor] = {}
        self._loaded_shards: set[int] = set()
        self._scale_loaded = False

    def hash_ids(self, history: torch.Tensor) -> torch.Tensor:
        """Map token history ``[B, S + 2]`` (int32) to table rows ``[B * S, heads]``."""
        eos = self.eos_token_id
        cur = history[:, 2:].reshape(-1)
        prev1 = history[:, 1:-1].reshape(-1)
        prev2 = history[:, :-2].reshape(-1)
        # The trigram context does not reach across a document boundary.
        prev2 = prev2 + (eos - prev2) * (prev1 == eos).to(prev2.dtype)
        t0 = self.hash_terms.index_select(0, cur)
        t1 = self.hash_terms.index_select(0, prev1)
        t2 = self.hash_terms.index_select(0, prev2)
        bi_hi = t0[:, 0] ^ t1[:, 2]
        bi_lo = t0[:, 1] ^ t1[:, 3]
        tri_hi = bi_hi ^ t2[:, 4]
        tri_lo = bi_lo ^ t2[:, 5]
        k = self.heads_per_ngram
        bigram = _mod_u63(bi_hi, bi_lo, self.head_primes[:k]) + self.head_offsets[:k]
        trigram = _mod_u63(tri_hi, tri_lo, self.head_primes[k:]) + self.head_offsets[k:]
        return torch.cat([bigram, trigram], dim=-1)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """Return the bf16 n-gram embeddings ``[B * S, ngram_heads * head_dim]``."""
        rows = self.hash_ids(history)
        num_tokens = rows.shape[0]
        local = rows - self.row_start
        valid = (local >= 0) & (local < self.rows_per_rank)
        local = local * valid.to(local.dtype)
        values = self.table.index_select(0, local.reshape(-1))
        if self.fp8_table:
            shifts = torch.arange(0, 32, 8, dtype=torch.int32, device=values.device)
            fp8_bytes = (values.unsqueeze(-1) >> shifts) & 255
            values = self.dequant_lut.index_select(0, fp8_bytes.reshape(-1))
        values = values.view(num_tokens, self.ngram_heads, self.head_dim)
        values = torch.where(valid.unsqueeze(-1), values, torch.zeros_like(values))
        values = values.reshape(num_tokens, -1)
        if self.tp_size > 1:
            values = tensor_model_parallel_all_reduce(values)
        return values

    # ---- weight loading -------------------------------------------------
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded: set[str] = set()
        for name, weight in weights:
            if name.startswith("ngram_embedding.shard_"):
                shard = int(name[len("ngram_embedding.shard_"):].split(".", 1)[0])
                self._load_table_shard(shard, weight)
                loaded.add("table")
            elif name == "ngram_embedding.weight_scale":
                assert self.fp8_table, f"{self.prefix}: weight_scale given for an unquantized PLE table"
                self._load_scale(weight)
                self._scale_loaded = True
            elif name in ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets"):
                self._hash_params[name] = weight.detach().to("cpu", torch.int64).reshape(-1)
                if len(self._hash_params) == 3:
                    self._build_hash_tables()
            elif name == "ngram_embedding.weight":
                # Unsharded checkpoint layout.
                if weight.shape[0] != self.num_rows:
                    raise ValueError(f"{self.prefix}: PLE table has {weight.shape[0]} rows, expected {self.num_rows}")
                self._load_rows(0, weight)
                self._loaded_shards.update(range(self.split_parts))
                loaded.add("table")
            else:
                raise ValueError(f"Unexpected PLE embedding weight {self.prefix}.{name}")
        return loaded

    def _load_table_shard(self, shard: int, weight: torch.Tensor) -> None:
        first_row = shard * self.checkpoint_shard_rows
        expected_rows = min(self.checkpoint_shard_rows, self.num_rows - first_row)
        if not 0 <= shard < self.split_parts or weight.shape[0] != expected_rows:
            raise ValueError(f"{self.prefix}: unexpected PLE table shard {shard} with shape {tuple(weight.shape)} "
                             f"(split_ngram_parts={self.split_parts}, {self.checkpoint_shard_rows} rows per shard)")
        self._load_rows(first_row, weight)
        self._loaded_shards.add(shard)

    def check_loaded(self) -> None:
        """Fail loudly instead of serving a partially zero table."""
        missing = sorted(set(range(self.split_parts)) - self._loaded_shards)
        if missing:
            raise ValueError(f"{self.prefix}: {len(missing)} PLE table shards missing from the checkpoint, "
                             f"e.g. shard_{missing[0]}")
        if len(self._hash_params) != 3:
            raise ValueError(f"{self.prefix}: PLE hash parameters missing from the checkpoint")
        if self.fp8_table and not self._scale_loaded:
            raise ValueError(f"{self.prefix}: PLE table weight_scale missing from the checkpoint")

    def _load_rows(self, first_row: int, weight: torch.Tensor) -> None:
        assert weight.dim() == 2 and weight.shape[1] == self.head_dim, weight.shape
        if self.fp8_table != (weight.element_size() == 1):
            raise ValueError(f"{self.prefix}: PLE table checkpoint dtype {weight.dtype} does not match the "
                             f"{'FP8' if self.fp8_table else 'unquantized'} quantization config")
        lo = max(first_row, self.row_start)
        hi = min(first_row + weight.shape[0], self.row_start + self.rows_per_rank)
        if lo >= hi:
            return
        rows = weight[lo - first_row:hi - first_row]
        if self.fp8_table:
            rows = rows.contiguous().view(torch.uint8).view(torch.int32)
        self.table.data[lo - self.row_start:hi - self.row_start].copy_(rows)

    def _load_scale(self, scale: torch.Tensor) -> None:
        codes = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn)
        lut = codes.float() * scale.detach().float().cpu().reshape(-1)[0]
        lut = torch.nan_to_num(lut, nan=0.0)
        self.dequant_lut.copy_(lut.to(self.dequant_lut.dtype))

    def _build_hash_tables(self) -> None:
        mult = self._hash_params["layer_multipliers"]
        sizes = self._hash_params["ngram_heads_vocab_sizes"]
        offsets = self._hash_params["ngram_heads_offsets"]
        assert mult.numel() == self.ngram_size and sizes.numel() == self.ngram_heads
        # Every per-token product (and their XOR) must stay a non-negative
        # 63-bit value, and the modular reduction needs 64 * p < 2**31.
        assert int(mult.max()) * (self.vocab_size - 1) < (1 << 63)
        assert int(sizes.max()) < (1 << 25)
        assert int(offsets.max() + sizes.max()) <= self.num_rows
        tokens = torch.arange(self.vocab_size, dtype=torch.int64)
        cols = []
        for m in mult.tolist():
            prod = tokens * m
            hi = prod >> 32
            lo = prod & 0xFFFFFFFF
            lo = torch.where(lo >= (1 << 31), lo - (1 << 32), lo)
            cols += [hi.to(torch.int32), lo.to(torch.int32)]
        self.hash_terms.copy_(torch.stack(cols, dim=-1))
        self.head_primes.copy_(sizes.to(torch.int32))
        self.head_offsets.copy_(offsets.to(torch.int32))


def _next_prime(value: int) -> int:

    def is_prime(n: int) -> bool:
        if n < 2:
            return False
        if n % 2 == 0:
            return n == 2
        d = 3
        while d * d <= n:
            if n % d == 0:
                return False
            d += 2
        return True

    value += 1
    while not is_prime(value):
        value += 1
    return value


class HpuQwen4ExpPLELayer(nn.Module):
    """Per-layer n-gram embedding injected into every HC stream.

    ``out = gated + silu(dilated_conv(norm_conv(gated)))`` with ``gated`` the
    n-gram value projection gated per stream by its similarity to the
    current hidden state.  The conv history and the previous tokens are kept
    in the host GDN layer's state slots (see ``HpuQwen4ExpPLEHostGDN``).
    """

    def __init__(self,
                 config,
                 ple_dense_layer_id: int,
                 fp8_table: bool = True,
                 dtype: torch.dtype = torch.bfloat16,
                 prefix: str = "") -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.hc_count = config.hc_count
        self.hc_hidden_size = self.hidden_size * self.hc_count
        self.context_len = config.ngram_size - 1
        self.conv_kernel_size = config.ple_conv_kernel_size
        self.conv_dilation = config.ngram_size
        self.conv_state_len = (self.conv_kernel_size - 1) * self.conv_dilation
        ple_embed_dim = config.ple_embed_dim
        self.ple_embedding = HpuQwen4ExpNGramEmbedding(config,
                                                       ple_embed_dim,
                                                       ple_dense_layer_id,
                                                       fp8_table=fp8_table,
                                                       dtype=dtype,
                                                       prefix=f"{prefix}.ple_embedding")
        self.eos_token_id = self.ple_embedding.eos_token_id
        self.kv_proj = MergedColumnParallelLinear(
            ple_embed_dim,
            [self.hc_hidden_size, self.hidden_size],
            bias=False,
            quant_config=None,
            return_bias=False,
            disable_tp=True,
            prefix=f"{prefix}.kv_proj",
        )
        self.norm_key = GroupedGemmaRMSNorm(self.hc_hidden_size, config.rms_norm_eps, self.hidden_size)
        self.norm_query = GroupedGemmaRMSNorm(self.hc_hidden_size, config.rms_norm_eps, self.hidden_size)
        self.norm_conv = GroupedGemmaRMSNorm(self.hc_hidden_size, config.rms_norm_eps, self.hidden_size)
        self.conv1d = nn.Conv1d(
            self.hc_hidden_size,
            self.hc_hidden_size,
            self.conv_kernel_size,
            groups=self.hc_hidden_size,
            dilation=self.conv_dilation,
            bias=False,
        )

    def _metadata(self, host: "HpuQwen4ExpPLEHostGDN", num_tokens: int):
        attn_metadata = get_forward_context().attn_metadata
        if attn_metadata is None or host.ple_conv_state is None:
            return None
        state_indices = host._resolve_state_indices(attn_metadata)
        if state_indices is None:
            return None
        is_prompt = bool(getattr(attn_metadata, "is_prompt", False))
        batch = state_indices.numel()
        if is_prompt:
            seq_len = num_tokens // batch
            qsl = attn_metadata.query_start_loc_p
            query_lens = (qsl[1:] - qsl[:-1]).to(torch.int32)
            if query_lens.numel() < batch:
                query_lens = F.pad(query_lens, (0, batch - query_lens.numel()))
            has_initial = attn_metadata.has_initial_states_p
            has_initial = has_initial.reshape(-1)[:batch].to(torch.int32)
            if has_initial.numel() < batch:
                has_initial = F.pad(has_initial, (0, batch - has_initial.numel()))
        else:
            seq_len = 1
            query_lens = None
            has_initial = None
        return is_prompt, state_indices, batch, seq_len, query_lens, has_initial

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor,
                host: "HpuQwen4ExpPLEHostGDN") -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, self.hc_hidden_size)
        num_tokens = hidden_states.shape[0]
        md = self._metadata(host, num_tokens)
        ids = input_ids.reshape(-1).to(torch.int32)
        if md is None:
            # Profiling / dummy run: no state, every sequence starts fresh.
            batch = input_ids.shape[0] if input_ids.dim() == 2 else 1
            seq_len = num_tokens // batch
            ids = ids.view(batch, seq_len)
            ctx = torch.full((batch, self.context_len), self.eos_token_id, dtype=torch.int32, device=ids.device)
            conv_init = torch.zeros(batch,
                                    self.conv_state_len,
                                    self.hc_hidden_size,
                                    dtype=hidden_states.dtype,
                                    device=hidden_states.device)
        else:
            is_prompt, state_indices, batch, seq_len, query_lens, has_initial = md
            ids = ids.view(batch, seq_len)
            safe_idx = torch.remainder(state_indices, host.ple_conv_state.shape[0])
            ctx = host.ple_ctx_state.index_select(0, safe_idx)
            conv_init = host.ple_conv_state.index_select(0, safe_idx)
            if is_prompt:
                keep = has_initial.view(-1, 1)
                ctx = ctx * keep + self.eos_token_id * (1 - keep)
                conv_init = conv_init * keep.view(-1, 1, 1).to(conv_init.dtype)

        history = torch.cat([ctx, ids], dim=1)
        embeddings = self.ple_embedding(history)

        key_value = self.kv_proj(embeddings.to(hidden_states.dtype))
        key, value = key_value.split([self.hc_hidden_size, self.hidden_size], dim=-1)
        key = self.norm_key(key).unflatten(-1, (self.hc_count, self.hidden_size))
        query = self.norm_query(hidden_states).unflatten(-1, (self.hc_count, self.hidden_size))
        gate = (key.float() * query.float()).sum(dim=-1, keepdim=True) * (self.hidden_size**-0.5)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        # Flatten before the downcast: HPU graph fusion of a 3-D fp32 -> bf16
        # cast followed by a reshape produces garbage here.
        gated = (torch.sigmoid(gate) * value.unsqueeze(-2).float()).flatten(-2).to(hidden_states.dtype)
        normed = self.norm_conv(gated)

        # Dilated depthwise causal conv over [conv history | current chunk].
        normed = normed.view(batch, seq_len, self.hc_hidden_size)
        padded = torch.cat([conv_init.to(normed.dtype), normed], dim=1)
        weight = self.conv1d.weight.view(self.hc_hidden_size, self.conv_kernel_size).float()
        conv = padded[:, 0:seq_len].float() * weight[:, 0]
        for k in range(1, self.conv_kernel_size):
            start = k * self.conv_dilation
            conv = conv + padded[:, start:start + seq_len].float() * weight[:, k]
        conv = conv.view(num_tokens, self.hc_hidden_size)
        conv = F.silu(conv.to(gated.dtype).float()).to(gated.dtype)
        out = gated + conv

        if md is not None:
            if is_prompt:
                # State after the last real token of each (right-padded) row.
                offs = query_lens.view(-1, 1) + torch.arange(self.conv_state_len, dtype=torch.int32,
                                                             device=out.device).view(1, -1)
                new_conv = torch.gather(padded, 1, offs.long().unsqueeze(-1).expand(-1, -1, self.hc_hidden_size))
                ctx_offs = query_lens.view(-1, 1) + torch.arange(self.context_len, dtype=torch.int32,
                                                                 device=out.device).view(1, -1)
                new_ctx = torch.gather(history, 1, ctx_offs.long())
            else:
                new_conv = padded[:, seq_len:]
                new_ctx = history[:, seq_len:]
            out = _save_ple_state(out, host.ple_conv_state, host.ple_ctx_state, state_indices, new_conv, new_ctx)
        return out.view(orig_shape)


class HpuQwen4ExpPLEHostGDN(HPUGatedDeltaNetAttention):
    """GDN layer that also owns the PLE recurrent state of its decoder layer.

    The PLE state is indexed by the same per-request slots as the GDN state,
    so it is allocated with the same number of slots whenever the runner
    binds the GDN cache.
    """

    def __init__(self, *args, ple_state_shape: tuple[int, int], ple_context_len: int, eos_token_id: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.ple_state_shape = ple_state_shape
        self.ple_context_len = ple_context_len
        self.ple_eos_token_id = eos_token_id
        self.ple_conv_state: torch.Tensor | None = None
        self.ple_ctx_state: torch.Tensor | None = None

    def bind_kv_cache(self, kv_cache) -> None:
        super().bind_kv_cache(kv_cache)
        ref = self.kv_cache[0]
        num_slots = ref.shape[0]
        if (self.ple_conv_state is None or self.ple_conv_state.shape[0] != num_slots
                or self.ple_conv_state.device != ref.device):
            self.ple_conv_state = torch.zeros(num_slots,
                                              *self.ple_state_shape,
                                              dtype=self.model_config.dtype,
                                              device=ref.device)
            self.ple_ctx_state = torch.full((num_slots, self.ple_context_len),
                                            self.ple_eos_token_id,
                                            dtype=torch.int32,
                                            device=ref.device)


class HpuQwen4ExpSparseMoeBlock(Qwen3NextSparseMoeBlock):

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        if vllm_config.parallel_config.use_sequence_parallel_moe:
            raise NotImplementedError("Qwen4Exp hyper-connections do not support sequence-parallel MoE")
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_text_config
        self.n_shared_experts = int(config.shared_expert_intermediate_size > 0)


class HpuQwen4ExpDecoderLayer(nn.Module):

    def __init__(self, vllm_config: VllmConfig, layer_type: str, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        model_config = vllm_config.model_config
        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)

        self.ple: HpuQwen4ExpPLELayer | None = None
        ple_layer_ids = list(config.ple_layer_ids or ())
        if (self.layer_idx + 1) in ple_layer_ids:
            if layer_type != "linear_attention":
                raise NotImplementedError("Qwen4Exp PLE on HPU requires the PLE layer to be a GDN layer")
            # FP8 checkpoints store the table as FP8 with a per-tensor scale;
            # BF16 checkpoints (also when quantized online) keep it unquantized.
            fp8_table = bool(getattr(vllm_config.quant_config, "is_checkpoint_fp8_serialized", False))
            self.ple = HpuQwen4ExpPLELayer(config,
                                           ple_dense_layer_id=ple_layer_ids.index(self.layer_idx + 1),
                                           fp8_table=fp8_table,
                                           dtype=model_config.dtype,
                                           prefix=f"{prefix}.ple")

        if layer_type == "linear_attention":
            if self.ple is not None:
                self.linear_attn = HpuQwen4ExpPLEHostGDN(
                    config,
                    vllm_config=vllm_config,
                    prefix=f"{prefix}.linear_attn",
                    gqa_interleaved_layout=False,
                    ple_state_shape=(self.ple.conv_state_len, self.ple.hc_hidden_size),
                    ple_context_len=self.ple.context_len,
                    eos_token_id=self.ple.eos_token_id,
                )
            else:
                self.linear_attn = HPUGatedDeltaNetAttention(
                    config,
                    vllm_config=vllm_config,
                    prefix=f"{prefix}.linear_attn",
                    gqa_interleaved_layout=False,
                )
        elif layer_type in _ATTENTION_LAYER_TYPES:
            self.self_attn = Qwen3NextAttention(
                config,
                model_config=model_config,
                cache_config=vllm_config.cache_config,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            raise ValueError(f"Invalid layer_type {layer_type}")

        decoder_sparse_step = getattr(config, "decoder_sparse_step", 1) or 1
        if (self.layer_idx in (getattr(config, "mlp_only_layers", None) or ())
                or (self.layer_idx + 1) % decoder_sparse_step != 0 or not getattr(config, "num_experts", 0)):
            raise NotImplementedError("Qwen4Exp on HPU supports MoE MLPs only; dense MLP layers are not implemented")
        self.mlp = HpuQwen4ExpSparseMoeBlock(vllm_config=vllm_config, prefix=f"{prefix}.mlp")
        self.attn_hyper_connection = HpuGatedResidual(config, prefix=f"{prefix}.attn_hyper_connection")
        self.mlp_hyper_connection = HpuGatedResidual(config, prefix=f"{prefix}.mlp_hyper_connection")

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        if self.ple is not None:
            hidden_states = hidden_states + self.ple(hidden_states, input_ids, self.linear_attn)

        block_input, injection = self.attn_hyper_connection.mix(hidden_states)
        if self.layer_type == "linear_attention":
            block_output = self.linear_attn(hidden_states=block_input)
        else:
            block_output = self.self_attn(positions=positions, hidden_states=block_input)
        hidden_states = self.attn_hyper_connection.combine(hidden_states, block_output, injection)

        block_input, injection = self.mlp_hyper_connection.mix(hidden_states)
        block_output = self.mlp(block_input)
        return self.mlp_hyper_connection.combine(hidden_states, block_output, injection)


class HpuQwen4ExpModel(nn.Module):
    hf_to_vllm_mapper = Qwen3_5Model.hf_to_vllm_mapper | _EXTRA_WEIGHTS_MAPPER

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.config = config
        self.vocab_size = config.vocab_size
        self.hc_count = config.hc_count
        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, config.hidden_size)
        if _QSA_LAYER_TYPE in config.layer_types:
            logger.warning_once(
                "Qwen4Exp sparse attention (QSA) is not implemented on HPU yet; its layers run dense attention, "
                "which matches QSA only while a sequence holds at most indexer_budget=%d tokens.",
                getattr(config, "indexer_budget", 0))

        def get_layer(prefix: str) -> HpuQwen4ExpDecoderLayer:
            layer_idx = extract_layer_index(prefix)
            return HpuQwen4ExpDecoderLayer(vllm_config, layer_type=config.layer_types[layer_idx], prefix=prefix)

        self.start_layer, self.end_layer, self.layers = make_layers(config.num_hidden_layers,
                                                                    get_layer,
                                                                    prefix=f"{prefix}.layers")
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size * config.hc_count)
        if get_pp_group().is_last_rank:
            self.hyper_connection_mixer = HpuGatedResidual(config,
                                                           use_combine=False,
                                                           prefix=maybe_prefix(prefix, "hyper_connection_mixer"))
        else:
            self.hyper_connection_mixer = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        deepstack_input_embeds: IntermediateTensors | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if input_ids is None:
            raise ValueError("Qwen4Exp needs the raw input_ids for its n-gram embedding")
        if get_pp_group().is_first_rank:
            hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
            hidden_states = hidden_states.repeat(*([1] * (hidden_states.dim() - 1)), self.hc_count)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        for layer_idx in range(self.start_layer, self.end_layer):
            hidden_states = self.layers[layer_idx](hidden_states, positions, input_ids)
            if deepstack_input_embeds is not None and layer_idx < len(deepstack_input_embeds):
                deepstack = deepstack_input_embeds[f"deepstack_input_embeds_{layer_idx}"]
                deepstack = deepstack.view(hidden_states.shape[:-1] + (deepstack.shape[-1], ))
                hidden_states = hidden_states + deepstack.repeat(*([1] * (deepstack.dim() - 1)), self.hc_count)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})
        hidden_states, _ = self.hyper_connection_mixer.mix(hidden_states)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        mapper = self.hf_to_vllm_mapper | WeightsMapper(
            orig_to_new_substr={substr: None
                                for substr in _SKIPPED_WEIGHT_SUBSTRS})
        ignore_prefixes = None if self.hyper_connection_mixer is not None else ["hyper_connection_mixer."]
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_prefixes=ignore_prefixes,
            ignore_unexpected_suffixes=_QWEN4_EXP_IGNORED_MISSING_SUFFIXES.copy(),
        )
        return loader.load_weights(weights, mapper=mapper)


def _check_ple_tables_loaded(model: nn.Module) -> None:
    # Called once the whole checkpoint has been consumed: AutoWeightsLoader
    # may call a submodule's load_weights several times, once per run of
    # consecutive checkpoint weights under its prefix.
    for module in model.modules():
        if isinstance(module, HpuQwen4ExpNGramEmbedding):
            module.check_loaded()


def _check_prefix_caching(vllm_config: VllmConfig) -> None:
    cache_config = vllm_config.cache_config
    if cache_config.mamba_cache_mode == "all":
        raise NotImplementedError("Qwen4Exp does not support 'all' mamba prefix caching")
    # Compact GDN (the default) does not cache recurrent state.  Without it,
    # cached GDN state would be copied between blocks but the PLE state that
    # rides along with it would not.
    compact_gdn = os.environ.get("VLLM_COMPACT_GDN", "1").strip().lower() in ("1", "true")
    if cache_config.enable_prefix_caching and not compact_gdn:
        raise NotImplementedError("Qwen4Exp does not support prefix caching with VLLM_COMPACT_GDN=0; "
                                  "pass --no-enable-prefix-caching")


class _HpuQwen4ExpMambaSpecMixin:
    """Mamba state geometry of the hybrid model.

    Only the GDN state is exposed to the KV-cache manager; the PLE state is
    allocated alongside the GDN slots of its host layer.  The
    ConditionalGeneration head inherits the same GDN geometry from
    ``Qwen3_5ForConditionalGeneration`` and takes only
    ``get_mamba_specs_from_config`` from here.
    """

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig) -> tuple[tuple[int, int], tuple[int, int, int]]:
        hf_config = vllm_config.model_config.hf_text_config
        num_spec = (vllm_config.speculative_config.num_speculative_tokens if vllm_config.speculative_config else 0)
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            vllm_config.parallel_config.tensor_parallel_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls):
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    @classmethod
    def get_mamba_specs_from_config(cls, vllm_config: VllmConfig) -> tuple[MambaSpec, ...]:
        return (MambaSpec(
            shapes=cls.get_mamba_state_shape_from_config(vllm_config),
            dtypes=cls.get_mamba_state_dtype_from_config(vllm_config),
            block_size=-1,
        ), )


class HpuQwen4ExpForCausalLM(nn.Module, HasInnerState, SupportsMRoPE, SupportsPP, IsHybrid, _HpuQwen4ExpMambaSpecMixin):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "kv_proj": ["key_proj", "value_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
        "input_mix_weight_down_block_inject": ["input_mix_weight_down", "block_inject_weight"],
    }
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"model.language_model.": "model."})
    requires_raw_input_tokens = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.config = config
        _check_prefix_caching(vllm_config)
        if vllm_config.parallel_config.pipeline_parallel_size > 1 and config.ple_layer_ids:
            raise NotImplementedError("Qwen4Exp n-gram embedding requires pipeline_parallel_size=1")
        self.model = HpuQwen4ExpModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, prefix=maybe_prefix(prefix, "lm_head"))
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors
        # Cleared when nested in the ConditionalGeneration head, which checks
        # after loading the whole checkpoint.
        self.check_ple_tables_on_load = True

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def get_mrope_input_positions(self, input_tokens: list[int], mm_features) -> tuple[torch.Tensor, int]:
        positions = torch.arange(len(input_tokens), dtype=torch.long)
        return positions.unsqueeze(0).expand(3, -1), 0

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        mapper = self.hf_to_vllm_mapper | WeightsMapper(orig_to_new_prefix={"mtp.": None})
        loader = AutoWeightsLoader(self, ignore_unexpected_suffixes=_QWEN4_EXP_IGNORED_MISSING_SUFFIXES.copy())
        loaded = loader.load_weights(weights, mapper=mapper)
        if self.check_ple_tables_on_load:
            _check_ple_tables_loaded(self)
        return loaded


class HpuQwen4ExpProcessingInfo(Qwen3VLProcessingInfo):

    def get_hf_config(self):
        from vllm.models.qwen4_exp.config import Qwen4ExpConfig
        return self.ctx.get_hf_config(Qwen4ExpConfig)


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=HpuQwen4ExpProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class HpuQwen4ExpForConditionalGeneration(Qwen3_5ForConditionalGeneration, HasInnerState, _HpuQwen4ExpMambaSpecMixin):
    """Qwen3-VL vision tower in front of the Qwen4Exp language model."""

    requires_raw_input_tokens = True
    packed_modules_mapping = Qwen3_5ForConditionalGeneration.packed_modules_mapping | {
        "kv_proj": ["key_proj", "value_proj"],
        "input_mix_weight_down_block_inject": ["input_mix_weight_down", "block_inject_weight"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model") -> None:
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        if multimodal_config is None:
            raise ValueError("Qwen4ExpForConditionalGeneration requires multimodal_config")
        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.language_model_only = multimodal_config.language_model_only
        if self.language_model_only:
            self.use_data_parallel = False
            self.is_multimodal_pruning_enabled = False
            self.video_pruning_method = None
            self.video_pruning_rate = 0.0
            self._tokenizer = None
            self.visual = StageMissingLayer("vision_tower")
            self._tower_model_names = []
        else:
            from vllm.model_executor.layers.fusion.mm_input_norm import build_mm_input_norm
            from vllm.tokenizers.registry import cached_tokenizer_from_config
            self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
            self._init_video_pruning(multimodal_config)
            self._tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
            with self._mark_tower_model(vllm_config, {"image", "video"}):
                self.visual = Qwen3_VisionTransformer(
                    config.vision_config,
                    norm_eps=config.text_config.rms_norm_eps,
                    quant_config=vllm_config.quant_config,
                    input_norm=build_mm_input_norm(self.model_config),
                    prefix=maybe_prefix(prefix, "visual"),
                )

        self.use_deepstack = (not self.language_model_only and bool(config.vision_config.deepstack_visual_indexes))
        self.deepstack_num_level = (len(config.vision_config.deepstack_visual_indexes) if self.use_deepstack else 0)
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level
        if self.use_deepstack:
            self.deepstack_input_embeds = [
                torch.zeros(vllm_config.scheduler_config.max_num_batched_tokens, config.text_config.hidden_size)
                for _ in range(self.deepstack_num_level)
            ]
            self.deepstack_input_embeds_num_tokens = 0

        with self._mark_language_model(vllm_config):
            self.language_model = HpuQwen4ExpForCausalLM(vllm_config=vllm_config,
                                                         prefix=maybe_prefix(prefix, "language_model"))
        self.language_model.check_ple_tables_on_load = False
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.language_model.embed_input_ids,
            is_multimodal=is_multimodal,
        )
        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds
        if self.language_model_only:
            raise ValueError("Qwen4Exp language_model_only does not accept multimodal embeddings")
        is_multimodal = _require_is_multimodal(is_multimodal)
        deepstack_input_embeds = None
        if self.use_deepstack:
            deepstack_input_embeds, multimodal_embeddings = self._compute_deepstack_embeds(
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=multimodal_embeddings,
                is_multimodal=is_multimodal,
            )
        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )
        if deepstack_input_embeds is not None:
            self._set_deepstack_input_embeds(deepstack_input_embeds)
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        deepstack_input_embeds = None
        if self.use_deepstack and inputs_embeds is not None and get_pp_group().is_first_rank:
            deepstack_input_embeds = self._get_deepstack_input_embeds(
                inputs_embeds.reshape(-1, inputs_embeds.shape[-1]).size(0))
        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            deepstack_input_embeds=deepstack_input_embeds,
        )
        if deepstack_input_embeds is not None:
            self._clear_deepstack_input_embeds(inputs_embeds.reshape(-1, inputs_embeds.shape[-1]).size(0))
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        mapper = self.hf_to_vllm_mapper | WeightsMapper(
            orig_to_new_substr={"mtp.": None},
            orig_to_new_prefix={"visual.": None} if self.language_model_only else {},
        )
        loader = AutoWeightsLoader(self, ignore_unexpected_suffixes=_QWEN4_EXP_IGNORED_MISSING_SUFFIXES.copy())
        loaded = loader.load_weights(weights, mapper=mapper)
        _check_ple_tables_loaded(self)
        return loaded

    @classmethod
    def get_mamba_state_copy_funcs(cls, mamba_types):
        return {mamba_type: cls.get_mamba_state_copy_func() for mamba_type in mamba_types}
