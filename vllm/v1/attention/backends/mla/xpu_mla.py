# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense (non-sparse) MLA backend for Intel GPUs (XPU).

This backend wires DeepSeek/Kimi-style Multi-head Latent Attention (MLA)
decode onto the XPU FlashAttention v2 kernel
(``vllm_xpu_kernels.flash_attn_interface.flash_attn_varlen_func``, exposed via
``vllm._xpu_ops.xpu_ops`` and re-imported through ``fa_utils``).

MLA is expressed to the kernel purely through tensor shapes (there is no
``mla`` flag): the query is the concatenation ``[q_nope, q_pe]`` of width
``kv_lora_rank + qk_rope_head_dim`` (e.g. 512 + 64 = 576), K is the full latent
cache row of the same width, and V is a zero-copy ``narrow`` view of the same
buffer covering only the first ``kv_lora_rank`` channels (512). The kernel
reads ``head_size_qk`` and ``head_size_vo`` independently and honors V's
per-tensor strides, so the asymmetric (576 vs 512) attention runs without any
extra copies.

Prefill reuses the platform-agnostic MLA prefill machinery from
``MLACommonImpl`` (the ``FlashAttnPrefillBackend``, which is already
XPU-compatible); only decode (``forward_mqa``) is implemented here.
"""

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonDecodeMetadata,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
)
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


class XPUMLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    @staticmethod
    def get_name() -> str:
        return "XPU_MLA"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @staticmethod
    def get_builder_cls() -> type["XPUMLAMetadataBuilder"]:
        return XPUMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["XPUMLAImpl"]:
        return XPUMLAImpl


@dataclass
class XPUMLADecodeMetadata(MLACommonDecodeMetadata):
    query_start_loc: torch.Tensor
    max_query_len: int
    max_seq_len: int


@dataclass
class XPUMLAMetadata(MLACommonMetadata[XPUMLADecodeMetadata]):
    pass


class XPUMLAMetadataBuilder(MLACommonMetadataBuilder[XPUMLAMetadata]):
    # XPU does not support full CUDA-graph capture for MLA decode.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
            XPUMLAMetadata,
        )

    def _build_decode(
        self,
        block_table_tensor: torch.Tensor,
        seq_lens_device: torch.Tensor,
        max_seq_len: int,
        query_start_loc_cpu: torch.Tensor,
        query_start_loc_device: torch.Tensor,
        num_decode_tokens: int,
        dcp_tot_seq_lens_device: torch.Tensor | None,
    ) -> XPUMLADecodeMetadata:
        query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        max_query_len = int(query_lens_cpu.max().item())

        return XPUMLADecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens_device,
            dcp_tot_seq_lens=dcp_tot_seq_lens_device,
            query_start_loc=query_start_loc_device,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
        )


class XPUMLAImpl(MLACommonImpl[XPUMLAMetadata]):
    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        **mla_args,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **mla_args,
        )

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "XPUMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "XPUMLAImpl"
            )

        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "XPU MLA with FP8 KV cache is not yet supported"
            )

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: XPUMLAMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        # ``q`` arrives either as a (q_nope, q_pe) tuple of shapes
        # (B, N, kv_lora_rank) and (B, N, qk_rope_head_dim), or (in the DCP
        # path) already concatenated. Concatenate to the latent+rope query of
        # width ``kv_lora_rank + qk_rope_head_dim`` expected by the kernel.
        if type(q) is tuple:
            q = torch.cat(q, dim=-1)
        assert isinstance(q, torch.Tensor)

        # The paged latent cache row is [..., kv_lora_rank + qk_rope_head_dim].
        # K is the full row; V is a zero-copy view of the first
        # kv_lora_rank channels. Add a singleton KV-head dim (MLA is MQA-like
        # with num_kv_heads == 1) so the cache is 4D as the kernel expects.
        k_cache = kv_c_and_k_pe_cache.unsqueeze(-2)
        v_cache = k_cache[..., : self.kv_lora_rank]

        # NOTE: During CUDA graph capture max_query_len can be 0, but the
        # kernel uses it for grid sizing; clamp to at least 1.
        max_seqlen_q = max(attn_metadata.decode.max_query_len, 1)

        attn_out = flash_attn_varlen_func(
            q=q,
            k=k_cache,
            v=v_cache,
            cu_seqlens_q=attn_metadata.decode.query_start_loc,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=attn_metadata.decode.max_seq_len,
            seqused_k=attn_metadata.decode.seq_lens,
            block_table=attn_metadata.decode.block_table,
            softmax_scale=self.scale,
            # Single-query decode: seqused_k already bounds the valid KV, so
            # there is nothing in the "future" to mask.
            causal=False,
            return_softmax_lse=self.need_to_return_lse_for_decode,
            fa_version=2,  # XPU kernel only implements FA2
        )

        if self.need_to_return_lse_for_decode:
            o, lse = attn_out
            # The XPU FA2 kernel sizes its output to the query head dim
            # (kv_lora_rank + qk_rope_head_dim); keep only the kv_lora_rank
            # latent channels expected by the v up-projection.
            o = o[..., : self.kv_lora_rank]
            # FA returns LSE in shape [H, B] but DCP wants [B, H].
            return o, lse.transpose(0, 1)
        # The XPU FA2 kernel sizes its output to the query head dim
        # (kv_lora_rank + qk_rope_head_dim); keep only the kv_lora_rank
        # latent channels expected by the v up-projection.
        attn_out = attn_out[..., : self.kv_lora_rank]
        return attn_out, None
