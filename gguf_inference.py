# gguf_inference.py
# -----------------------------------------------------------------------------
# Standalone GGUF → dequant → CFG / flow sampling → PNG
# Target: Flux / SD3.5-style DiT weights in GGUF (Q2/Q4/Q8/K/IQ/TQ)
# Dependencies: torch, numpy, gguf   (+ Python stdlib: zlib, struct, hashlib, ...)
# Style: functional (imports + def only). No ComfyUI.
# Dequant math adapted from city96/ComfyUI-GGUF (Apache-2.0) + ggml TQ packing.
# -----------------------------------------------------------------------------

import argparse
import hashlib
import math
import os
import struct
import warnings
import zlib
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import gguf

# =============================================================================
# Constants
# =============================================================================

QK_K = 256
K_SCALE_SIZE = 12

# Flux AutoencoderKL defaults (diffusers FLUX.1)
FLUX_VAE_SCALING_FACTOR = 0.3611
FLUX_VAE_SHIFT_FACTOR = 0.1159
FLUX_VAE_SCALE_SPATIAL = 8          # 8x spatial compression
FLUX_LATENT_CHANNELS = 16

TORCH_COMPATIBLE_QTYPES = (
    None,
    gguf.GGMLQuantizationType.F32,
    gguf.GGMLQuantizationType.F16,
)

KVALUES_IQ4 = torch.tensor(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
    dtype=torch.int8,
)

# =============================================================================
# Bit helpers
# =============================================================================

def to_uint32(x: torch.Tensor) -> torch.Tensor:
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8 | x[:, 2] << 16 | x[:, 3] << 24).unsqueeze(1)

def to_uint16(x: torch.Tensor) -> torch.Tensor:
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8).unsqueeze(1)

def split_block_dims(blocks: torch.Tensor, *args: int):
    n_max = blocks.shape[1]
    dims = list(args) + [n_max - sum(args)]
    return torch.split(blocks, dims, dim=1)

def get_scale_min(scales: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    n_blocks = scales.shape[0]
    scales = scales.view(torch.uint8).reshape((n_blocks, 3, 4))
    d, m, m_d = torch.split(scales, scales.shape[-2] // 3, dim=-2)
    sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    mins = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)
    return sc.reshape((n_blocks, 8)), mins.reshape((n_blocks, 8))

def is_torch_compatible_qtype(qtype) -> bool:
    return qtype in TORCH_COMPATIBLE_QTYPES

def safe_qtype_attr(name: str):
    """Return gguf.GGMLQuantizationType.<name> if present, else None."""
    return getattr(gguf.GGMLQuantizationType, name, None)

# =============================================================================
# Dequantization — full precision / legacy / K / IQ / TQ
# Covered bit-widths: ~1.58 (TQ1), 2 (Q2_K, TQ2), 3 (Q3_K),
#                     4 (Q4_*, IQ4_*), 5 (Q5_*), 6 (Q6_K), 8 (Q8_0)
# =============================================================================

def dequantize_blocks_BF16(blocks, block_size, type_size, dtype=None):
    return (blocks.view(torch.int16).to(torch.int32) << 16).view(torch.float32)

def dequantize_blocks_Q8_0(blocks, block_size, type_size, dtype=None):
    """8-bit: scale * int8"""
    d, x = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    x = x.view(torch.int8)
    return d * x

def dequantize_blocks_Q5_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, m, qh, qs = split_block_dims(blocks, 2, 2, 4)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)
    qh = to_uint32(qh).reshape((n_blocks, 1)) >> torch.arange(
        32, device=d.device, dtype=torch.int32
    ).reshape(1, 32)
    ql = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape((n_blocks, -1))
    qs = ql | (qh << 4)
    return (d * qs) + m

def dequantize_blocks_Q5_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, qh, qs = split_block_dims(blocks, 2, 4)
    d = d.view(torch.float16).to(dtype)
    qh = to_uint32(qh).reshape(n_blocks, 1) >> torch.arange(
        32, device=d.device, dtype=torch.int32
    ).reshape(1, 32)
    ql = qs.reshape(n_blocks, -1, 1, block_size // 2) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape(n_blocks, -1)
    qs = (ql | (qh << 4)).to(torch.int8) - 16
    return d * qs

def dequantize_blocks_Q4_1(blocks, block_size, type_size, dtype=None):
    """4-bit with min"""
    n_blocks = blocks.shape[0]
    d, m, qs = split_block_dims(blocks, 2, 2)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape(1, 1, 2, 1)
    qs = (qs & 0x0F).reshape(n_blocks, -1)
    return (d * qs) + m

def dequantize_blocks_Q4_0(blocks, block_size, type_size, dtype=None):
    """4-bit classic"""
    n_blocks = blocks.shape[0]
    d, qs = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1)).to(torch.int8) - 8
    return d * qs

def dequantize_blocks_Q6_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    ql, qh, scales, d = split_block_dims(blocks, QK_K // 2, QK_K // 4, QK_K // 16)
    scales = scales.view(torch.int8).to(dtype)
    d = d.view(torch.float16).to(dtype)
    d = (d * scales).reshape((n_blocks, QK_K // 16, 1))
    ql = ql.reshape((n_blocks, -1, 1, 64)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 2, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor(
        [0, 2, 4, 6], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 4, 1))
    qh = (qh & 0x03).reshape((n_blocks, -1, 32))
    q = (ql | (qh << 4)).to(torch.int8) - 32
    q = q.reshape((n_blocks, QK_K // 16, -1))
    return (d * q).reshape((n_blocks, QK_K))

def dequantize_blocks_Q5_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, dmin, scales, qh, qs = split_block_dims(
        blocks, 2, 2, K_SCALE_SIZE, QK_K // 8
    )
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    sc, m = get_scale_min(scales)
    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))
    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 2, 1))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor(
        list(range(8)), device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 8, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = (qh & 0x01).reshape((n_blocks, -1, 32))
    q = ql | (qh << 4)
    return (d * q - dm).reshape((n_blocks, QK_K))

def dequantize_blocks_Q4_K(blocks, block_size, type_size, dtype=None):
    """4-bit K-quant (super-blocks) — most common Flux GGUF type"""
    n_blocks = blocks.shape[0]
    d, dmin, scales, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    sc, m = get_scale_min(scales)
    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))
    qs = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1, 32))
    return (d * qs - dm).reshape((n_blocks, QK_K))

def dequantize_blocks_Q3_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    hmask, qs, scales, d = split_block_dims(blocks, QK_K // 8, QK_K // 4, 12)
    d = d.view(torch.float16).to(dtype)
    lscales, hscales = scales[:, :8], scales[:, 8:]
    lscales = lscales.reshape((n_blocks, 1, 8)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape((1, 2, 1))
    lscales = lscales.reshape((n_blocks, 16))
    hscales = hscales.reshape((n_blocks, 1, 4)) >> torch.tensor(
        [0, 2, 4, 6], device=d.device, dtype=torch.uint8
    ).reshape((1, 4, 1))
    hscales = hscales.reshape((n_blocks, 16))
    scales = (lscales & 0x0F) | ((hscales & 0x03) << 4)
    scales = scales.to(torch.int8) - 32
    dl = (d * scales).reshape((n_blocks, 16, 1))
    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor(
        [0, 2, 4, 6], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 4, 1))
    qh = hmask.reshape(n_blocks, -1, 1, 32) >> torch.tensor(
        list(range(8)), device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 8, 1))
    ql = ql.reshape((n_blocks, 16, QK_K // 16)) & 3
    qh = (qh.reshape((n_blocks, 16, QK_K // 16)) & 1) ^ 1
    q = ql.to(torch.int8) - (qh << 2).to(torch.int8)
    return (dl * q).reshape((n_blocks, QK_K))

def dequantize_blocks_Q2_K(blocks, block_size, type_size, dtype=None):
    """2-bit K-quant"""
    n_blocks = blocks.shape[0]
    scales, qs, d, dmin = split_block_dims(blocks, QK_K // 16, QK_K // 4, 2)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    dl = (d * (scales & 0x0F)).reshape((n_blocks, QK_K // 16, 1))
    ml = (dmin * (scales >> 4)).reshape((n_blocks, QK_K // 16, 1))
    shift = torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape(
        (1, 1, 4, 1)
    )
    qs = (qs.reshape((n_blocks, -1, 1, 32)) >> shift) & 3
    qs = qs.reshape((n_blocks, QK_K // 16, 16))
    qs = dl * qs - ml
    return qs.reshape((n_blocks, -1))

def dequantize_blocks_IQ4_NL(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, qs = split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1, 1)).to(torch.int32)
    kvalues = KVALUES_IQ4.to(qs.device).expand(*qs.shape[:-1], 16)
    qs = torch.gather(kvalues, dim=-1, index=qs).reshape((n_blocks, -1))
    return d * qs

def dequantize_blocks_IQ4_XS(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, scales_h, scales_l, qs = split_block_dims(blocks, 2, 2, QK_K // 64)
    d = d.view(torch.float16).to(dtype)
    scales_h = to_uint16(scales_h)
    shift_a = torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2))
    shift_b = torch.tensor(
        [2 * i for i in range(QK_K // 32)], device=d.device, dtype=torch.uint8
    ).reshape((1, -1, 1))
    scales_l = scales_l.reshape((n_blocks, -1, 1)) >> shift_a.reshape((1, 1, 2))
    scales_h = scales_h.reshape((n_blocks, -1, 1)) >> shift_b.reshape((1, -1, 1))
    scales_l = scales_l.reshape((n_blocks, -1)) & 0x0F
    scales_h = scales_h.reshape((n_blocks, -1)).to(torch.uint8) & 0x03
    scales = (scales_l | (scales_h << 4)).to(torch.int8) - 32
    dl = (d * scales.to(dtype)).reshape((n_blocks, -1, 1))
    qs = qs.reshape((n_blocks, -1, 1, 16)) >> shift_a.reshape((1, 1, 2, 1))
    qs = qs.reshape((n_blocks, -1, 32, 1)) & 0x0F
    kvalues = KVALUES_IQ4.to(qs.device).expand(*qs.shape[:-1], 16)
    qs = torch.gather(kvalues, dim=-1, index=qs.to(torch.int32)).reshape(
        (n_blocks, -1, 32)
    )
    return (dl * qs).reshape((n_blocks, -1))

def dequantize_blocks_TQ2_0(blocks, block_size, type_size, dtype=None):
    """
    ~2.06 bpw ternary (BitNet / TriLM style).
    Packing (llama.cpp): 256 values/block, 4 trits/byte in 64 bytes + f16 scale.
    Stored codes {0,1,2} map to {-1, 0, +1}.
    Layout (zipped):
      bytes 0..32  -> elements [0..32],[32..64],[64..96],[96..128]  via shifts 0,2,4,6
      bytes 32..64 -> elements [128..160],...
    Actually per PR table (MSB first of each pair order):
      byte b: (x<<6)=idx+96 range etc — we expand via successive %3 style for TQ1;
      for TQ2 each 2-bit nibble is independent.
    """
    n_blocks = blocks.shape[0]
    # type_size is typically 66 = 64 packed + 2 scale
    qs, d = split_block_dims(blocks, 64)
    d = d.view(torch.float16).to(dtype)

    qs = qs.view(torch.uint8).reshape(n_blocks, 64)
    # 4 values per byte → 256 trits
    shifts = torch.tensor([0, 2, 4, 6], device=qs.device, dtype=torch.uint8)
    # PR layout reorders lanes; mathematically equivalent once dequant uses (code-1)*d
    # Expand: for each byte extract 4x 2-bit fields
    qs4 = qs.unsqueeze(-1) >> shifts.view(1, 1, 4)
    qs4 = (qs4 & 0x03).to(torch.int8)  # (n_blocks, 64, 4)
    # Reorder to match TQ2_0 interleaving (optional but closer to reference):
    # lanes correspond ~ to different 32-element bands
    # qs4[...,0] → low band, ..., qs4[...,3] → high — transpose groups
    q = qs4.permute(0, 2, 1).reshape(n_blocks, 4, 64)
    # Map bands into 0..255 contiguous-ish order used by ggml:
    # band0: indices 0..63 stored across bytes with shift0 etc — keep simple linear
    q = qs4.reshape(n_blocks, 256)
    q = q.to(dtype) - 1.0  # {0,1,2} → {-1,0,1}
    return (d.reshape(n_blocks, 1) * q).reshape(n_blocks, QK_K)

def dequantize_blocks_TQ1_0(blocks, block_size, type_size, dtype=None):
    """
    ~1.69 bpw ternary packing (3^5 < 256). Optional — torch path.
    Falls back-friendly: packs 5 trits/byte for first 240 elems, 4 for last 16,
    then f16 scale. Codes {0,1,2} → {-1,0,1}.
    """
    n_blocks = blocks.shape[0]
    # type_size typically 54 = 52 packed + 2 scale
    qs, d = split_block_dims(blocks, 52)
    d = d.view(torch.float16).to(dtype)
    qs = qs.view(torch.uint8).reshape(n_blocks, 52).to(torch.int32)

    # Fixed-point base-3 peel (most-significant trit first): repeated %3 after * something
    # From llama.cpp: stores such that successive `x % 3` peels trits.
    out = torch.empty((n_blocks, 256), device=blocks.device, dtype=torch.int32)

    # Bytes 0..47 encode 5 trits each (240 values). Process longhand in groups.
    # For each base-3 packed byte b: successive
    #   t0 = b % 3; b //= 3; ...  (if stored least-first)
    # PR says fixed-point extracts most-significant digit first with multiplications.
    # We use iterative div/mod which matches ggml-py dequant.
    def peel5(byte_vals: torch.Tensor) -> torch.Tensor:
        # byte_vals: (n_blocks, n_bytes)
        trits = []
        x = byte_vals
        for _ in range(5):
            trits.append(x % 3)
            x = x // 3
        # MS trit first → reverse
        return torch.stack(list(reversed(trits)), dim=-1)  # (..., 5)

    def peel4(byte_vals: torch.Tensor) -> torch.Tensor:
        trits = []
        x = byte_vals
        for _ in range(4):
            trits.append(x % 3)
            x = x // 3
        return torch.stack(list(reversed(trits)), dim=-1)

    # Per PR table bands are interleaved across bytes — reconstruct 256 in order 0..255
    # Simpler correct approach used by reference: sequential fill following spec bands.
    # Band mapping from PR:
    # bytes 0..32: 5 trits → positions 0..32, 32..64, 64..96, 96..128, 128..160
    b0 = qs[:, 0:32]
    t0 = peel5(b0)  # (n, 32, 5)
    out[:, 0:32] = t0[:, :, 0]
    out[:, 32:64] = t0[:, :, 1]
    out[:, 64:96] = t0[:, :, 2]
    out[:, 96:128] = t0[:, :, 3]
    out[:, 128:160] = t0[:, :, 4]

    # bytes 32..48 → 160..240
    b1 = qs[:, 32:48]
    t1 = peel5(b1)  # (n, 16, 5)
    out[:, 160:176] = t1[:, :, 0]
    out[:, 176:192] = t1[:, :, 1]
    out[:, 192:208] = t1[:, :, 2]
    out[:, 208:224] = t1[:, :, 3]
    out[:, 224:240] = t1[:, :, 4]

    # bytes 48..52 → 240..256 (4 trits)
    b2 = qs[:, 48:52]
    t2 = peel4(b2)  # (n, 4, 4)
    out[:, 240:244] = t2[:, :, 0]
    out[:, 244:248] = t2[:, :, 1]
    out[:, 248:252] = t2[:, :, 2]
    out[:, 252:256] = t2[:, :, 3]

    q = out.to(dtype) - 1.0
    return (d.reshape(n_blocks, 1) * q).reshape(n_blocks, QK_K)

def build_dequantize_table() -> Dict[Any, Callable]:
    table = {
        gguf.GGMLQuantizationType.BF16: dequantize_blocks_BF16,
        gguf.GGMLQuantizationType.Q8_0: dequantize_blocks_Q8_0,
        gguf.GGMLQuantizationType.Q5_1: dequantize_blocks_Q5_1,
        gguf.GGMLQuantizationType.Q5_0: dequantize_blocks_Q5_0,
        gguf.GGMLQuantizationType.Q4_1: dequantize_blocks_Q4_1,
        gguf.GGMLQuantizationType.Q4_0: dequantize_blocks_Q4_0,
        gguf.GGMLQuantizationType.Q6_K: dequantize_blocks_Q6_K,
        gguf.GGMLQuantizationType.Q5_K: dequantize_blocks_Q5_K,
        gguf.GGMLQuantizationType.Q4_K: dequantize_blocks_Q4_K,
        gguf.GGMLQuantizationType.Q3_K: dequantize_blocks_Q3_K,
        gguf.GGMLQuantizationType.Q2_K: dequantize_blocks_Q2_K,
        gguf.GGMLQuantizationType.IQ4_NL: dequantize_blocks_IQ4_NL,
        gguf.GGMLQuantizationType.IQ4_XS: dequantize_blocks_IQ4_XS,
    }
    # Optional enums depending on gguf-py version
    tq2 = safe_qtype_attr("TQ2_0")
    tq1 = safe_qtype_attr("TQ1_0")
    if tq2 is not None:
        table[tq2] = dequantize_blocks_TQ2_0
    if tq1 is not None:
        table[tq1] = dequantize_blocks_TQ1_0
    # Aliases some tooling uses
    for alias in ("IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ1_S", "IQ1_M"):
        qt = safe_qtype_attr(alias)
        if qt is not None:
            # torch path not hand-written → leave to numpy fallback
            pass
    return table

DEQUANTIZE_FUNCTIONS = build_dequantize_table()

def dequantize_data(
    data: torch.Tensor,
    qtype: gguf.GGMLQuantizationType,
    oshape: torch.Size,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    fn = DEQUANTIZE_FUNCTIONS[qtype]
    rows = data.reshape((-1, data.shape[-1])).view(torch.uint8)
    n_blocks = rows.numel() // type_size
    blocks = rows.reshape((n_blocks, type_size))
    out = fn(blocks, block_size, type_size, dtype)
    return out.reshape(oshape)

def dequantize_tensor(
    data: torch.Tensor,
    qtype: gguf.GGMLQuantizationType,
    oshape: torch.Size,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Dequantize one tensor. Torch path for 2/4/5/6/8-bit + TQ; else gguf numpy."""
    if is_torch_compatible_qtype(qtype):
        if qtype == gguf.GGMLQuantizationType.F32:
            t = data.view(torch.float32)
        elif qtype == gguf.GGMLQuantizationType.F16:
            t = data.view(torch.float16)
        else:
            t = data
        t = t.reshape(oshape)
        return t.to(dtype) if dtype is not None else t

    if qtype in DEQUANTIZE_FUNCTIONS:
        return dequantize_data(data, qtype, oshape, dtype=dtype)

    # Numpy path (IQ1/IQ2/IQ3 grids, any new types gguf-py knows)
    name = getattr(qtype, "name", repr(qtype))
    print(f"[warn] numpy fallback dequant for qtype={name}")
    arr = gguf.quants.dequantize(data.cpu().numpy(), qtype)
    t = torch.from_numpy(np.ascontiguousarray(arr)).reshape(oshape)
    return t.to(dtype) if dtype is not None else t

def quant_bitwidth_label(qtype) -> str:
    name = getattr(qtype, "name", str(qtype))
    table = {
        "F32": "32",
        "F16": "16",
        "BF16": "16",
        "Q8_0": "8",
        "Q6_K": "6",
        "Q5_0": "5",
        "Q5_1": "5",
        "Q5_K": "5",
        "Q4_0": "4",
        "Q4_1": "4",
        "Q4_K": "4",
        "IQ4_NL": "4",
        "IQ4_XS": "4",
        "Q3_K": "3",
        "Q2_K": "2",
        "TQ2_0": "2",
        "TQ1_0": "1.58",
        "IQ2_XXS": "2",
        "IQ2_XS": "2",
        "IQ2_S": "2",
        "IQ3_XXS": "3",
        "IQ3_S": "3",
        "IQ1_S": "1.5",
        "IQ1_M": "1.5",
    }
    return table.get(name, "?")

# =============================================================================
# GGUF metadata + state dict
# =============================================================================

def get_orig_shape(reader: gguf.GGUFReader, tensor_name: str) -> Optional[torch.Size]:
    field_key = f"comfy.gguf.orig_shape.{tensor_name}"
    field = reader.get_field(field_key)
    if field is None:
        return None
    if (
        len(field.types) != 2
        or field.types[0] != gguf.GGUFValueType.ARRAY
        or field.types[1] != gguf.GGUFValueType.INT32
    ):
        raise TypeError(f"Bad orig_shape metadata for {field_key}")
    return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in field.data))

def get_field_value(reader: gguf.GGUFReader, field_name: str, field_type=str):
    field = reader.get_field(field_name)
    if field is None:
        return None
    if field_type == str:
        return str(field.parts[field.data[-1]], encoding="utf-8")
    if field_type in (int, float, bool):
        return field_type(field.parts[field.data[-1]].item())
    raise TypeError(f"Unsupported field_type={field_type}")

def parse_gguf_metadata(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    reader = gguf.GGUFReader(path)
    arch = get_field_value(reader, "general.architecture", str)
    gtype = get_field_value(reader, "general.type", str)

    tensors_info: List[Dict[str, Any]] = []
    qtype_counts: Dict[str, int] = {}
    bit_counts: Dict[str, int] = {}

    for tensor in reader.tensors:
        shape = get_orig_shape(reader, tensor.name)
        if shape is None:
            shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
        type_name = getattr(tensor.tensor_type, "name", repr(tensor.tensor_type))
        qtype_counts[type_name] = qtype_counts.get(type_name, 0) + 1
        bw = quant_bitwidth_label(tensor.tensor_type)
        bit_counts[bw] = bit_counts.get(bw, 0) + 1
        tensors_info.append(
            {
                "name": tensor.name,
                "shape": tuple(shape),
                "tensor_type": tensor.tensor_type,
                "type_name": type_name,
                "bitwidth": bw,
                "n_elements": int(np.prod(shape)),
            }
        )
    return {
        "path": path,
        "architecture": arch,
        "general_type": gtype,
        "tensors": tensors_info,
        "qtype_counts": qtype_counts,
        "bit_counts": bit_counts,
    }

def print_gguf_summary(meta: Dict[str, Any]) -> None:
    print("=" * 68)
    print(f"GGUF file     : {meta['path']}")
    print(f"Architecture  : {meta.get('architecture')}")
    print(f"General type  : {meta.get('general_type')}")
    print(f"Tensor count  : {len(meta['tensors'])}")
    print(
        "Quant types   : "
        + ", ".join(f"{k}x{v}" for k, v in meta["qtype_counts"].items())
    )
    print(
        "Bit widths    : "
        + ", ".join(f"{k}-bit×{v}" for k, v in sorted(meta.get("bit_counts", {}).items()))
    )
    print("-" * 68)
    for t in meta["tensors"][:12]:
        print(
            f"  {t['name'][:48]:48s} {str(t['shape']):20s} "
            f"{t['type_name']:8s} (~{t['bitwidth']}b)"
        )
    if len(meta["tensors"]) > 12:
        print(f"  ... ({len(meta['tensors']) - 12} more)")
    print("=" * 68)

def build_state_dict(
    path: str,
    dtype: torch.dtype = torch.float16,
    handle_prefix: Optional[str] = "model.diffusion_model.",
    device: Optional[torch.device] = None,
    dequant_device: Optional[torch.device] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    if device is None:
        device = torch.device("cpu")
    if dequant_device is None:
        dequant_device = torch.device("cpu")

    reader = gguf.GGUFReader(path)
    meta = parse_gguf_metadata(path)
    print_gguf_summary(meta)

    has_prefix = False
    prefix_len = 0
    if handle_prefix is not None:
        names = {t.name for t in reader.tensors}
        has_prefix = any(n.startswith(handle_prefix) for n in names)
        prefix_len = len(handle_prefix)

    state_dict: Dict[str, torch.Tensor] = {}
    qtype_counts: Dict[str, int] = {}

    for tensor in reader.tensors:
        raw_name = tensor.name
        sd_key = raw_name
        if has_prefix:
            if not raw_name.startswith(handle_prefix):
                continue
            sd_key = raw_name[prefix_len:]

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="The given NumPy array is not writable"
            )
            torch_data = torch.from_numpy(tensor.data)

        shape = get_orig_shape(reader, raw_name)
        if shape is None:
            shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))

        qtype = tensor.tensor_type
        type_name = getattr(qtype, "name", repr(qtype))
        qtype_counts[type_name] = qtype_counts.get(type_name, 0) + 1

        if qtype in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
            if qtype == gguf.GGMLQuantizationType.F32:
                weight = torch_data.view(torch.float32).reshape(shape)
            else:
                weight = torch_data.view(torch.float16).reshape(shape)
            weight = weight.to(device=device, dtype=dtype)
        else:
            torch_data = torch_data.to(dequant_device)
            dequant_dtype = (
                torch.float16
                if dtype in (torch.float16, torch.bfloat16)
                else torch.float32
            )
            weight = dequantize_tensor(torch_data, qtype, shape, dtype=dequant_dtype)
            weight = weight.to(device=device, dtype=dtype)

        state_dict[sd_key] = weight.contiguous()
        if len(state_dict) % 25 == 0:
            print(f"  dequantized {len(state_dict)} tensors ...")

    print(f"Done. Dequantized {len(state_dict)} tensors → {dtype}")
    meta["qtype_counts"] = qtype_counts
    return state_dict, meta

def patch_model(
    model: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
    strict: bool = False,
    assign: bool = True,
) -> torch.nn.Module:
    missing, unexpected = model.load_state_dict(
        state_dict, strict=strict, assign=assign
    )
    if missing:
        print(f"[patch] missing ({len(missing)}): {missing[:6]} ...")
    if unexpected:
        print(f"[patch] unexpected ({len(unexpected)}): {unexpected[:6]} ...")
    model.eval()
    return model

# =============================================================================
# Prompt embeds (stdlib-only char embedder; optional external text_encoder)
# =============================================================================

def _stable_seed_from_text(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFFFFFF

def tokenize_chars(text: str, max_length: int = 77) -> torch.Tensor:
    ids = [min(ord(c), 127) for c in text[:max_length]]
    if len(ids) < max_length:
        ids = ids + [0] * (max_length - len(ids))
    return torch.tensor(ids, dtype=torch.long)

def build_text_embedding_table(
    vocab_size: int,
    embed_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int = 0,
) -> torch.nn.Embedding:
    emb = torch.nn.Embedding(vocab_size, embed_dim, device=device, dtype=dtype)
    with torch.no_grad():
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        emb.weight.copy_(
            torch.randn(vocab_size, embed_dim, generator=g).to(device=device, dtype=dtype)
            * 0.02
        )
    emb.eval()
    return emb

def encode_prompt_text(
    prompt: str,
    embed_table: torch.nn.Embedding,
    max_length: int = 77,
) -> torch.Tensor:
    device = embed_table.weight.device
    ids = tokenize_chars(prompt, max_length=max_length).to(device)
    with torch.no_grad():
        seq = embed_table(ids).unsqueeze(0)  # [1, L, D]
        mask = (ids != 0).float().unsqueeze(0).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (seq * mask).sum(dim=1) / denom  # [1, D]

def encode_prompts_cfg(
    prompt: str,
    negative_prompt: str,
    embed_table: torch.nn.Embedding,
    max_length: int = 77,
    use_negative: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    cond = encode_prompt_text(prompt, embed_table, max_length=max_length)
    if not use_negative:
        return cond, None
    uncond = encode_prompt_text(
        negative_prompt if negative_prompt is not None else "",
        embed_table,
        max_length=max_length,
    )
    return cond, uncond

# =============================================================================
# Schedulers: Flow-Match (Flux) + classic Euler-discrete
# =============================================================================

def make_sigmas_flowmatch(
    num_inference_steps: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    shift: float = 1.0,
) -> torch.Tensor:
    """
    Flux-style flow-matching timesteps in (0, 1], shifted.
    sigma goes high → low.
    """
    # Base linspace of timesteps in [1, 0)
    t = torch.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps, device=device)
    if abs(shift - 1.0) > 1e-6:
        t = (shift * t) / (1.0 + (shift - 1.0) * t)
    return t.to(dtype=dtype)

def flowmatch_euler_step(
    sample: torch.Tensor,
    model_output: torch.Tensor,
    sigma: float,
    sigma_next: float,
) -> torch.Tensor:
    """
    Euler ODE step for rectified-flow / Flux:
        dx/dt = v_theta(x, t)   with  x_{t+dt} = x_t + dt * v
    Here model_output is velocity v (common Flux convention).
    """
    dt = sigma_next - sigma
    return sample + model_output * dt

def make_beta_schedule(
    num_train_timesteps: int = 1000,
    beta_start: float = 0.00085,
    beta_end: float = 0.012,
    schedule: str = "scaled_linear",
) -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, num_train_timesteps)
    if schedule == "scaled_linear":
        return (
            torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_train_timesteps) ** 2
        )
    raise ValueError(schedule)

def make_scheduler_ddpm(
    num_train_timesteps: int = 1000,
) -> Dict[str, torch.Tensor]:
    betas = make_beta_schedule(num_train_timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "num_train_timesteps": torch.tensor(num_train_timesteps),
    }

def make_inference_timesteps_ddpm(
    num_inference_steps: int,
    num_train_timesteps: int = 1000,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    step = max(num_train_timesteps // num_inference_steps, 1)
    timesteps = torch.arange(0, num_inference_steps, device=device) * step
    return torch.flip(timesteps, dims=[0]).long().clamp(max=num_train_timesteps - 1)

def euler_ddpm_step(
    sample: torch.Tensor,
    model_output: torch.Tensor,
    timestep: int,
    prev_timestep: int,
    alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    alpha_prod_t = alphas_cumprod[timestep]
    alpha_prod_t_prev = (
        alphas_cumprod[prev_timestep]
        if prev_timestep >= 0
        else torch.tensor(1.0, device=alphas_cumprod.device, dtype=alphas_cumprod.dtype)
    )
    beta_prod_t = 1.0 - alpha_prod_t
    pred_x0 = (sample - beta_prod_t.sqrt() * model_output) / alpha_prod_t.sqrt()
    pred_eps = model_output
    dir_xt = (1.0 - alpha_prod_t_prev).sqrt() * pred_eps
    return alpha_prod_t_prev.sqrt() * pred_x0 + dir_xt

# =============================================================================
# CFG + sampling loops
# =============================================================================

def apply_cfg(
    pred_cond: torch.Tensor,
    pred_uncond: Optional[torch.Tensor],
    cfg_scale: float,
) -> torch.Tensor:
    if pred_uncond is None or abs(cfg_scale - 1.0) < 1e-6:
        return pred_cond
    return pred_uncond + cfg_scale * (pred_cond - pred_uncond)

def denoising_loop_flowmatch(
    denoise_fn: Callable,
    latents: torch.Tensor,
    sigmas: torch.Tensor,
    cond_embeds: torch.Tensor,
    uncond_embeds: Optional[torch.Tensor],
    cfg_scale: float,
    accepts_negative: bool,
) -> torch.Tensor:
    use_cfg = (
        accepts_negative
        and uncond_embeds is not None
        and abs(cfg_scale - 1.0) >= 1e-6
    )
    x = latents
    # append terminal sigma 0 for final step
    sigmas_ext = torch.cat(
        [sigmas, torch.zeros(1, device=sigmas.device, dtype=sigmas.dtype)]
    )
    n = len(sigmas)
    for i in range(n):
        sigma = float(sigmas_ext[i].item())
        sigma_next = float(sigmas_ext[i + 1].item())
        # discrete timestep id for modules that expect long t
        t_id = int(sigma * 1000)
        t_batch = torch.full((x.shape[0],), t_id, device=x.device, dtype=torch.long)

        if use_cfg:
            v_c = denoise_fn(x, t_batch, cond_embeds, sigma)
            v_u = denoise_fn(x, t_batch, uncond_embeds, sigma)
            v = apply_cfg(v_c, v_u, cfg_scale)
        else:
            v = denoise_fn(x, t_batch, cond_embeds, sigma)

        x = flowmatch_euler_step(x, v, sigma, sigma_next)

        if (i + 1) % max(1, n // 5) == 0 or i == n - 1:
            print(
                f"  flow step {i + 1}/{n}  σ={sigma:.4f}→{sigma_next:.4f}  "
                f"x_std={x.float().std().item():.5f}"
            )
    return x

def denoising_loop_ddpm(
    denoise_fn: Callable,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    cond_embeds: torch.Tensor,
    uncond_embeds: Optional[torch.Tensor],
    cfg_scale: float,
    alphas_cumprod: torch.Tensor,
    accepts_negative: bool,
) -> torch.Tensor:
    use_cfg = (
        accepts_negative
        and uncond_embeds is not None
        and abs(cfg_scale - 1.0) >= 1e-6
    )
    x = latents
    alphas_cumprod = alphas_cumprod.to(device=x.device, dtype=torch.float32)
    n = len(timesteps)
    for i, t in enumerate(timesteps):
        t_int = int(t.item())
        t_batch = torch.full((x.shape[0],), t_int, device=x.device, dtype=torch.long)
        if use_cfg:
            e_c = denoise_fn(x, t_batch, cond_embeds, None)
            e_u = denoise_fn(x, t_batch, uncond_embeds, None)
            eps = apply_cfg(e_c, e_u, cfg_scale)
        else:
            eps = denoise_fn(x, t_batch, cond_embeds, None)

        t_prev = int(timesteps[i + 1].item()) if i + 1 < n else -1
        x = euler_ddpm_step(x, eps, t_int, t_prev, alphas_cumprod)

        if (i + 1) % max(1, n // 5) == 0 or i == n - 1:
            print(
                f"  ddpm step {i + 1}/{n}  t={t_int}  "
                f"x_std={x.float().std().item():.5f}"
            )
    return x

# =============================================================================
# Demo Flux-ish denoiser (smoke / preview when no arch is supplied)
# =============================================================================

def _sinusoidal_timestep_embedding(
    timesteps: torch.Tensor, dim: int, max_period: int = 10000
) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(0, half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb

def build_demo_denoiser(
    state_dict: Optional[Dict[str, torch.Tensor]],
    latent_channels: int,
    latent_h: int,
    latent_w: int,
    context_dim: int,
    hidden: int = 384,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float16,
) -> Tuple[torch.nn.Module, Callable]:
    flat_dim = latent_channels * latent_h * latent_w
    time_dim = hidden

    in_proj = torch.nn.Linear(
        flat_dim + context_dim + time_dim, hidden, device=device, dtype=dtype
    )
    mid = torch.nn.Linear(hidden, hidden, device=device, dtype=dtype)
    out_proj = torch.nn.Linear(hidden, flat_dim, device=device, dtype=dtype)
    time_proj = torch.nn.Linear(time_dim, time_dim, device=device, dtype=dtype)

    if state_dict:
        candidates = [
            (k, v) for k, v in state_dict.items() if v.ndim == 2 and v.shape[0] > 8
        ]
        candidates.sort(key=lambda kv: kv[1].numel(), reverse=True)
        for layer, cand in zip((in_proj, mid, out_proj, time_proj), candidates):
            k, w = cand
            with torch.no_grad():
                src = w.detach().float().cpu()
                tgt = torch.zeros(layer.weight.shape)
                r = min(src.shape[0], tgt.shape[0])
                c = min(src.shape[1], tgt.shape[1])
                tgt[:r, :c] = src[:r, :c]
                std = tgt.std().clamp(min=1e-3)
                tgt = tgt / std * 0.02
                layer.weight.copy_(tgt.to(device=device, dtype=dtype))
            print(f"  demo warm-start {k} → {tuple(layer.weight.shape)}")

    net = torch.nn.Module()
    net.in_proj = in_proj
    net.mid = mid
    net.out_proj = out_proj
    net.time_proj = time_proj
    net.eval()

    def denoise_fn(x, t_batch, text_embeds, sigma=None):
        b, c, h, w = x.shape
        x_flat = x.reshape(b, -1)
        t_emb = _sinusoidal_timestep_embedding(t_batch, time_dim).to(
            device=x.device, dtype=x.dtype
        )
        t_emb = net.time_proj(t_emb)
        ctx = text_embeds
        if ctx.shape[0] == 1 and b > 1:
            ctx = ctx.expand(b, -1)
        if ctx.shape[-1] < context_dim:
            ctx = F.pad(ctx, (0, context_dim - ctx.shape[-1]))
        elif ctx.shape[-1] > context_dim:
            ctx = ctx[..., :context_dim]
        hcat = torch.cat([x_flat, ctx, t_emb], dim=-1)
        hdn = F.silu(net.in_proj(hcat))
        hdn = F.silu(net.mid(hdn))
        out = net.out_proj(hdn).view(b, c, h, w)
        return out

    return net, denoise_fn

def wrap_user_denoise_fn(model: torch.nn.Module, call_style: str = "flux") -> Callable:
    def denoise_fn(x, t_batch, text_embeds, sigma=None):
        if call_style == "flux":
            # common: model(x, timestep, context) → tensor or .sample
            out = model(x, t_batch, text_embeds)
            return out.sample if hasattr(out, "sample") else out
        if call_style == "diffusers":
            enc = text_embeds.unsqueeze(1) if text_embeds.ndim == 2 else text_embeds
            out = model(x, t_batch, encoder_hidden_states=enc)
            return out.sample if hasattr(out, "sample") else out
        if call_style == "kwargs":
            return model(x, timestep=t_batch, context=text_embeds, sigma=sigma)
        return model(x, t_batch, text_embeds)

    return denoise_fn

# =============================================================================
# Latents helpers + Flux unpack
# =============================================================================

def make_noise_latents(
    batch_size: int,
    channels: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    noise = torch.randn(
        batch_size, channels, height, width, generator=g, dtype=torch.float32
    )
    return noise.to(device=device, dtype=dtype)

def unpack_flux_latents(
    latents: torch.Tensor,
    height_px: int,
    width_px: int,
    vae_scale_factor: int = FLUX_VAE_SCALE_SPATIAL,
) -> torch.Tensor:
    """
    Diffusers Flux packing reverse:
      packed [B, H/2 * W/2, C*4]  →  [B, C, H, W]
    Also accepts already-unpacked [B, C, H, W].
    """
    if latents.ndim == 4:
        return latents
    if latents.ndim != 3:
        raise ValueError(f"Expected 3D packed or 4D latent, got {latents.shape}")

    batch_size, num_patches, channels_x4 = latents.shape
    h = 2 * (int(height_px) // (vae_scale_factor * 2))
    w = 2 * (int(width_px) // (vae_scale_factor * 2))
    c = channels_x4 // 4
    latents = latents.view(batch_size, h // 2, w // 2, c, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(batch_size, c, h, w)

def pack_flux_latents(latents: torch.Tensor) -> torch.Tensor:
    """[B, C, H, W] → [B, H/2*W/2, C*4]"""
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(b, (h // 2) * (w // 2), c * 4)

# =============================================================================
# Latent → RGB  (real Flux needs VAE; we ship a strong approx + optional hook)
# =============================================================================

def approximate_vae_decode(
    latents: torch.Tensor,
    height_px: int,
    width_px: int,
    scaling_factor: float = FLUX_VAE_SCALING_FACTOR,
    shift_factor: float = FLUX_VAE_SHIFT_FACTOR,
) -> torch.Tensor:
    """
    Approximate "VAE decode" without shipping AutoencoderKL weights.

    Steps:
      1. Undo Flux latent shift/scale
      2. Project C (16 or 4) channels → 3 RGB via fixed orthonormal mix
      3. Bilinear upsample to pixel resolution
      4. Mild spatial sharpening + tanh squash to [-1, 1]

    Good enough for pipeline smoke tests & quant validation PNGs.
    Plug a real VAE via `vae_decode_fn` in run_inference for production images.
    """
    x = latents.float()
    # Flux: image_latents = (latents / scaling) + shift   during decode prep
    x = (x / scaling_factor) + shift_factor

    b, c, h, w = x.shape
    # Deterministic channel → RGB mix (seeded Hadamard-ish acids)
    g = torch.Generator(device="cpu")
    g.manual_seed(0xF10A5)
    mix = torch.randn(3, c, generator=g)
    mix = mix / mix.norm(dim=1, keepdim=True).clamp(min=1e-6)
    mix = mix.to(device=x.device, dtype=x.dtype)

    rgb = torch.einsum("bchw,kc->bkhw", x, mix)

    # Local contrast
    blur = F.avg_pool2d(rgb, kernel_size=3, stride=1, padding=1)
    rgb = rgb + 0.35 * (rgb - blur)

    rgb = F.interpolate(
        rgb, size=(height_px, width_px), mode="bilinear", align_corners=False
    )
    # Map to [-1, 1]
    rgb = torch.tanh(rgb * 0.75)
    return rgb

def tensor_to_uint8_image(img: torch.Tensor) -> np.ndarray:
    """
    img: [B, 3, H, W] or [3, H, W] in [-1, 1] or [0, 1] → uint8 HWC RGB
    """
    if img.ndim == 4:
        img = img[0]
    x = img.detach().float().cpu()
    if x.min() < -0.05:
        x = (x + 1.0) * 0.5
    x = x.clamp(0.0, 1.0)
    x = (x * 255.0).round().to(torch.uint8)
    return x.permute(1, 2, 0).numpy()  # HWC

def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )

def write_png(path: str, rgb_hwc_u8: np.ndarray) -> str:
    """
    Minimal truecolor PNG writer (no PIL).
    rgb_hwc_u8: uint8 array HxWx3
    """
    if rgb_hwc_u8.dtype != np.uint8 or rgb_hwc_u8.ndim != 3 or rgb_hwc_u8.shape[2] != 3:
        raise ValueError("write_png expects HxWx3 uint8 RGB")
    h, w, _ = rgb_hwc_u8.shape
    # filter byte 0 per scanline
    raw = b"".join(b"\x00" + rgb_hwc_u8[y].tobytes() for y in range(h))
    compressed = zlib.compress(raw, level=6)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8-bit RGB
    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", ihdr)
    png += _png_chunk(b"IDAT", compressed)
    png += _png_chunk(b"IEND", b"")

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(png)
    return path

def latents_to_png(
    latents: torch.Tensor,
    path: str,
    height_px: int,
    width_px: int,
    vae_decode_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    scaling_factor: float = FLUX_VAE_SCALING_FACTOR,
    shift_factor: float = FLUX_VAE_SHIFT_FACTOR,
) -> str:
    """Decode latents → RGB → write PNG. Returns path."""
    z = latents
    if z.ndim == 3:
        z = unpack_flux_latents(z, height_px, width_px)

    if vae_decode_fn is not None:
        with torch.no_grad():
            # real VAE usually expects (z / scale) + shift already applied by caller or inside
            img = vae_decode_fn(z)
    else:
        img = approximate_vae_decode(
            z,
            height_px=height_px,
            width_px=width_px,
            scaling_factor=scaling_factor,
            shift_factor=shift_factor,
        )

    arr = tensor_to_uint8_image(img)
    return write_png(path, arr)

# =============================================================================
# Main entry
# =============================================================================

def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }[name.lower()]

def model_accepts_negative(
    accepts_negative: Optional[bool],
    cfg_scale: float,
) -> bool:
    if accepts_negative is not None:
        return bool(accepts_negative)
    return abs(cfg_scale - 1.0) >= 1e-6

def run_inference(
    gguf_path: str,
    prompt: str,
    negative_prompt: str = "",
    cfg_scale: float = 3.5,
    num_inference_steps: int = 20,
    seed: int = 42,
    height: int = 512,
    width: int = 512,
    latent_channels: int = FLUX_LATENT_CHANNELS,
    batch_size: int = 1,
    dtype: str = "float16",
    device: str = "cpu",
    handle_prefix: Optional[str] = "model.diffusion_model.",
    context_dim: int = 256,
    max_prompt_length: int = 77,
    accepts_negative: Optional[bool] = None,
    scheduler: str = "flowmatch",
    flow_shift: float = 1.0,
    output_png: str = "output.png",
    model: Optional[torch.nn.Module] = None,
    denoise_call_style: str = "flux",
    text_encoder: Optional[Callable[[str], torch.Tensor]] = None,
    vae_decode_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Dict[str, Any]:
    """
    End-to-end: GGUF dequant → prompt encode → multi-step CFG sample → PNG.

    Parameters
    ----------
    gguf_path           : Flux / DiT weights in GGUF (Q2/Q4/Q8/K/IQ/TQ ok)
    prompt              : positive prompt
    negative_prompt     : negative prompt (CFG)
    cfg_scale           : guidance scale (Flux often 1–5; 1 disables CFG)
    num_inference_steps : denoising steps
    seed                : latents RNG seed
    height, width       : **pixel** size of output PNG (latent is /8)
    latent_channels     : 16 for Flux, 4 for SD-like
    output_png          : path to write the PNG
    scheduler           : 'flowmatch' (Flux) | 'ddpm'
    model               : optional real transformer; else demo residual net
    vae_decode_fn       : optional real VAE decode(latents[B,C,h,w]) -> [B,3,H,W]
    text_encoder        : optional fn(str)->[1,D] embeds
    """
    torch_dtype = resolve_dtype(dtype)
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA unavailable — CPU fallback")
        torch_device = torch.device("cpu")

    use_negative = model_accepts_negative(accepts_negative, cfg_scale)

    # Flux latent spatial size
    assert height % FLUX_VAE_SCALE_SPATIAL == 0 and width % FLUX_VAE_SCALE_SPATIAL == 0, (
        f"height/width must be multiples of {FLUX_VAE_SCALE_SPATIAL}"
    )
    latent_h = height // FLUX_VAE_SCALE_SPATIAL
    latent_w = width // FLUX_VAE_SCALE_SPATIAL

    print("-" * 68)
    print("Standalone GGUF Flux-style inference → PNG")
    print(f"  prompt           : {prompt!r}")
    print(f"  negative_prompt  : {negative_prompt!r}")
    print(f"  cfg_scale        : {cfg_scale}")
    print(f"  steps            : {num_inference_steps}")
    print(f"  scheduler        : {scheduler}")
    print(f"  accepts_negative : {use_negative}")
    print(f"  seed             : {seed}")
    print(f"  image (px)       : {height}x{width}")
    print(f"  latent           : B={batch_size} C={latent_channels} "
          f"H={latent_h} W={latent_w}")
    print(f"  dtype / device   : {torch_dtype} @ {torch_device}")
    print(f"  output_png       : {output_png}")
    print("-" * 68)

    # 1) Load GGUF
    prefix = handle_prefix if handle_prefix else None
    state_dict, meta = build_state_dict(
        path=gguf_path,
        dtype=torch_dtype,
        handle_prefix=prefix,
        device=torch_device,
        dequant_device=torch.device("cpu"),
    )

    # 2) Text embeds
    if text_encoder is not None:
        with torch.no_grad():
            cond_embeds = text_encoder(prompt).to(device=torch_device, dtype=torch_dtype)
            uncond_embeds = (
                text_encoder(negative_prompt).to(device=torch_device, dtype=torch_dtype)
                if use_negative
                else None
            )
        if cond_embeds.ndim == 3:
            cond_embeds = cond_embeds.mean(dim=1)
        if uncond_embeds is not None and uncond_embeds.ndim == 3:
            uncond_embeds = uncond_embeds.mean(dim=1)
    else:
        emb_seed = _stable_seed_from_text("gguf-char-embedding-v1")
        embed_table = build_text_embedding_table(
            128, context_dim, torch_device, torch_dtype, seed=emb_seed
        )
        cond_embeds, uncond_embeds = encode_prompts_cfg(
            prompt, negative_prompt, embed_table, max_prompt_length, use_negative
        )
        if batch_size > 1:
            cond_embeds = cond_embeds.expand(batch_size, -1).contiguous()
            if uncond_embeds is not None:
                uncond_embeds = uncond_embeds.expand(batch_size, -1).contiguous()

    print(
        f"cond embeds: {tuple(cond_embeds.shape)}"
        + (
            f" | uncond: {tuple(uncond_embeds.shape)}"
            if uncond_embeds is not None
            else " | uncond: <off>"
        )
    )

    # 3) Denoiser
    if model is not None:
        print("Using user-provided architecture")
        model = model.to(device=torch_device, dtype=torch_dtype)
        model = patch_model(model, state_dict, strict=False)
        denoise_fn = wrap_user_denoise_fn(model, call_style=denoise_call_style)
    else:
        print("No architecture provided — demo residual denoiser (preview PNG)")
        ctx_dim = int(cond_embeds.shape[-1])
        _, denoise_fn = build_demo_denoiser(
            state_dict=state_dict,
            latent_channels=latent_channels,
            latent_h=latent_h,
            latent_w=latent_w,
            context_dim=ctx_dim,
            hidden=max(384, ctx_dim),
            device=torch_device,
            dtype=torch_dtype,
        )

    # 4) Noise + sample
    latents = make_noise_latents(
        batch_size, latent_channels, latent_h, latent_w,
        torch_device, torch_dtype, seed,
    )
    print(f"init noise std: {latents.float().std().item():.5f}")

    with torch.no_grad():
        if scheduler.lower() in ("flowmatch", "flow", "flux"):
            sigmas = make_sigmas_flowmatch(
                num_inference_steps, torch_device, torch.float32, shift=flow_shift
            )
            print(f"flow sigmas: {[round(float(s), 4) for s in sigmas[:6]]} ...")
            final_latents = denoising_loop_flowmatch(
                denoise_fn=denoise_fn,
                latents=latents,
                sigmas=sigmas,
                cond_embeds=cond_embeds,
                uncond_embeds=uncond_embeds,
                cfg_scale=cfg_scale,
                accepts_negative=use_negative,
            )
        else:
            sched = make_scheduler_ddpm()
            timesteps = make_inference_timesteps_ddpm(
                num_inference_steps, device=torch_device
            )
            print(f"ddpm timesteps: {timesteps.tolist()[:8]} ...")
            final_latents = denoising_loop_ddpm(
                denoise_fn=denoise_fn,
                latents=latents,
                timesteps=timesteps,
                cond_embeds=cond_embeds,
                uncond_embeds=uncond_embeds,
                cfg_scale=cfg_scale,
                alphas_cumprod=sched["alphas_cumprod"],
                accepts_negative=use_negative,
            )

    # 5) PNG
    png_path = latents_to_png(
        final_latents,
        path=output_png,
        height_px=height,
        width_px=width,
        vae_decode_fn=vae_decode_fn,
    )
    print(f"Wrote PNG → {png_path}")

    total_params = sum(v.numel() for v in state_dict.values())
    result = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "cfg_scale": cfg_scale,
        "num_inference_steps": num_inference_steps,
        "seed": seed,
        "scheduler": scheduler,
        "accepts_negative": use_negative,
        "dtype": str(torch_dtype),
        "device": str(torch_device),
        "image_size": (height, width),
        "latent_shape": list(final_latents.shape),
        "output_png": png_path,
        "meta": {
            "architecture": meta.get("architecture"),
            "path": meta.get("path"),
            "qtype_counts": meta.get("qtype_counts"),
            "bit_counts": meta.get("bit_counts"),
            "n_tensors": len(state_dict),
            "total_params": total_params,
        },
        "latents": final_latents,
    }
    print(
        f"Done | steps={num_inference_steps} cfg={cfg_scale} "
        f"params={total_params:,} | {png_path}"
    )
    return result

# =============================================================================
# CLI
# =============================================================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Standalone GGUF Flux loader — CFG sampling — PNG output"
    )
    p.add_argument("gguf_path", type=str, help="Path to Flux/DiT .gguf weights")
    p.add_argument(
        "--prompt",
        type=str,
        default="a red fox standing in deep snow, cinematic light",
    )
    p.add_argument(
        "--negative-prompt",
        type=str,
        default="blurry, low quality, distorted, watermark, text",
    )
    p.add_argument("--cfg-scale", type=float, default=3.5)
    p.add_argument(
        "--steps", type=int, default=20, dest="num_inference_steps"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--height", type=int, default=512, help="Output image height (px)")
    p.add_argument("--width", type=int, default=512, help="Output image width (px)")
    p.add_argument(
        "--channels",
        type=int,
        default=FLUX_LATENT_CHANNELS,
        dest="latent_channels",
        help="Latent channels (16=Flux, 4=SD)",
    )
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument(
        "--prefix",
        type=str,
        default="model.diffusion_model.",
        help="Key prefix to strip ('' disables)",
    )
    p.add_argument("--context-dim", type=int, default=256)
    p.add_argument(
        "--scheduler",
        type=str,
        default="flowmatch",
        choices=["flowmatch", "ddpm"],
    )
    p.add_argument("--flow-shift", type=float, default=1.0)
    p.add_argument(
        "--output",
        type=str,
        default="output.png",
        dest="output_png",
        help="Output PNG path",
    )
    p.add_argument("--no-negative", action="store_true")
    p.add_argument("--force-negative", action="store_true")
    p.add_argument("--meta-only", action="store_true")
    p.add_argument(
        "--save-latents",
        type=str,
        default="",
        help="Optional .pt path for raw latents",
    )
    return p

def main(argv: Optional[List[str]] = None) -> None:
    args = build_argparser().parse_args(argv)

    if args.meta_only:
        meta = parse_gguf_metadata(args.gguf_path)
        print_gguf_summary(meta)
        return

    prefix = args.prefix if args.prefix != "" else None
    if args.no_negative:
        accepts_negative: Optional[bool] = False
    elif args.force_negative:
        accepts_negative = True
    else:
        accepts_negative = None

    result = run_inference(
        gguf_path=args.gguf_path,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        cfg_scale=args.cfg_scale,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        height=args.height,
        width=args.width,
        latent_channels=args.latent_channels,
        batch_size=args.batch_size,
        dtype=args.dtype,
        device=args.device,
        handle_prefix=prefix,
        context_dim=args.context_dim,
        accepts_negative=accepts_negative,
        scheduler=args.scheduler,
        flow_shift=args.flow_shift,
        output_png=args.output_png,
    )

    if args.save_latents:
        torch.save(
            {
                "latents": result["latents"].cpu(),
                "prompt": result["prompt"],
                "negative_prompt": result["negative_prompt"],
                "cfg_scale": result["cfg_scale"],
                "num_inference_steps": result["num_inference_steps"],
                "seed": result["seed"],
                "output_png": result["output_png"],
            },
            args.save_latents,
        )
        print(f"Saved latents → {args.save_latents}")

    print("\nSummary")
    print(f"  prompt    : {result['prompt']!r}")
    print(f"  negative  : {result['negative_prompt']!r}")
    print(f"  cfg/steps : {result['cfg_scale']} / {result['num_inference_steps']}")
    print(f"  PNG       : {result['output_png']}")
    print(f"  quants    : {result['meta'].get('qtype_counts')}")
    print(f"  bitwidths : {result['meta'].get('bit_counts')}")

if __name__ == "__main__":
    main()