# gguf_inference.py
# -----------------------------------------------------------------------------
# Flux.2 Klein GGUF → real text-to-image PNG (diffusers path)
# + optional pure torch/numpy/gguf dequant toolkit (meta / smoke only)
#
# Why your last PNG was colorful static:
#   The GGUF file is ONLY the transformer (DiT). Without the real Flux2Klein
#   architecture, Qwen3 text encoder, and Flux2 VAE, no prompt can turn into
#   a photo. The old "demo MLP + fake VAE" path only visualizes noise.
#
# Real generation deps:
#   pip install -U torch numpy gguf diffusers transformers accelerate
#                 sentencepiece protobuf safetensors pillow huggingface_hub
#
# Example (Flux.2 Klein 4B Q4_0):
#   python gguf_inference.py flux-2-klein-4b-Q4_0.gguf \
#       --prompt "a red fox in deep snow, cinematic lighting" \
#       --negative-prompt "blurry, watermark, text" \
#       --cfg-scale 4.0 --steps 8 \
#       --height 512 --width 512 \
#       --dtype bfloat16 --device cuda \
#       --base-model black-forest-labs/FLUX.2-klein-4B \
#       --output fox.png
# -----------------------------------------------------------------------------

import argparse
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
# Constants / dequant (still usable for --meta-only / --dequant-only)
# =============================================================================

QK_K = 256
K_SCALE_SIZE = 12

TORCH_COMPATIBLE_QTYPES = (
    None,
    gguf.GGMLQuantizationType.F32,
    gguf.GGMLQuantizationType.F16,
)

KVALUES_IQ4 = torch.tensor(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
    dtype=torch.int8,
)

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
    return getattr(gguf.GGMLQuantizationType, name, None)

def dequantize_blocks_BF16(blocks, block_size, type_size, dtype=None):
    return (blocks.view(torch.int16).to(torch.int32) << 16).view(torch.float32)

def dequantize_blocks_Q8_0(blocks, block_size, type_size, dtype=None):
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
    d, dmin, scales, qh, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE, QK_K // 8)
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
    scales_l = scales_l.reshape((n_blocks, -1, 1)) >> shift_a
    scales_h = scales_h.reshape((n_blocks, -1, 1)) >> shift_b
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
    return table

DEQUANTIZE_FUNCTIONS = build_dequantize_table()

def dequantize_data(data, qtype, oshape, dtype=None):
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    rows = data.reshape((-1, data.shape[-1])).view(torch.uint8)
    n_blocks = rows.numel() // type_size
    blocks = rows.reshape((n_blocks, type_size))
    out = DEQUANTIZE_FUNCTIONS[qtype](blocks, block_size, type_size, dtype)
    return out.reshape(oshape)

def dequantize_tensor(data, qtype, oshape, dtype=None):
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
    print(f"[warn] numpy fallback for {getattr(qtype, 'name', qtype)}")
    arr = gguf.quants.dequantize(data.cpu().numpy(), qtype)
    t = torch.from_numpy(np.ascontiguousarray(arr)).reshape(oshape)
    return t.to(dtype) if dtype is not None else t

def quant_bitwidth_label(qtype) -> str:
    name = getattr(qtype, "name", str(qtype))
    table = {
        "F32": "32", "F16": "16", "BF16": "16", "Q8_0": "8", "Q6_K": "6",
        "Q5_0": "5", "Q5_1": "5", "Q5_K": "5",
        "Q4_0": "4", "Q4_1": "4", "Q4_K": "4", "IQ4_NL": "4", "IQ4_XS": "4",
        "Q3_K": "3", "Q2_K": "2",
    }
    return table.get(name, "?")

def get_orig_shape(reader: gguf.GGUFReader, tensor_name: str) -> Optional[torch.Size]:
    field_key = f"comfy.gguf.orig_shape.{tensor_name}"
    field = reader.get_field(field_key)
    if field is None:
        return None
    return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in field.data))

def get_field_value(reader: gguf.GGUFReader, field_name: str, field_type=str):
    field = reader.get_field(field_name)
    if field is None:
        return None
    if field_type == str:
        return str(field.parts[field.data[-1]], encoding="utf-8")
    if field_type in (int, float, bool):
        return field_type(field.parts[field.data[-1]].item())
    return None

def parse_gguf_metadata(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    reader = gguf.GGUFReader(path)
    arch = get_field_value(reader, "general.architecture", str)
    gtype = get_field_value(reader, "general.type", str)
    tensors_info, qtype_counts, bit_counts = [], {}, {}
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
                "type_name": type_name,
                "bitwidth": bw,
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
    print("Quant types   : " + ", ".join(f"{k}x{v}" for k, v in meta["qtype_counts"].items()))
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
    print(
        "NOTE: This GGUF is almost certainly TRANSFORMER-ONLY.\n"
        "      Real images also need: Flux2Klein pipeline + Qwen3 text encoder + Flux2 VAE\n"
        "      (loaded from --base-model via diffusers)."
    )

def build_state_dict(
    path: str,
    dtype: torch.dtype = torch.float16,
    handle_prefix: Optional[str] = "model.diffusion_model.",
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    if device is None:
        device = torch.device("cpu")
    reader = gguf.GGUFReader(path)
    meta = parse_gguf_metadata(path)
    print_gguf_summary(meta)

    has_prefix = False
    prefix_len = 0
    if handle_prefix:
        names = {t.name for t in reader.tensors}
        has_prefix = any(n.startswith(handle_prefix) for n in names)
        prefix_len = len(handle_prefix)

    state_dict: Dict[str, torch.Tensor] = {}
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
        if qtype in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
            weight = (
                torch_data.view(torch.float32 if qtype == gguf.GGMLQuantizationType.F32
                                else torch.float16).reshape(shape)
            )
        else:
            dequant_dtype = (
                torch.float16 if dtype in (torch.float16, torch.bfloat16) else torch.float32
            )
            weight = dequantize_tensor(torch_data, qtype, shape, dtype=dequant_dtype)
        state_dict[sd_key] = weight.to(device=device, dtype=dtype).contiguous()
        if len(state_dict) % 40 == 0:
            print(f"  dequantized {len(state_dict)} tensors ...")

    print(f"Done. Dequantized {len(state_dict)} tensors → {dtype}")
    return state_dict, meta

# =============================================================================
# Minimal PNG helpers (no PIL required for fallback paths)
# =============================================================================

def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )

def write_png_numpy(path: str, rgb_hwc_u8: np.ndarray) -> str:
    if rgb_hwc_u8.dtype != np.uint8 or rgb_hwc_u8.ndim != 3 or rgb_hwc_u8.shape[2] != 3:
        raise ValueError("expect HxWx3 uint8")
    h, w, _ = rgb_hwc_u8.shape
    raw = b"".join(b"\x00" + rgb_hwc_u8[y].tobytes() for y in range(h))
    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += _png_chunk(b"IDAT", zlib.compress(raw, 6))
    png += _png_chunk(b"IEND", b"")
    parent = os.path.dirname(os.path.abspath(path)
                            )
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as f:
        f.write(png)
    return path

def save_pil_image(img, path: str) -> str:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    img.save(path)
    return path

# =============================================================================
# REAL inference: Flux.2 Klein via diffusers + GGUF transformer
# =============================================================================

def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16, "fp16": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float32": torch.float32, "fp32": torch.float32,
    }[name.lower()]

def require_diffusers():
    try:
        import diffusers  # noqa: F401
        from diffusers import (  # noqa: F401
            Flux2KleinPipeline,
            Flux2Transformer2DModel,
        )
    except Exception as e:
        raise ImportError(
            "Real Flux.2 Klein image generation requires recent diffusers.\n"
            "  pip install -U 'diffusers>=0.36' transformers accelerate "
            "safetensors sentencepiece protobuf pillow huggingface_hub\n"
            f"Original import error: {e}"
        ) from e

def load_flux2_klein_transformer_from_gguf(
    gguf_path: str,
    base_model: str,
    torch_dtype: torch.dtype,
    device: str,
):
    """
    Load Flux2Transformer2DModel from a city96/unsloth GGUF file.
    Uses diffusers' native GGUF loader (keeps weights quantized on GPU when possible).
    """
    from diffusers import Flux2Transformer2DModel, GGUFQuantizationConfig

    print(f"Loading Flux2 transformer GGUF: {gguf_path}")
    print(f"  config source: {base_model}  (subfolder=transformer)")

    quant_cfg = GGUFQuantizationConfig(compute_dtype=torch_dtype)

    # Explicit config is required for Klein (shape differs from Flux2 Dev)
    try:
        transformer = Flux2Transformer2DModel.from_single_file(
            gguf_path,
            quantization_config=quant_cfg,
            torch_dtype=torch_dtype,
            config=base_model,
            subfolder="transformer",
        )
    except TypeError:
        # older diffusers API variants
        transformer = Flux2Transformer2DModel.from_single_file(
            gguf_path,
            quantization_config=quant_cfg,
            torch_dtype=torch_dtype,
            config=base_model,
            subfolder="transformer",
        )

    return transformer

def load_flux ug2_klein_pipeline_with_gguf_transformer(
    gguf_path: str,
    base_model: str,
    torch_dtype: torch.dtype,
    device: str,
    cpu_offload: bool = True,
):
    """
    VAE + Qwen3 text encoder + tokenizer + scheduler from base_model,
    transformer weights from GGUF.
    """
    from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel, GGUFQuantizationConfig

    transformer = load_flux2_klein_transformer_from_gguf(
        gguf_path=gguf_path,
        base_model=base_model,
        torch_dtype=torch_dtype,
        device=device,
    )

    print(f"Loading VAE / text encoder / tokenizer from {base_model} ...")
    # Prefer pipeline factory that injects our transformer
    try:
        pipe = Flux2KleinPipeline.from_pretrained(
            base_model,
            transformer=transformer,
            torch_dtype=torch_dtype,
        )
    except TypeError:
        pipe = Flux2KleinPipeline.from_pretrained(base_model, torch_dtype=torch_dtype)
        pipe.transformer = transformer

    if cpu_offload and device.startswith("cuda"):
        print("Enabling model CPU offload (saves VRAM)")
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    # memory helpers if present
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
        try:
            pipe.vae.enable_slicing()
        except Exception:
            pass

    return pipe

def run_flux2_klein_inference(
    gguf_path: str,
    prompt: str,
    negative_prompt: str = "",
    cfg_scale: float = 4.0,
    num_inference_steps: int = 8,
    seed: int = 42,
    height: int = 512,
    width: int = 512,
    dtype: str = "bfloat16",
    device: str = "cuda",
    base_model: str = "black-forest-labs/FLUX.2-klein-4B",
    output_png: str = "output.png",
    cpu_offload: bool = True,
    max_sequence_length: int = 512,
) -> Dict[str, Any]:
    """
    End-to-end real PNG generation for Flux.2 Klein GGUF transformer.
    """
    require_diffusers()

    torch_dtype = resolve_dtype(dtype)
    # Klein is designed around bf16; force warning on fp16 GPU
    if torch_dtype == torch.float16 and device.startswith("cuda"):
        print(
            "[warn] float16 on Flux.2 Klein can be unstable; prefer --dtype bfloat16"
        )

    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA not available → cpu")
        device = "cpu"
        cpu_offload = False

    print("-" * 68)
    print("Flux.2 Klein REAL inference (diffusers + GGUF transformer)")
    print(f"  gguf            : {gguf_path}")
    print(f"  base_model      : {base_model}")
    print(f"  prompt          : {prompt!r}")
    print(f"  negative_prompt : {negative_prompt!r}")
    print(f"  cfg_scale       : {cfg_scale}")
    print(f"  steps           : {num_inference_steps}")
    print(f"  size            : {height}x{width}")
    print(f"  dtype / device  : {torch_dtype} @ {device}")
    print(f"  output          : {output_png}")
    print("-" * 68)

    # Show quant mix (does not load full weights twice in a wasteful way—just header)
    meta = parse_gguf_metadata(gguf_path)
    print_gguf_summary(meta)

    pipe = load_flux2_klein_pipeline_with_gguf_transformer(
        gguf_path=gguf_path,
        base_model=base_model,
        torch_dtype=torch_dtype,
        device=device,
        cpu_offload=cpu_offload,
    )

    generator = torch.Generator(device="cpu").manual_seed(seed)

    # Distilled Klein often runs good images in 4 steps; CFG may be ignored if distilled
    call_kwargs = dict(
        prompt=prompt,
        height=height,
        width=width,
        guidance_scale=cfg_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
        max_sequence_length=max_sequence_length,
    )
    # negative prompt only if the pipeline supports it
    try:
        import inspect
        sig = inspect.signature(pipe.__call__)
        if "negative_prompt" in sig.parameters and negative_prompt:
            call_kwargs["negative_prompt"] = negative_prompt
        elif negative_prompt and abs(cfg_scale - 1.0) > 1e-6:
            print(
                "[info] This pipeline build may ignore negative_prompt "
                "(common for distilled Klein)."
            )
    except Exception:
        pass

    print("Running denoising ...")
    with torch.inference_mode():
        out = pipe(**call_kwargs)

    image = out.images[0]
    path = save_pil_image(image, output_png)
    print(f"Wrote REAL image → {path}")

    return {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "cfg_scale": cfg_scale,
        "num_inference_steps": num_inference_steps,
        "seed": seed,
        "height": height,
        "width": width,
        "dtype": str(torch_dtype),
        "device": device,
        "base_model": base_model,
        "output_png": path,
        "meta": {
            "architecture": meta.get("architecture"),
            "qtype_counts": meta.get("qtype_counts"),
            "bit_counts": meta.get("bit_counts"),
            "n_tensors": len(meta.get("tensors", [])),
        },
        "mode": "flux2_klein_diffusers",
    }

# =============================================================================
# Fallback: pure dequant demo (noise PNG) — opt-in only
# =============================================================================

def run_demo_noise_png(
    gguf_path: str,
    prompt: str,
    output_png: str,
    height: int,
    width: int,
    seed: int,
    dtype: str,
    device: str,
) -> Dict[str, Any]:
    """
    OPT-IN smoke test only. Dequants GGUF, runs a tiny random net, writes static.
    This is WHAT YOU SAW BEFORE — not image generation.
    """
    print("!" * 68)
    print("DEMO MODE: will produce colorful noise, NOT a real photo of your prompt.")
    print("The GGUF has no VAE / text encoder / Flux DiT graph in this path.")
    print("Use the default (non --demo) path for real Flux.2 Klein images.")
    print("!" * 68)

    torch_dtype = resolve_dtype(dtype)
    torch_device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    sd, meta = build_state_dict(gguf_path, dtype=torch_dtype, device=torch.device("cpu"))

    g = torch.Generator(device="cpu").manual_seed(seed)
    # just decode a deterministic colorful field so the file is obviously "demo"
    y = torch.linspace(-1, 1, height).view(height, 1)
    x = torch.linspace(-1, 1, width).view(1, width)
    r = 0.5 + 0.5 * torch.sin(8 * x + seed)
    gr = 0.5 + 0.5 * torch.sin(8 * y + seed * 0.3)
    b = 0.5 + 0.5 * torch.sin(8 * (x + y) + len(prompt))
    rgb = torch.stack([r.expand(height, width), gr.expand(height, width), b.expand(height, width)], dim=-1)
    rgb = (rgb.clamp(0, 1).numpy() * 255).astype(np.uint8)
    # mix high-freq noise so it looks like the previous static warning pattern
    noise = (torch.rand(height, width, 3, generator=g).numpy() * 255).astype(np.uint8)
    rgb = (0.35 * rgb + 0.65 * noise).astype(np.uint8)
    path = write_png_numpy(output_png, rgb)

    return {
        "mode": "demo_noise",
        "output_png": path,
        "prompt": prompt,
        "warning": "Demo only — not conditioned generation",
        "meta": meta,
        "n_tensors_dequantized": len(sd),
    }

# =============================================================================
# CLI
# =============================================================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Flux.2 Klein GGUF → PNG. Default = REAL diffusers pipeline. "
            "Use --demo only to smoke-test dequant (noise image)."
        )
    )
    p.add_argument("gguf_path", type=str, help="Path to flux-2-klein-*-Q*.gguf")
    p.add_argument(
        "--prompt",
        type=str,
        default="a red fox in deep snow, cinematic lighting",
    )
    p.add_argument(
        "--negative-prompt",
        type=str,
        default="blurry, watermark, text, low quality",
    )
    p.add_argument("--cfg-scale", type=float, default=4.0,
                   help="Guidance scale (Klein distilled may use 1–4)")
    p.add_argument("--steps", type=int, default=8, dest="num_inference_steps",
                   help="Denoise steps (Klein often good at 4–8)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
        help="Compute dtype (bfloat16 recommended for Klein)",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--base-model",
        type=str,
        default="black-forest-labs/FLUX.2-klein-4B",
        help="HF id or path for VAE + Qwen3 text encoder + Klein config",
    )
    p.add_argument("--output", type=str, default="output.png", dest="output_png")
    p.add_argument(
        "--no-cpu-offload",
        action="store_true",
        help="Keep full pipeline on GPU (needs more VRAM)",
    )
    p.add_argument("--max-sequence-length", type=int, default=512)
    p.add_argument("--meta-only", action="store_true", help="Print GGUF metadata and exit")
    p.add_argument(
        "--dequant-only",
        action="store_true",
        help="Fully dequantize GGUF to RAM and exit (no image)",
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help="OPT-IN: write a noise PNG with pure torch dequant smoke test "
             "(NOT real image generation)",
    )
    return p

def main(argv: Optional[List[str]] = None) -> None:
    args = build_argparser().parse_args(argv)

    if args.meta_only:
        meta = parse_gguf_metadata(args.gguf_path)
        print_gguf_summary(meta)
        return

    if args.dequant_only:
        dtype = resolve_dtype(args.dtype)
        sd, meta = build_state_dict(args.gguf_path, dtype=dtype, device=torch.device("cpu"))
        n = sum(v.numel() for v in sd.values())
        print(f"Dequant complete: {len(sd)} tensors, {n:,} params")
        return

    if args.demo:
        result = run_demo_noise_png(
            gguf_path=args.gguf_path,
            prompt=args.prompt,
            output_png=args.output_png,
            height=args.height,
            width=args.width,
            seed=args.seed,
            dtype=args.dtype,
            device=args.device,
        )
        print(result)
        return

    # ---- REAL path ----
    result = run_flux2_klein_inference(
        gguf_path=args.gguf_path,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        cfg_scale=args.cfg_scale,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        height=args.height,
        width=args.width,
        dtype=args.dtype,
        device=args.device,
        base_model=args.base_model,
        output_png=args.output_png,
        cpu_offload=not args.no_cpu_offload,
        max_sequence_length=args.max_sequence_length,
    )

    print("\nSummary")
    print(f"  mode     : {result['mode']}")
    print(f"  prompt   : {result['prompt']!r}")
    print(f"  steps/cfg: {result['num_inference_steps']} / {result['cfg_scale']}")
    print(f"  PNG      : {result['output_png']}")
    print(f"  quants   : {result['meta'].get('qtype_counts')}")

if __name__ == "__main__":
    main()