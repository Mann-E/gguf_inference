# gguf_inference.py
# =============================================================================
# Flux.2 Klein — fully quantized consumer-GPU text-to-image (single file)
#
# Stack (matches Unsloth + city96 ComfyUI-GGUF practices):
#   1. DiT transformer ........ GGUF  (Q2/Q3/Q4/Q5/Q6/Q8/K — stays quantized on GPU
#                                      via diffusers GGUFQuantizationConfig)
#   2. Text encoder ........... Qwen3 GGUF  OR  bitsandbytes 4/8-bit
#                                      (Klein-4B → Qwen3-4B, Klein-9B → Qwen3-8B)
#   3. VAE .................... flux2-vae.safetensors (~336 MB, Unsloth/Comfy
#                                      standard — VAE has no useful public GGUF;
#                                      kept fp16 + sliced/tilled + offloaded)
#   4. Tokenizer / scheduler .. tiny configs only (not weight checkpoints)
#
# Why NOT load black-forest-labs/* full weights:
#   Full Qwen3-4B alone is ~8 GB fp16. Consumer path never downloads those.
#
# Install:
#   pip install -U torch numpy gguf "diffusers>=0.36" transformers accelerate \
#     bitsandbytes safetensors pillow huggingface_hub sentencepiece protobuf
#
# Recommended downloads (Unsloth / Comfy-Org):
#   DiT:  unsloth/FLUX.2-klein-4B-GGUF          (e.g. flux-2-klein-4B-Q4_K_M.gguf)
#   TE:   unsloth/Qwen3-4B-GGUF                 (e.g. Qwen3-4B-Q4_K_M.gguf)
#   VAE:  Comfy-Org/flux2-dev  or
#         Comfy-Org/vae-text-encorder-for-flux-klein-4b
#         → split_files/vae/flux2-vae.safetensors
#
# Example:
#   python gguf_inference.py flux-2-klein-4b-Q4_0.gguf \
#     --text-encoder-gguf Qwen3-4B-Q4_K_M.gguf \
#     --vae flux2-vae.safetensors \
#     --prompt "a red fox in deep snow, cinematic lighting" \
#     --cfg-scale 1.0 --steps 12 \
#     --height 512 --width 512 \
#     --dtype bfloat16 --device cuda \
#     --output fox.png
#
# Auto-fetch missing TE/VAE from Unsloth/Comfy defaults:
#   python gguf_inference.py flux-2-klein-4b-Q4_0.gguf --download-missing ...
# =============================================================================

from __future__ import annotations

import argparse
import gc
import inspect
import os
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# =============================================================================
# Defaults (Unsloth + Comfy-Org, consumer) — Klein 4B
# =============================================================================

DEFAULT_BASE_CONFIG = "black-forest-labs/FLUX.2-klein-4B"  # configs only
DEFAULT_DIT_REPO = "unsloth/FLUX.2-klein-4B-GGUF"
DEFAULT_TE_REPO = "unsloth/Qwen3-4B-GGUF"
DEFAULT_TE_BNB = "Qwen/Qwen3-4B"  # NF4 quant fall-back (no GGUF file needed)
DEFAULT_VAE_REPO = "Comfy-Org/vae-text-encorder-for-flux-klein-4b"
DEFAULT_VAE_FILE = "split_files/vae/flux2-vae.safetensors"
DEFAULT_VAE_REPO_ALT = "Comfy-Org/flux2-dev"
DEFAULT_VAE_FILE_ALT = "split_files/vae/flux2-vae.safetensors"

# Prefer these quant filenames when --download-missing (first hit wins)
TE_GGUF_CANDIDATES = [
    "Qwen3-4B-Q4_K_M.gguf",
    "Qwen3-4B-Q4_K_S.gguf",
    "Qwen3-4B-Q4_0.gguf",
    "Qwen3-4B-Q5_K_M.gguf",
    "Qwen3-4B-Q8_0.gguf",
    "Qwen3-4B-UD-Q4_K_XL.gguf",
    "Qwen3-4B.Q4_K_M.gguf",
]

DIT_GGUF_HINT = "flux-2-klein-4B-Q4_K_M.gguf / flux-2-klein-4b-Q4_0.gguf"

# =============================================================================
# Small utils
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

def free_mem() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def require(pkg_hint: str, err: Exception) -> None:
    raise ImportError(
        f"Missing dependency for consumer-GGUF path.\n"
        f"  pip install -U {pkg_hint}\n"
        f"Original error: {err}"
    ) from err

def file_exists(path: Optional[str]) -> bool:
    return bool(path) and os.path.isfile(path)

def print_banner(lines: List[str]) -> None:
    print("-" * 70)
    for ln in lines:
        print(ln)
    print("-" * 70)

# =============================================================================
# Optional GGUF metadata (DiT inspection only)
# =============================================================================

def parse_gguf_meta(path: str) -> Dict[str, Any]:
    import gguf

    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    reader = gguf.GGUFReader(path)
    arch = None
    field = reader.get_field("general.architecture")
    if field is not None:
        try:
            arch = str(field.parts[field.data[-1]], encoding="utf-8")
        except Exception:
            arch = None
    counts: Dict[str, int] = {}
    for t in reader.tensors:
        name = getattr(t.tensor_type, "name", str(t.tensor_type))
        counts[name] = counts.get(name, 0) + 1
    return {
        "path": path,
        "architecture": arch,
        "n_tensors": len(reader.tensors),
        "qtype_counts": counts,
    }

def print_gguf_meta(meta: Dict[str, Any]) -> None:
    print("=" * 70)
    print(f"GGUF         : {meta['path']}")
    print(f"Architecture : {meta.get('architecture')}")
    print(f"Tensors      : {meta.get('n_tensors')}")
    print(
        "Quants       : "
        + ", ".join(f"{k}x{v}" for k, v in meta.get("qtype_counts", {}).items())
    )
    print("=" * 70)

# =============================================================================
# HF download helpers (only quantized / tiny assets)
# =============================================================================

def hf_download(repo_id: str, filename: str, dest_dir: str = "models") -> str:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        require("huggingface_hub", e)

    os.makedirs(dest_dir, exist_ok=True)
    print(f"[download] {repo_id} :: {filename}")
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=dest_dir,
        local_dir_use_symlinks=False,
    )
    return path

def hf_download_first(repo_id: str, candidates: List[str], dest_dir: str) -> str:
    last_err: Optional[Exception] = None
    for name in candidates:
        try:
            return hf_download(repo_id, name, dest_dir=dest_dir)
        except Exception as e:
            last_err = e
            print(f"[download] miss {name}: {e}")
    raise FileNotFoundError(
        f"Could not download any of {candidates} from {repo_id}: {last_err}"
    )

def ensure_vae(path: Optional[str], download: bool, dest_dir: str) -> str:
    if file_exists(path):
        return path  # type: ignore[return-value]
    if not download:
        raise FileNotFoundError(
            "VAE not found. Pass --vae flux2-vae.safetensors or --download-missing.\n"
            f"  Expected ~336MB Flux2 VAE (Unsloth/Comfy standard).\n"
            f"  Repo: {DEFAULT_VAE_REPO}"
        )
    try:
        return hf_download(DEFAULT_VAE_REPO, DEFAULT_VAE_FILE, dest_dir=dest_dir)
    except Exception as e1:
        print(f"[download] primary VAE repo failed: {e1}")
        return hf_download(DEFAULT_VAE_REPO_ALT, DEFAULT_VAE_FILE_ALT, dest_dir=dest_dir)

def ensure_text_encoder_gguf(
    path: Optional[str], download: bool, dest_dir: str
) -> Optional[str]:
    if file_exists(path):
        return path
    if not download:
        return path  # may be None → bnb path later
    return hf_download_first(DEFAULT_TE_REPO, TE_GGUF_CANDIDATES, dest_dir=dest_dir)

# =============================================================================
# 1) DiT — GGUF (stays block-quantized on device)
# =============================================================================

def load_dit_gguf(
    gguf_path: str,
    base_config: str,
    torch_dtype: torch.dtype,
):
    """
    city96 / Unsloth diffusion GGUF → Flux2Transformer2DModel.
    Weights remain low-bit; dequant happens per-forward (diffusers GGUF engine,
    derived from ComfyUI-GGUF math).
    """
    try:
        from diffusers import Flux2Transformer2DModel, GGUFQuantizationConfig
    except Exception as e:
        require("'diffusers>=0.36'", e)

    if not os.path.isfile(gguf_path):
        raise FileNotFoundError(gguf_path)

    print(f"[dit] loading GGUF (quantized in-memory): {gguf_path}")
    qcfg = GGUFQuantizationConfig(compute_dtype=torch_dtype)

    # Explicit config required for Klein (differs from Flux2-dev)
    transformer = Flux2Transformer2DModel.from_single_file(
        gguf_path,
        quantization_config=qcfg,
        torch_dtype=torch_dtype,
        config=base_config,
        subfolder="transformer",
    )
    print("[dit] ready (GGUF dynamic dequant)")
    return transformer

# =============================================================================
# 2) Text encoder — Qwen3 GGUF or bitsandbytes 4/8-bit
# =============================================================================

def load_tokenizer(base_config: str, text_encoder_gguf: Optional[str], te_bnb: str):
    try:
        from transformers import AutoTokenizer
    except Exception as e:
        require("transformers", e)

    # Prefer chat template from base config tokenizer if available
    try:
        tok = AutoTokenizer.from_pretrained(base_config, subfolder="tokenizer")
        print(f"[tok] from {base_config}/tokenizer")
        return tok
    except Exception:
        pass

    if file_exists(text_encoder_gguf):
        # transformers can materialize tokenizer from GGUF metadata
        repo_or_dir = os.path.dirname(os.path.abspath(text_encoder_gguf)) or "."
        fname = os.path.basename(text_encoder_gguf)
        try:
            tok = AutoTokenizer.from_pretrained(repo_or_dir, gguf_file=fname)
            print(f"[tok] from GGUF metadata: {fname}")
            return tok
        except Exception as e:
            print(f"[tok] GGUF meta failed ({e}); falling back to {te_bnb}")
    tok = AutoTokenizer.from_pretrained(te_bnb)
    print(f"[tok] from {te_bnb}")
    return tok

def load_text_encoder_gguf(
    gguf_path: str,
    torch_dtype: torch.dtype,
    device_map: str = "cpu",
):
    """
    Load Qwen3 GGUF via transformers.
    Note: HF currently dequantizes GGUF → dense weights on load; we keep it on CPU
    and use pipeline CPU offload so VRAM stays free. System RAM needed ≈ dequant size.
    """
    try:
        from transformers import AutoModelForCausalLM
    except Exception as e:
        require("transformers", e)

    repo_or_dir = os.path.dirname(os.path.abspath(gguf_path)) or "."
    fname = os.path.basename(gguf_path)
    print(f"[te] loading Qwen3 GGUF → dense for pipeline: {gguf_path}")
    print("[te] (ComfyUI-GGUF keeps runtime quant; HF dequants — we offload to CPU)")

    model = AutoModelForCausalLM.from_pretrained(
        repo_or_dir,
        gguf_file=fname,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model

def load_text_encoder_bnb(
    model_id: str,
    torch_dtype: torch.dtype,
    bits: int = 4,
    device: str = "cuda",
):
    """
    True consumer-VRAM path: Qwen3 held as NF4 / int8 via bitsandbytes.
    Prefer this when you don't want GGUF dequant RAM spike.
    """
    try:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    except Exception as e:
        require("transformers bitsandbytes", e)

    print(f"[te] loading bitsandbytes {bits}-bit: {model_id}")
    if bits == 4:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        bnb = BitsAndBytesConfig(load_in_8bit=True)

    # bitsandbytes usually wants the model on GPU
    device_map = "auto" if str(device).startswith("cuda") else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map=device_map,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model

def load_text_encoder(
    text_encoder_gguf: Optional[str],
    te_mode: str,
    te_bnb_id: str,
    torch_dtype: torch.dtype,
    device: str,
):
    """
    te_mode:
      - auto  : GGUF if path given else bnb4
      - gguf  : require GGUF file
      - bnb4  : bitsandbytes NF4
      - bnb8  : bitsandbytes int8
    """
    mode = te_mode.lower()
    if mode == "auto":
        mode = "gguf" if file_exists(text_encoder_gguf) else "bnb4"

    if mode == "gguf":
        if not file_exists(text_encoder_gguf):
            raise FileNotFoundError(
                "te_mode=gguf requires --text-encoder-gguf / --download-missing"
            )
        # stay on CPUMmap; pipeline offload moves it for encode only
        return load_text_encoder_gguf(
            text_encoder_gguf, torch_dtype=torch_dtype, device_map="cpu"
        ), "gguf-dequant+cpu-offload"

    if mode == "bnb4":
        return (
            load_text_encoder_bnb(te_bnb_id, torch_dtype, bits=4, device=device),
            "bnb4-nf4",
        )
    if mode == "bnb8":
        return (
            load_text_encoder_bnb(te_bnb_id, torch_dtype, bits=8, device=device),
            "bnb8",
        )

    raise ValueError(f"Unknown --te-mode {te_mode}")

# =============================================================================
# 3) VAE — artifacts of Unsloth/Comfy (small safetensors)
# =============================================================================

def load_vae_flux2(vae_path: str, torch_dtype: torch.dtype):
    """
    Flux2 VAE is ~336 MB. Unsloth docs: DiT+TE as GGUF, VAE as safetensors.
    No widely used Flux2 VAE GGUF exists; we keep fp16 + slicing (consumer-safe).
    """
    try:
        from diffusers import AutoencoderKLFlux2
    except Exception:
        try:
            # older naming fallback
            from diffusers import AutoencoderKL as AutoencoderKLFlux2  # type: ignore
        except Exception as e:
            require("'diffusers>=0.36' (AutoencoderKLFlux2)", e)

    print(f"[vae] loading Flux2 VAE safetensors: {vae_path}")
    # Prefer single-file load (Comfy layout)
    vae = None
    err: Optional[Exception] = None
    try:
        vae = AutoencoderKLFlux2.from_single_file(vae_path, torch_dtype=torch_dtype)
    except Exception as e:
        err = e
        print(f"[vae] from_single_file failed ({e}); trying generic paths...")

    if vae is None:
        try:
            from diffusers import AutoencoderKL

            vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=torch_dtype)
        except Exception as e2:
            raise RuntimeError(
                f"Could not load VAE from {vae_path}.\n"
                f"first={err}\nsecond={e2}\n"
                "Download flux2-vae.safetensors (Comfy-Org)."
            ) from e2

    # consumer decode helpers
    if hasattr(vae, "enable_slicing"):
        try:
            vae.enable_slicing()
        except Exception:
            pass
    if hasattr(vae, "enable_tiling"):
        try:
            vae.enable_tiling()
        except Exception:
            pass

    vae.to(dtype=torch_dtype)
    vae.eval()
    print("[vae] ready (fp16/bf16 safetensors, sliced)")
    return vae

# =============================================================================
# 4) Scheduler from config only
# =============================================================================

def load_scheduler(base_config: str):
    try:
        from diffusers import FlowMatchEulerDiscreteScheduler
    except Exception as e:
        require("diffusers", e)
    print(f"[sched] {base_config}/scheduler")
    return FlowMatchEulerDiscreteScheduler.from_pretrained(
        base_config, subfolder="scheduler"
    )

# =============================================================================
# Assemble pipeline WITHOUT pulling full DiT/TE weights
# =============================================================================

def build_consumer_pipeline(
    dit_gguf: str,
    vae_path: str,
    text_encoder_gguf: Optional[str],
    te_mode: str,
    te_bnb_id: str,
    base_config: str,
    torch_dtype: torch.dtype,
    device: str,
    cpu_offload: bool,
    is_distilled: bool = True,
):
    try:
        from diffusers import Flux2KleinPipeline
    except Exception as e:
        require("'diffusers>=0.36' (Flux2KleinPipeline)", e)

    # Order chosen for RAM peaks: TE first (may be heavy if GGUF dequant), then DiT GGUF, then VAE
    text_encoder, te_label = load_text_encoder(
        text_encoder_gguf=text_encoder_gguf,
        te_mode=te_mode,
        te_bnb_id=te_bnb_id,
        torch_dtype=torch_dtype,
        device=device,
    )
    free_mem()

    tokenizer = load_tokenizer(base_config, text_encoder_gguf, te_bnb_id)
    scheduler = load_scheduler(base_config)
    free_mem()

    transformer = load_dit_gguf(dit_gguf, base_config, torch_dtype)
    free_mem()

    vae = load_vae_flux2(vae_path, torch_dtype)
    free_mem()

    print("[pipe] assembling Flux2KleinPipeline (no full-weight base download)")
    try:
        pipe = Flux2KleinPipeline(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            is_distilled=is_distilled,
        )
    except TypeError:
        # older signature without is_distilled
        pipe = Flux2KleinPipeline(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
        )

    # VRAM strategy: Comfy-like sequential residency (TE → DiT → VAE)
    if str(device).startswith("cuda") and torch.cuda.is_available():
        if cpu_offload:
            print("[pipe] enable_model_cpu_offload (consumer VRAM)")
            try:
                pipe.enable_model_cpu_offload()
            except Exception as e:
                print(f"[pipe] cpu_offload failed ({e}); trying sequential")
                try:
                    pipe.enable_sequential_cpu_offload()
                except Exception as e2:
                    print(f"[pipe] sequential failed ({e2}); .to(cuda)")
                    pipe.to(device)
        else:
            pipe.to(device)
    else:
        pipe.to("cpu")

    return pipe, te_label

# =============================================================================
# Inference
# =============================================================================

def run_inference(
    dit_gguf: str,
    prompt: str,
    negative_prompt: str = "",
    cfg_scale: float = 1.0,
    num_inference_steps: int = 12,
    seed: int = 42,
    height: int = 512,
    width: int = 512,
    dtype: str = "bfloat16",
    device: str = "cuda",
    output_png: str = "fox.png",
    base_config: str = DEFAULT_BASE_CONFIG,
    text_encoder_gguf: Optional[str] = None,
    vae_path: Optional[str] = None,
    te_mode: str = "auto",
    te_bnb_id: str = DEFAULT_TE_BNB,
    download_missing: bool = False,
    models_dir: str = "models",
    cpu_offload: bool = True,
    is_distilled: bool = True,
    max_sequence_length: int = 512,
) -> Dict[str, Any]:
    torch_dtype = resolve_dtype(dtype)

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA not available → cpu")
        device = "cpu"
        cpu_offload = False

    if torch_dtype == torch.float16 and str(device).startswith("cuda"):
        print("[warn] prefer --dtype bfloat16 for Flux.2 Klein stability")

    # Resolve paths (optionally download TE/VAE quants / tiny VAE only — never full DiT)
    vae_path = ensure_vae(vae_path, download=download_missing, dest_dir=models_dir)
    text_encoder_gguf = ensure_text_encoder_gguf(
        text_encoder_gguf, download=download_missing, dest_dir=models_dir
    )

    if not os.path.isfile(dit_gguf):
        raise FileNotFoundError(
            f"DiT GGUF not found: {dit_gguf}\n"
            f"Download from {DEFAULT_DIT_REPO} (e.g. {DIT_GGUF_HINT})"
        )

    meta = parse_gguf_meta(dit_gguf)
    print_banner(
        [
            "Flux.2 Klein CONSUMER-GPU inference (all heavy weights quantized)",
            f"  DiT GGUF     : {dit_gguf}",
            f"  TE mode      : {te_mode}  file={text_encoder_gguf}",
            f"  TE bnb id    : {te_bnb_id}",
            f"  VAE          : {vae_path}",
            f"  base config  : {base_config} (configs only)",
            f"  prompt       : {prompt!r}",
            f"  negative     : {negative_prompt!r}",
            f"  cfg / steps  : {cfg_scale} / {num_inference_steps}",
            f"  size         : {height}x{width}",
            f"  dtype/device : {torch_dtype} @ {device}  offload={cpu_offload}",
            f"  output       : {output_png}",
        ]
    )
    print_gguf_meta(meta)

    pipe, te_label = build_consumer_pipeline(
        dit_gguf=dit_gguf,
        vae_path=vae_path,
        text_encoder_gguf=text_encoder_gguf,
        te_mode=te_mode,
        te_bnb_id=te_bnb_id,
        base_config=base_config,
        torch_dtype=torch_dtype,
        device=device,
        cpu_offload=cpu_offload,
        is_distilled=is_distilled,
    )

    generator = torch.Generator(device="cpu").manual_seed(seed)

    call_kwargs: Dict[str, Any] = dict(
        prompt=prompt,
        height=height,
        width=width,
        guidance_scale=cfg_scale,
        num_inference_steps=num_inference_steps,
        generator=generator,
    )

    # max_sequence_length / negative_prompt only if supported
    try:
        sig = inspect.signature(pipe.__call__)
        params = sig.parameters
        if "max_sequence_length" in params:
            call_kwargs["max_sequence_length"] = max_sequence_length
        if "negative_prompt" in params and negative_prompt:
            call_kwargs["negative_prompt"] = negative_prompt
        elif negative_prompt and abs(cfg_scale - 1.0) > 1e-6:
            print(
                "[info] pipeline ignores negative_prompt "
                "(distilled Klein often has CFG baked in)"
            )
    except Exception:
        pass

    print("[run] denoising ...")
    with torch.inference_mode():
        out = pipe(**call_kwargs)
    image = out.images[0]

    parent = os.path.dirname(os.path.abspath(output_png))
    if parent:
        os.makedirs(parent, exist_ok=True)
    image.save(output_png)
    print(f"[run] wrote REAL image → {output_png}")

    return {
        "mode": "consumer_gguf_stack",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "cfg_scale": cfg_scale,
        "num_inference_steps": num_inference_steps,
        "seed": seed,
        "height": height,
        "width": width,
        "dtype": str(torch_dtype),
        "device": device,
        "dit_gguf": dit_gguf,
        "text_encoder": te_label,
        "text_encoder_gguf": text_encoder_gguf,
        "vae": vae_path,
        "output_png": output_png,
        "dit_meta": meta,
    }

# =============================================================================
# CLI
# =============================================================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Flux.2 Klein consumer-GPU GGUF runner. "
            "DiT=GGUF · TE=Qwen3 GGUF|bnb4 · VAE=flux2 safetensors (Unsloth/Comfy)."
        )
    )
    p.add_argument(
        "dit_gguf",
        type=str,
        help=f"Path to DiT GGUF (Unsloth Klein). e.g. {DIT_GGUF_HINT}",
    )
    p.add_argument(
        "--text-encoder-gguf",
        type=str,
        default="",
        help="Path to Qwen3-4B GGUF (unsloth/Qwen3-4B-GGUF). Optional if --te-mode bnb4.",
    )
    p.add_argument(
        "--vae",
        type=str,
        default="",
        help="Path to flux2-vae.safetensors (~336MB). Optional with --download-missing.",
    )
    p.add_argument(
        "--te-mode",
        type=str,
        default="auto",
        choices=["auto", "gguf", "bnb4", "bnb8"],
        help="Text encoder quant mode (default auto: GGUF if provided else bnb4)",
    )
    p.add_argument(
        "--te-bnb",
        type=str,
        default=DEFAULT_TE_BNB,
        help="HF id for bitsandbytes TE (default Qwen/Qwen3-4B)",
    )
    p.add_argument(
        "--base-config",
        type=str,
        default=DEFAULT_BASE_CONFIG,
        help="HF id used ONLY for scheduler/tokenizer/transformer config",
    )
    p.add_argument(
        "--download-missing",
        action="store_true",
        help="Auto-download Qwen3 GGUF + flux2 VAE from Unsloth/Comfy-Org",
    )
    p.add_argument("--models-dir", type=str, default="models")
    p.add_argument("--prompt", type=str, default="a red fox in deep snow, cinematic lighting")
    p.add_argument(
        "--negative-prompt",
        type=str,
        default="blurry, watermark, text, low quality",
    )
    p.add_argument("--cfg-scale", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=12, dest="num_inference_steps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "fp16", "bfloat16", "bf16", "float32", "fp32"],
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output", type=str, default="fox.png", dest="output_png")
    p.add_argument(
        "--no-cpu-offload",
        action="store_true",
        help="Keep modules on GPU (needs more VRAM)",
    )
    p.add_argument(
        "--not-distilled",
        action="store_true",
        help="Set is_distilled=False (base models)",
    )
    p.add_argument("--max-sequence-length", type=int, default=512)
    p.add_argument("--meta-only", action="store_true")
    return p

def main(argv: Optional[List[str]] = None) -> None:
    args = build_argparser().parse_args(argv)

    if args.meta_only:
        print_gguf_meta(parse_gguf_meta(args.dit_gguf))
        return

    te_gguf = args.text_encoder_gguf or None
    vae = args.vae or None

    result = run_inference(
        dit_gguf=args.dit_gguf,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        cfg_scale=args.cfg_scale,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        height=args.height,
        width=args.width,
        dtype=args.dtype,
        device=args.device,
        output_png=args.output_png,
        base_config=args.base_config,
        text_encoder_gguf=te_gguf,
        vae_path=vae,
        te_mode=args.te_mode,
        te_bnb_id=args.te_bnb,
        download_missing=args.download_missing,
        models_dir=args.models_dir,
        cpu_offload=not args.no_cpu_offload,
        is_distilled=not args.not_distilled,
        max_sequence_length=args.max_sequence_length,
    )

    print("\nSummary")
    for k in (
        "mode",
        "output_png",
        "dit_gguf",
        "text_encoder",
        "text_encoder_gguf",
        "vae",
        "cfg_scale",
        "num_inference_steps",
        "dtype",
        "device",
    ):
        print(f"  {k:18s}: {result.get(k)}")
    print(f"  dit quants        : {result['dit_meta'].get('qtype_counts')}")

if __name__ == "__main__":
    main()