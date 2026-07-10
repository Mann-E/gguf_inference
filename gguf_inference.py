# gguf_inference.py
# =============================================================================
# Flux.2 Klein consumer-GPU runner (single file)
#
# Model-type–aware VAE loading  ← this fixed your last crash
#   flux2_*  → AutoencoderKLFlux2  (32-ch latents; conv_out = 64)
#              NEVER AutoencoderKL / from_single_file for Flux2 VAE
#   Options:  (1) from_pretrained(base, subfolder="vae")
#             (2) local safetensors → Flux2 config + load_state_dict
#                 (+ BFL→diffusers key remap if needed)
#             (3) small-decoder: black-forest-labs/FLUX.2-small-decoder
#
# Quantized consumer stack (Unsloth / ComfyUI-GGUF style):
#   DiT  = GGUF (Q2–Q8) via GGUFQuantizationConfig
#   TE   = Qwen3 GGUF | bitsandbytes NF4/int8
#   VAE  = Flux2 AutoencoderKLFlux2 (fp16/bf16; tiny vs DiT/TE)
#
# Install:
#   pip install -U torch numpy gguf "diffusers>=0.36" transformers accelerate \
#     bitsandbytes safetensors pillow huggingface_hub sentencepiece protobuf
#
# Run:
#   python gguf_inference.py flux-2-klein-4b-Q4_0.gguf \
#     --model-type flux2_klein_4b \
#     --download-missing \
#     --te-mode bnb4 \
#     --prompt "a red fox in deep snow, cinematic lighting" \
#     --cfg-scale 1.0 --steps 12 \
#     --height 512 --width 512 \
#     --dtype bfloat16 --device cuda \
#     --output fox.png
# =============================================================================

from __future__ import annotations

import argparse
import gc
import inspect
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch

# =============================================================================
# Model-type registry
# =============================================================================

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Distilled Klein 4B (your file)
    "flux2_klein_4b": {
        "family": "flux2",
        "base_config": "black-forest-labs/FLUX.2-klein-4B",
        "pipeline": "Flux2KleinPipeline",
        "transformer": "Flux2Transformer2DModel",
        "vae_class": "AutoencoderKLFlux2",
        "vae_subfolder": "vae",
        "vae_latent_channels": 32,
        "te_kind": "qwen3",
        "te_bnb": "Qwen/Qwen3-4B",
        "te_gguf_repo": "unsloth/Qwen3-4B-GGUF",
        "te_gguf_candidates": [
            "Qwen3-4B-Q4_K_M.gguf",
            "Qwen3-4B-Q4_K_S.gguf",
            "Qwen3-4B-Q4_0.gguf",
            "Qwen3-4B-Q5_K_M.gguf",
            "Qwen3-4B-UD-Q4_K_XL.gguf",
        ],
        "is_distilled": True,
        "default_cfg": 1.0,
        "default_steps": 8,
        "dit_config_subfolder": "transformer",
        "small_vae_repo": "black-forest-labs/FLUX.2-small-decoder",
        "comfy_vae_repos": [
            ("Comfy-Org/vae-text-encorder-for-flux-klein-4b", "split_files/vae/flux2-vae.safetensors"),
            ("Comfy-Org/flux2-dev", "split_files/vae/flux2-vae.safetensors"),
        ],
    },
    "flux2_klein_9b": {
        "family": "flux2",
        "base_config": "black-forest-labs/FLUX.2-klein-9B",
        "pipeline": "Flux2KleinPipeline",
        "transformer": "Flux2Transformer2DModel",
        "vae_class": "AutoencoderKLFlux2",
        "vae_subfolder": "vae",
        "vae_latent_channels": 32,
        "te_kind": "qwen3",
        "te_bnb": "Qwen/Qwen3-8B",
        "te_gguf_repo": "unsloth/Qwen3-8B-GGUF",
        "te_gguf_candidates": [
            "Qwen3-8B-Q4_K_M.gguf",
            "Qwen3-8B-Q4_0.gguf",
            "Qwen3-8B-Q5_K_M.gguf",
        ],
        "is_distilled": True,
        "default_cfg": 1.0,
        "default_steps": 8,
        "dit_config_subfolder": "transformer",
        "small_vae_repo": "black-forest-labs/FLUX.2-small-decoder",
        "comfy_vae_repos": [
            ("Comfy-Org/flux2-dev", "split_files/vae/flux2-vae.safetensors"),
        ],
    },
    "flux2_klein_base_4b": {
        "family": "flux2",
        "base_config": "black-forest-labs/FLUX.2-klein-base-4B",
        "pipeline": "Flux2KleinPipeline",
        "transformer": "Flux2Transformer2DModel",
        "vae_class": "AutoencoderKLFlux2",
        "vae_subfolder": "vae",
        "vae_latent_channels": 32,
        "te_kind": "qwen3",
        "te_bnb": "Qwen/Qwen3-4B",
        "te_gguf_repo": "unsloth/Qwen3-4B-GGUF",
        "te_gguf_candidates": ["Qwen3-4B-Q4_K_M.gguf", "Qwen3-4B-Q4_0.gguf"],
        "is_distilled": False,
        "default_cfg": 4.0,
        "default_steps": 28,
        "dit_config_subfolder": "transformer",
        "small_vae_repo": "black-forest-labs/FLUX.2-small-decoder",
        "comfy_vae_repos": [
            ("Comfy-Org/flux2-dev", "split_files/vae/flux2-vae.safetensors"),
        ],
    },
    "flux2_dev": {
        "family": "flux2",
        "base_config": "black-forest-labs/FLUX.2-dev",
        "pipeline": "Flux2Pipeline",
        "transformer": "Flux2Transformer2DModel",
        "vae_class": "AutoencoderKLFlux2",
        "vae_subfolder": "vae",
        "vae_latent_channels": 32,
        "te_kind": "mistral",  # full Flux2-dev uses Mistral; consumer usually remote / bnb
        "te_bnb": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "te_gguf_repo": None,
        "te_gguf_candidates": [],
        "is_distilled": True,
        "default_cfg": 4.0,
        "default_steps": 28,
        "dit_config_subfolder": "transformer",
        "small_vae_repo": "black-forest-labs/FLUX.2-small-decoder",
        "comfy_vae_repos": [
            ("Comfy-Org/flux2-dev", "split_files/vae/flux2-vae.safetensors"),
        ],
    },
}

ALIASES = {
    "klein": "flux2_klein_4b",
    "klein4b": "flux2_klein_4b",
    "klein-4b": "flux2_klein_4b",
    "flux2-klein-4b": "flux2_klein_4b",
    "flux2_klein": "flux2_klein_4b",
    "klein9b": "flux2_klein_9b",
    "klein-9b": "flux2_klein_9b",
    "flux2": "flux2_dev",
    "flux2-dev": "flux2_dev",
}


def resolve_model_type(name: str) -> str:
    key = name.strip().lower().replace(" ", "_")
    if key in MODEL_REGISTRY:
        return key
    if key in ALIASES:
        return ALIASES[key]
    raise ValueError(
        f"Unknown --model-type {name!r}. Choose one of: "
        + ", ".join(MODEL_REGISTRY.keys())
    )


def guess_model_type_from_path(path: str) -> str:
    p = os.path.basename(path).lower()
    if "klein" in p and "9b" in p:
        return "flux2_klein_9b"
    if "klein" in p and "base" in p:
        return "flux2_klein_base_4b"
    if "klein" in p:
        return "flux2_klein_4b"
    if "flux2" in p or "flux-2" in p:
        return "flux2_dev"
    return "flux2_klein_4b"


# =============================================================================
# Utils
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


def require(hint: str, err: Exception) -> None:
    raise ImportError(f"Missing dep.\n  pip install -U {hint}\nOriginal: {err}") from err


def file_ok(p: Optional[str]) -> bool:
    return bool(p) and os.path.isfile(p)


def print_banner(lines: List[str]) -> None:
    print("-" * 72)
    for ln in lines:
        print(ln)
    print("-" * 72)


# =============================================================================
# GGUF meta (DiT only)
# =============================================================================

def parse_gguf_meta(path: str) -> Dict[str, Any]:
    import gguf

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
        n = getattr(t.tensor_type, "name", str(t.tensor_type))
        counts[n] = counts.get(n, 0) + 1
    return {
        "path": path,
        "architecture": arch,
        "n_tensors": len(reader.tensors),
        "qtype_counts": counts,
    }


def print_gguf_meta(meta: Dict[str, Any]) -> None:
    print("=" * 72)
    print(f"GGUF         : {meta['path']}")
    print(f"Architecture : {meta.get('architecture')}")
    print(f"Tensors      : {meta.get('n_tensors')}")
    print(
        "Quants       : "
        + ", ".join(f"{k}x{v}" for k, v in meta.get("qtype_counts", {}).items())
    )
    print("=" * 72)


# =============================================================================
# HF download
# =============================================================================

def hf_download(repo_id: str, filename: str, dest_dir: str) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        require("huggingface_hub", e)
    os.makedirs(dest_dir, exist_ok=True)
    print(f"[download] {repo_id} :: {filename}")
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=dest_dir,
        local_dir_use_symlinks=False,
    )


def hf_download_first(repo_id: str, names: List[str], dest_dir: str) -> str:
    last = None
    for n in names:
        try:
            return hf_download(repo_id, n, dest_dir)
        except Exception as e:
            last = e
            print(f"[download] miss {n}: {e}")
    raise FileNotFoundError(f"None of {names} in {repo_id}: {last}")


# =============================================================================
# VAE key conversion (BFL / Comfy → diffusers AutoencoderKLFlux2)
# =============================================================================

def _is_bfl_vae_sd(keys) -> bool:
    return any(
        k.startswith("encoder.down.")
        or k.startswith("decoder.up.")
        or k.startswith("decoder.mid.block_")
        or k.startswith("encoder.mid.block_")
        or k.startswith("decoder.norm_out")
        or k.startswith("encoder.norm_out")
        for k in keys
    )


def convert_bfl_flux2_vae_to_diffusers(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    BFL / some Comfy dumps use encoder.down.X / decoder.up.X naming.
    Diffusers AutoencoderKLFlux2 expects down_blocks / up_blocks / mid_block.
    Adapted from InvokeAI Flux2VAELoader.
    """
    out: Dict[str, torch.Tensor] = {}

    def rename(k: str) -> str:
        k = k.replace("encoder.down.", "encoder.down_blocks.")
        k = k.replace("decoder.up.", "decoder.up_blocks.")
        k = k.replace(".block.", ".resnets.")
        k = k.replace(".nin_shortcut.", ".conv_shortcut.")
        k = k.replace("encoder.mid.block_1", "encoder.mid_block.resnets.0")
        k = k.replace("encoder.mid.block_2", "encoder.mid_block.resnets.1")
        k = k.replace("decoder.mid.block_1", "decoder.mid_block.resnets.0")
        k = k.replace("decoder.mid.block_2", "decoder.mid_block.resnets.1")
        k = k.replace("encoder.mid.attn_1.", "encoder.mid_block.attentions.0.")
        k = k.replace("decoder.mid.attn_1.", "decoder.mid_block.attentions.0.")
        k = k.replace(".q.weight", ".to_q.weight").replace(".q.bias", ".to_q.bias")
        k = k.replace(".k.weight", ".to_k.weight").replace(".k.bias", ".to_k.bias")
        k = k.replace(".v.weight", ".to_v.weight").replace(".v.bias", ".to_v.bias")
        k = k.replace(".proj_out.weight", ".to_out.0.weight").replace(
            ".proj_out.bias", ".to_out.0.bias"
        )
        k = k.replace("encoder.norm_out.", "encoder.conv_norm_out.")
        k = k.replace("decoder.norm_out.", "decoder.conv_norm_out.")
        k = k.replace("encoder.quant_conv.", "quant_conv.")
        k = k.replace("decoder.post_quant_conv.", "post_quant_conv.")
        # downsample
        k = re.sub(
            r"encoder\.down_blocks\.(\d+)\.downsample\.conv\.",
            r"encoder.down_blocks.\1.downsamplers.0.conv.",
            k,
        )
        k = re.sub(
            r"decoder\.up_blocks\.(\d+)\.upsample\.conv\.",
            r"decoder.up_blocks.\1.upsamplers.0.conv.",
            k,
        )
        return k

    for k, v in sd.items():
        out[rename(k)] = v

    # decoder up_blocks are reversed in BFL vs diffusers in some dumps — only if both exist
    # leave as-is if already converted names match AutoencoderKLFlux2
    return out


def peek_vae_latent_channels(sd: Dict[str, torch.Tensor]) -> Optional[int]:
    """Infer latent width from encoder.conv_out (2 * z_channels for mean+logvar)."""
    for key in (
        "encoder.conv_out.weight",
        "encoder.conv_out.bias",
        "quant_conv.weight",
    ):
        if key in sd:
            c = int(sd[key].shape[0])
            # mean+logvar → half is latent channels for KLVAEs
            if c % 2 == 0 and c >= 8:
                return c // 2
            return c
    return None


# =============================================================================
# VAE loaders (model-type aware)  ··· THE FIX
# =============================================================================

def get_autoencoder_kl_flux2():
    try:
        from diffusers import AutoencoderKLFlux2
        return AutoencoderKLFlux2
    except Exception as e:
        require("'diffusers>=0.36' with AutoencoderKLFlux2", e)


def load_vae_flux2_from_pretrained(
    repo_or_id: str,
    subfolder: Optional[str],
    torch_dtype: torch.dtype,
):
    AutoencoderKLFlux2 = get_autoencoder_kl_flux2()
    kwargs = dict(torch_dtype=torch_dtype)
    if subfolder:
        kwargs["subfolder"] = subfolder
    print(f"[vae] AutoencoderKLFlux2.from_pretrained({repo_or_id!r}, {kwargs})")
    vae = AutoencoderKLFlux2.from_pretrained(repo_or_id, **kwargs)
    return vae


def load_vae_flux2_from_safetensors(
    weights_path: str,
    config_source: str,
    config_subfolder: str,
    torch_dtype: torch.dtype,
    expected_latent_channels: int = 32,
):
    """
    Reliable path when from_single_file is broken for AutoencoderKLFlux2:
      1) load config from HF (or local) AutoencoderKLFlux2 config
      2) construct empty model
      3) load local safetensors weights (remap BFL keys if needed)
    """
    from safetensors.torch import load_file

    AutoencoderKLFlux2 = get_autoencoder_kl_flux2()

    print(f"[vae] loading weights: {weights_path}")
    sd = load_file(weights_path, device="cpu")

    # strip common prefixes
    cleaned = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("vae."):
            nk = nk[4:]
        if nk.startswith("model."):
            nk = nk[6:]
        cleaned[nk] = v
    sd = cleaned

    z = peek_vae_latent_channels(sd)
    if z is not None:
        print(f"[vae] detected latent channels ≈ {z} "
              f"(encoder.conv_out out_ch={z * 2 if z else '?'})")
        if z not in (expected_latent_channels, expected_latent_channels * 2):
            # *2 would mean we double-counted; just log
            print(
                f"[vae] note: expected_latent_channels={expected_latent_channels}, "
                f"inferred={z}"
            )

    if _is_bfl_vae_sd(sd.keys()):
        print("[vae] BFL/Comfy key layout → converting to diffusers")
        sd = convert_bfl_flux2_vae_to_diffusers(sd)

    # Config from model-type base (correct Flux2 VAE shapes: 32-ch, conv_out 64)
    print(f"[vae] config from {config_source} subfolder={config_subfolder}")
    try:
        vae = AutoencoderKLFlux2.from_pretrained(
            config_source,
            subfolder=config_subfolder,
            torch_dtype=torch_dtype,
            # low_cpu first with meta; then we load weights
        )
        # Replace weights with local file (Comfy dump may equal official ae)
        missing, unexpected = vae.load_state_dict(sd, strict=False)
        if missing:
            print(f"[vae] missing keys after load: {len(missing)} "
                  f"(sample {missing[:5]})")
        if unexpected:
            print(f"[vae] unexpected keys: {len(unexpected)} "
                  f"(sample {unexpected[:5]})")
        # If nearly everything missing, config+weights mismatch — rebuild empty
        n_params = sum(p.numel() for p in vae.parameters())
        if missing and len(missing) > 20:
            print("[vae] many missing keys — rebuild from config + assign")
            cfg = AutoencoderKLFlux2.load_config(
                config_source, subfolder=config_subfolder
            )
            vae = AutoencoderKLFlux2.from_config(cfg)
            # cast tensors
            sd16 = {k: v.to(torch_dtype) for k, v in sd.items()}
            missing, unexpected = vae.load_state_dict(sd16, strict=False, assign=True)
            print(f"[vae] reloaded strict=False missing={len(missing)} "
                  f"unexpected={len(unexpected)}")
    except Exception as e:
        print(f"[vae] from_pretrained(config) failed ({e}); from_config only")
        cfg = AutoencoderKLFlux2.load_config(
            config_source, subfolder=config_subfolder
        )
        vae = AutoencoderKLFlux2.from_config(cfg)
        sd16 = {k: v.to(torch_dtype) for k, v in sd.items()}
        missing, unexpected = vae.load_state_dict(sd16, strict=False, assign=True)
        print(f"[vae] load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
        if missing and len(missing) > max(5, len(sd) // 2):
            raise RuntimeError(
                "Local VAE weights do not match AutoencoderKLFlux2 config.\n"
                f"  weights: {weights_path}\n"
                f"  config:  {config_source}/{config_subfolder}\n"
                "Use --vae-source pretrained or --vae-source small-decoder."
            ) from e

    vae.to(dtype=torch_dtype)
    return vae


def enable_vae_consumer_opts(vae) -> None:
    for name in ("enable_slicing", "enable_tiling"):
        if hasattr(vae, name):
            try:
                getattr(vae, name)()
            except Exception:
                pass
    vae.eval()


def load_vae_for_model_type(
    model_type: str,
    vae_source: str,
    vae_path: Optional[str],
    torch_dtype: torch.dtype,
    download_missing: bool,
    models_dir: str,
):
    """
    vae_source:
      - auto         : local path if given, else pretrained subfolder, else small-decoder
      - pretrained   : base_config/vae from HF (correct shapes always)
      - small-decoder: black-forest-labs/FLUX.2-small-decoder (consumer-friendly)
      - local        : require --vae file; map into AutoencoderKLFlux2 via config
    """
    info = MODEL_REGISTRY[model_type]
    family = info["family"]

    if family != "flux2":
        raise NotImplementedError(
            f"family={family} not wired in this consumer script (Flux2 only)"
        )

    # Flux2 VAE hates float16 on some GPUs — prefer bf16/fp32 for the VAE module
    vae_dtype = torch_dtype
    if torch_dtype == torch.float16:
        print("[vae] float16 is unstable for Flux2 VAE → using bfloat16 for VAE")
        try:
            _ = torch.zeros(1, dtype=torch.bfloat16)
            vae_dtype = torch.bfloat16
        except Exception:
            vae_dtype = torch.float32

    source = vae_source.lower()
    base = info["base_config"]
    sub = info["vae_subfolder"]
    z_ch = info["vae_latent_channels"]

    # Resolve auto
    if source == "auto":
        if file_ok(vae_path):
            source = "local"
        else:
            source = "pretrained"

    if source == "small-decoder":
        repo = info.get("small_vae_repo") or "black-forest-labs/FLUX.2-small-decoder"
        vae = load_vae_flux2_from_pretrained(repo, subfolder=None, torch_dtype=vae_dtype)
        enable_vae_consumer_opts(vae)
        print(f"[vae] mode=small-decoder  dtype={vae_dtype}")
        return vae, f"small-decoder:{repo}"

    if source == "pretrained":
        # CORRECT path used by official docs — never from_single_file
        try:
            vae = load_vae_flux2_from_pretrained(base, subfolder=sub, torch_dtype=vae_dtype)
        except Exception as e:
            print(f"[vae] pretrained {base}/{sub} failed ({e}); try small-decoder")
            repo = info.get("small_vae_repo") or "black-forest-labs/FLUX.2-small-decoder"
            vae = load_vae_flux2_from_pretrained(repo, subfolder=None, torch_dtype=vae_dtype)
            enable_vae_consumer_opts(vae)
            return vae, f"small-decoder-fallback:{repo}"
        enable_vae_consumer_opts(vae)
        print(f"[vae] mode=pretrained  {base}/{sub}  dtype={vae_dtype}")
        return vae, f"pretrained:{base}/{sub}"

    if source == "local":
        path = vae_path
        if not file_ok(path) and download_missing:
            # download Comfy flux2-vae.safetensors (weights only)
            last = None
            for repo, fname in info.get("comfy_vae_repos", []):
                try:
                    path = hf_download(repo, fname, models_dir)
                    break
                except Exception as e:
                    last = e
                    path = None
            if not file_ok(path):
                raise FileNotFoundError(f"Could not fetch Comfy VAE: {last}")
        if not file_ok(path):
            raise FileNotFoundError(
                "vae-source=local requires --vae path or --download-missing"
            )

        # Prefer config from official base; weights from local Comfy/BFL file
        try:
            vae = load_vae_flux2_from_safetensors(
                weights_path=path,
                config_source=base,
                config_subfolder=sub,
                torch_dtype=vae_dtype,
                expected_latent_channels=z_ch,
            )
        except Exception as e:
            print(f"[vae] local+config failed ({e}); fall back to pure pretrained")
            vae = load_vae_flux2_from_pretrained(base, subfolder=sub, torch_dtype=vae_dtype)
            enable_vae_consumer_opts(vae)
            return vae, f"pretrained-fallback:{base}/{sub}"

        enable_vae_consumer_opts(vae)
        print(f"[vae] mode=local  weights={path}  config={base}/{sub}")
        return vae, f"local:{path}"

    raise ValueError(f"Unknown --vae-source {vae_source}")


# =============================================================================
# DiT GGUF
# =============================================================================

def load_dit_gguf(gguf_path: str, model_type: str, torch_dtype: torch.dtype):
    try:
        from diffusers import Flux2Transformer2DModel, GGUFQuantizationConfig
    except Exception as e:
        require("'diffusers>=0.36'", e)

    info = MODEL_REGISTRY[model_type]
    base = info["base_config"]
    sub = info["dit_config_subfolder"]
    print(f"[dit] GGUF {gguf_path}")
    print(f"[dit] config {base}/{sub}")
    qcfg = GGUFQuantizationConfig(compute_dtype=torch_dtype)
    transformer = Flux2Transformer2DModel.from_single_file(
        gguf_path,
        quantization_config=qcfg,
        torch_dtype=torch_dtype,
        config=base,
        subfolder=sub,
    )
    return transformer


# =============================================================================
# Text encoder (quantized)
# =============================================================================

def load_tokenizer(model_type: str, te_gguf: Optional[str], te_bnb: str):
    try:
        from transformers import AutoTokenizer
    except Exception as e:
        require("transformers", e)

    base = MODEL_REGISTRY[model_type]["base_config"]
    try:
        tok = AutoTokenizer.from_pretrained(base, subfolder="tokenizer")
        print(f"[tok] {base}/tokenizer")
        return tok
    except Exception:
        pass

    if file_ok(te_gguf):
        d = os.path.dirname(os.path.abspath(te_gguf)) or "."
        f = os.path.basename(te_gguf)
        try:
            tok = AutoTokenizer.from_pretrained(d, gguf_file=f)
            print(f"[tok] from GGUF {f}")
            return tok
        except Exception as e:
            print(f"[tok] GGUF meta failed: {e}")

    tok = AutoTokenizer.from_pretrained(te_bnb)
    print(f"[tok] {te_bnb}")
    return tok


def load_text_encoder(
    model_type: str,
    te_mode: str,
    te_gguf: Optional[str],
    te_bnb_id: str,
    torch_dtype: torch.dtype,
    device: str,
    download_missing: bool,
    models_dir: str,
):
    try:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    except Exception as e:
        require("transformers bitsandbytes", e)

    info = MODEL_REGISTRY[model_type]
    mode = te_mode.lower()
    if mode == "auto":
        mode = "gguf" if file_ok(te_gguf) else "bnb4"

    if mode == "gguf":
        if not file_ok(te_gguf) and download_missing and info.get("te_gguf_repo"):
            te_gguf = hf_download_first(
                info["te_gguf_repo"], info["te_gguf_candidates"], models_dir
            )
        if not file_ok(te_gguf):
            print("[te] no GGUF → fall back to bnb4")
            mode = "bnb4"
        else:
            d = os.path.dirname(os.path.abspath(te_gguf)) or "."
            f = os.path.basename(te_gguf)
            print(f"[te] Qwen3 GGUF→dense (CPU, offloaded): {te_gguf}")
            model = AutoModelForCausalLM.from_pretrained(
                d,
                gguf_file=f,
                torch_dtype=torch_dtype,
                device_map="cpu",
                low_cpu_mem_usage=True,
            )
            model.eval()
            return model, f"gguf:{f}"

    bits = 4 if mode == "bnb4" else 8
    print(f"[te] bitsandbytes {bits}-bit  {te_bnb_id}")
    if bits == 4:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        bnb = BitsAndBytesConfig(load_in_8bit=True)

    device_map = "auto" if str(device).startswith("cuda") else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        te_bnb_id,
        quantization_config=bnb,
        device_map=device_map,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, f"bnb{bits}:{te_bnb_id}"


# =============================================================================
# Scheduler + pipeline
# =============================================================================

def load_scheduler(base_config: str):
    try:
        from diffusers import FlowMatchEulerDiscreteScheduler
    except Exception as e:
        require("diffusers", e)
    return FlowMatchEulerDiscreteScheduler.from_pretrained(
        base_config, subfolder="scheduler"
    )


def build_pipeline(
    model_type: str,
    dit_gguf: str,
    vae,
    text_encoder,
    tokenizer,
    torch_dtype: torch.dtype,
    device: str,
    cpu_offload: bool,
):
    info = MODEL_REGISTRY[model_type]
    base = info["base_config"]

    try:
        from diffusers import Flux2KleinPipeline, Flux2Pipeline
    except Exception as e:
        require("'diffusers>=0.36'", e)

    transformer = load_dit_gguf(dit_gguf, model_type, torch_dtype)
    free_mem()
    scheduler = load_scheduler(base)

    pipe_name = info["pipeline"]
    Pipe = Flux2KleinPipeline if pipe_name == "Flux2KleinPipeline" else Flux2Pipeline

    print(f"[pipe] {pipe_name}  distilled={info['is_distilled']}")
    kwargs = dict(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
    )
    # is_distilled only on Klein pipeline constructor in recent diffusers
    try:
        if pipe_name == "Flux2KleinPipeline":
            pipe = Pipe(**kwargs, is_distilled=info["is_distilled"])
        else:
            pipe = Pipe(**kwargs)
    except TypeError:
        pipe = Pipe(**kwargs)

    if str(device).startswith("cuda") and torch.cuda.is_available():
        if cpu_offload:
            print("[pipe] enable_model_cpu_offload")
            try:
                pipe.enable_model_cpu_offload()
            except Exception as e:
                print(f"[pipe] cpu_offload failed ({e}); sequential")
                try:
                    pipe.enable_sequential_cpu_offload()
                except Exception as e2:
                    print(f"[pipe] sequential failed ({e2}); .to({device})")
                    pipe.to(device)
        else:
            pipe.to(device)
    else:
        pipe.to("cpu")
    return pipe


# =============================================================================
# Inference
# =============================================================================

def run_inference(
    dit_gguf: str,
    prompt: str,
    model_type: str = "auto",
    negative_prompt: str = "",
    cfg_scale: Optional[float] = None,
    num_inference_steps: Optional[int] = None,
    seed: int = 42,
    height: int = 512,
    width: int = 512,
    dtype: str = "bfloat16",
    device: str = "cuda",
    output_png: str = "fox.png",
    text_encoder_gguf: Optional[str] = None,
    te_mode: str = "auto",
    te_bnb: Optional[str] = None,
    vae_path: Optional[str] = None,
    vae_source: str = "auto",
    download_missing: bool = False,
    models_dir: str = "models",
    cpu_offload: bool = True,
    max_sequence_length: int = 512,
    base_config_override: Optional[str] = None,
) -> Dict[str, Any]:
    if model_type == "auto":
        model_type = guess_model_type_from_path(dit_gguf)
    else:
        model_type = resolve_model_type(model_type)

    info = MODEL_REGISTRY[model_type]
    if base_config_override:
        info = dict(info)
        info["base_config"] = base_config_override
        MODEL_REGISTRY[model_type] = info  # local to this process

    torch_dtype = resolve_dtype(dtype)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        print("[warn] no CUDA → cpu")
        device = "cpu"
        cpu_offload = False

    if cfg_scale is None:
        cfg_scale = float(info["default_cfg"])
    if num_inference_steps is None:
        num_inference_steps = int(info["default_steps"])

    te_bnb_id = te_bnb or info["te_bnb"]

    if not os.path.isfile(dit_gguf):
        raise FileNotFoundError(dit_gguf)

    meta = parse_gguf_meta(dit_gguf)
    print_banner(
        [
            "Flux.2 consumer-GPU inference (model-type–aware VAE)",
            f"  model_type   : {model_type}  family={info['family']}",
            f"  DiT GGUF     : {dit_gguf}",
            f"  base_config  : {info['base_config']}",
            f"  vae_source   : {vae_source}  path={vae_path}",
            f"  vae_class    : {info['vae_class']}  z_ch={info['vae_latent_channels']}",
            f"  te_mode      : {te_mode}  bnb={te_bnb_id}",
            f"  prompt       : {prompt!r}",
            f"  cfg / steps  : {cfg_scale} / {num_inference_steps}",
            f"  size         : {height}x{width}",
            f"  dtype/device : {torch_dtype} @ {device}  offload={cpu_offload}",
            f"  output       : {output_png}",
        ]
    )
    print_gguf_meta(meta)

    # --- VAE first (config-heavy, small) ---
    vae, vae_label = load_vae_for_model_type(
        model_type=model_type,
        vae_source=vae_source,
        vae_path=vae_path,
        torch_dtype=torch_dtype,
        download_missing=download_missing,
        models_dir=models_dir,
    )
    free_mem()

    # --- Text encoder quantized ---
    text_encoder, te_label = load_text_encoder(
        model_type=model_type,
        te_mode=te_mode,
        te_gguf=text_encoder_gguf,
        te_bnb_id=te_bnb_id,
        torch_dtype=torch_dtype,
        device=device,
        download_missing=download_missing,
        models_dir=models_dir,
    )
    free_mem()

    tokenizer = load_tokenizer(model_type, text_encoder_gguf, te_bnb_id)
    free_mem()

    pipe = build_pipeline(
        model_type=model_type,
        dit_gguf=dit_gguf,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        torch_dtype=torch_dtype,
        device=device,
        cpu_offload=cpu_offload,
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
    try:
        params = inspect.signature(pipe.__call__).parameters
        if "max_sequence_length" in params:
            call_kwargs["max_sequence_length"] = max_sequence_length
        if "negative_prompt" in params and negative_prompt:
            call_kwargs["negative_prompt"] = negative_prompt
        elif negative_prompt and abs(cfg_scale - 1.0) > 1e-6:
            print("[info] negative_prompt ignored by this pipeline")
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
        "mode": "consumer_gguf_modeltype_vae",
        "model_type": model_type,
        "prompt": prompt,
        "cfg_scale": cfg_scale,
        "num_inference_steps": num_inference_steps,
        "seed": seed,
        "dtype": str(torch_dtype),
        "device": device,
        "dit_gguf": dit_gguf,
        "vae": vae_label,
        "text_encoder": te_label,
        "output_png": output_png,
        "dit_meta": meta,
    }


# =============================================================================
# CLI
# =============================================================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Flux.2 Klein GGUF consumer runner — model-type–aware VAE"
    )
    p.add_argument("dit_gguf", type=str, help="Path to DiT GGUF (Unsloth/city96)")
    p.add_argument(
        "--model-type",
        type=str,
        default="auto",
        help="flux2_klein_4b | flux2_klein_9b | flux2_klein_base_4b | flux2_dev | auto",
    )
    p.add_argument(
        "--vae-source",
        type=str,
        default="pretrained",
        choices=["auto", "pretrained", "local", "small-decoder"],
        help=(
            "How to load VAE (default pretrained). "
            "pretrained = AutoencoderKLFlux2.from_pretrained(base/vae) — ALWAYS works. "
            "local = your flux2-vae.safetensors + Flux2 config (no from_single_file). "
            "small-decoder = FLUX.2-small-decoder. "
            "auto = local if --vae else pretrained."
        ),
    )
    p.add_argument(
        "--vae",
        type=str,
        default="",
        help="Optional local flux2-vae.safetensors (for --vae-source local|auto)",
    )
    p.add_argument("--text-encoder-gguf", type=str, default="")
    p.add_argument(
        "--te-mode",
        type=str,
        default="bnb4",
        choices=["auto", "gguf", "bnb4", "bnb8"],
        help="Default bnb4 for lowest VRAM / no TE GGUF dequant spike",
    )
    p.add_argument("--te-bnb", type=str, default="", help="Override TE id for bnb")
    p.add_argument(
        "--base-config",
        type=str,
        default="",
        help="Override HF id used for configs (DiT/VAE/tokenizer/scheduler)",
    )
    p.add_argument("--download-missing", action="store_true")
    p.add_argument("--models-dir", type=str, default="models")
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
    p.add_argument(
        "--cfg-scale",
        type=float,
        default=None,
        help="Default from model-type (1.0 distilled Klein)",
    )
    p.add_argument("--steps", type=int, default=None, dest="num_inference_steps")
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
    p.add_argument("--no-cpu-offload", action="store_true")
    p.add_argument("--max-sequence-length", type=int, default=512)
    p.add_argument("--meta-only", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_argparser().parse_args(argv)

    if args.meta_only:
        print_gguf_meta(parse_gguf_meta(args.dit_gguf))
        return

    result = run_inference(
        dit_gguf=args.dit_gguf,
        prompt=args.prompt,
        model_type=args.model_type,
        negative_prompt=args.negative_prompt,
        cfg_scale=args.cfg_scale,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        height=args.height,
        width=args.width,
        dtype=args.dtype,
        device=args.device,
        output_png=args.output_png,
        text_encoder_gguf=args.text_encoder_gguf or None,
        te_mode=args.te_mode,
        te_bnb=args.te_bnb or None,
        vae_path=args.vae or None,
        vae_source=args.vae_source,
        download_missing=args.download_missing,
        models_dir=args.models_dir,
        cpu_offload=not args.no_cpu_offload,
        max_sequence_length=args.max_sequence_length,
        base_config_override=args.base_config or None,
    )

    print("\nSummary")
    for k in (
        "mode",
        "model_type",
        "output_png",
        "dit_gguf",
        "vae",
        "text_encoder",
        "cfg_scale",
        "num_inference_steps",
        "dtype",
        "device",
    ):
        print(f"  {k:20s}: {result.get(k)}")
    print(f"  dit quants          : {result['dit_meta'].get('qtype_counts')}")


if __name__ == "__main__":
    main()