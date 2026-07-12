# Mann-E GGUF Inference Engine

<p align="center">
    <img src="mosque.png" />
</p>

## What is this project? 

This project is a _standalone_ tool for running image generation models which are quantized in GGUF format, bease on [Comfy UI GGUF Node](https://github.com/city96/ComfyUI-GGUF). This repository will help you get your hands on image generation through a terminal or remote server and is a good replacement for AUTO1111 or ComfyUI if you want to __only__ use the terminal window or a JupyterLab or Notebook environment. 

We also developed it as a part of _Mann-E's Inference Engine_ to keep our resources free and use them optimally. Currently we're working on getting all types of model to work using this engine. 

## AI Use in The Project 

The project has been understood (from Comfy's GGUF node) and planned by _Calude Fable 5_ and coded by _Grok 4.5_. Since Mann-E is an AI company, it was part of our plans to code most of our projects and products using reliable and good existing models. However both Claude Fable and Grok were very good in their tasks and they're currently part of our production flow. No agentic AI used by the way, it was just asking the chat interface to do the thing. 

## How to use?

The fastest and easiest way is to use this [Jupyter Notebook](GGUF_Inference_for_FLUX_Stable_Diffusion.ipynb) to understand how it works, however if you want to run it locally or outside of Jupyter, this is the way. 

### Installing dependencies

In this part, we consider you have created a new environment or use RunPod or similar services.

1. Install `pytorch`. If you use Google Colab or RunPod, you usually have a profile which has pytorch installed. 
2. Install other libraries: 
    ```
    pip install gguf transformers accelerate safetensors sentencepiece protobuf pillow huggingface_hub bitsandbytes -q
    ```
3. Clone and navigate to the repository's directory. 
    ```
    git clone https://github.com/Mann-E/gguf_inference
    cd gguf_inference
    ```
4. Download your desired model. For example, we've done tests on Klein 4B (Q4_K_M/Q4_0 quantizations)
    ```
    wget -c https://huggingface.co/unsloth/FLUX.2-klein-4B-GGUF/resolve/main/flux-2-klein-4b-Q4_0.gguf?download=true -O "flux-2-klein-4b-Q4_0.gguf"
    ``` 
5. You can use `--meta-only` flag to get the quantization information. 
    ```
    python gguf_inference.py flux-2-klein-4b-Q4_0.gguf --meta-only
    ``` 
6. You can run the project and get your desired image outputs:
    ```
    python gguf_inference.py flux-2-klein-4b-Q4_0.gguf \
    --model-type flux2_klein_4b \
    --vae-source pretrained \
    --te-mode bnb4 \
    --prompt "arabesque paper marbling painting of a mosque, architectural picture" \
    --cfg-scale 1.0 --steps 12 \
    --height 576 --width 1024 \
    --dtype bfloat16 --device cuda \
    --output mosque.png
    ``` 
## Supported Models

Currently the tool supports:

- Flux 2 Klein 4B (and fine tunes)
- Flux 2 Klein 9B (and fine tunes)
- Flux 2 Dev (and fine tunes)

## Known Issues 

Currently there is no known bug, except that for smallest model on the list above you need at least 8GB of VRAM. 

## TODO List

_TBD_