#!/usr/bin/env python
# Simple Wan2.1 T2V script - modified to use quantized models
import os
import sys
import time
import argparse
import torch
import gc
from pathlib import Path
from datetime import datetime
import json
import wan
from wan.configs import WAN_CONFIGS
from wan.utils.utils import cache_video
from wan.modules.attention import get_attention_modes, get_supported_attention_modes
from mmgp import offload, safetensors2, profile_type
from huggingface_hub import hf_hub_download, snapshot_download
from wan.utils import prompt_parser
import numpy as np
import traceback
import random
from tqdm import tqdm

# Helper Functions
def download_models(transformer_filename, text_encoder_filename):
    """Check for required model files and warn if missing."""
    # Modified to check local files instead of downloading
    missing_files = []
    
    # Check if files exist
    if not os.path.isfile(transformer_filename):
        missing_files.append(transformer_filename)
    
    if not os.path.isfile(text_encoder_filename):
        missing_files.append(text_encoder_filename)
    
    if not os.path.isfile("ckpts/Wan2.1_VAE.safetensors"):
        missing_files.append("ckpts/Wan2.1_VAE.safetensors")
    
    if missing_files:
        print("ERROR: The following required files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        print("Please make sure these files are available before running the script.")
        sys.exit(1)
    else:
        print("All required model files found.")

def get_auto_attention():
    """Get the best available attention mechanism."""
    attention_modes_supported = get_supported_attention_modes()
    for attn in ["sage2", "sage", "sdpa"]:
        if attn in attention_modes_supported:
            return attn
    return "sdpa"

def sanitize_file_name(file_name, rep=""):
    """Clean filename of invalid characters."""
    return file_name.replace("/", rep).replace("\\", rep).replace(":", rep).replace("|", rep).replace("?", rep).replace("<", rep).replace(">", rep).replace("\"", rep)

def setup_teacache(model, enabled, multiplier, start_step_percent, steps, model_type):
    """Configure TeaCache settings for model acceleration."""
    model.enable_teacache = enabled
    
    if not enabled:
        return
    
    model.teacache_multiplier = multiplier
    model.rel_l1_thresh = 0
    model.teacache_start_step = int(start_step_percent * steps / 100)
    model.num_steps = steps
    model.teacache_skipped_steps = 0
    model.previous_residual_uncond = None
    model.previous_residual_cond = None
    
    # Set appropriate coefficients based on model type
    if model_type == '14B':
        model.coefficients = [-5784.54975374, 5449.50911966, -1811.16591783, 256.27178429, -13.02252404]
    else:  # 1.3B
        model.coefficients = [2.39676752e+03, -1.31110545e+03, 2.01331979e+02, -8.29855975e+00, 1.37887774e-01]
    
    print(f"TeaCache enabled with multiplier {multiplier}, starting at step {model.teacache_start_step}")

def parse_args():
    parser = argparse.ArgumentParser(description="Simple Wan2.1 text-to-video generator (quantized models)")
    
    # Basic parameters
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for video generation")
    parser.add_argument("--negative-prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save videos")
    parser.add_argument("--resolution", type=str, default="832x480", 
                        choices=["832x480", "480x832", "1280x720", "720x1280", "1024x1024"], 
                        help="Video resolution")
    parser.add_argument("--frames", type=int, default=81, help="Number of frames to generate")
    parser.add_argument("--steps", type=int, default=30, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed (-1 for random)")
    parser.add_argument("--guidance-scale", type=float, default=5.0, 
                        help="Guidance scale (higher = more prompt adherence)")
    parser.add_argument("--flow-shift", type=float, default=5.0, help="Flow shift scale")
    
    # Model parameters - modified to default to quantized models
    parser.add_argument("--model", type=str, default="14B", choices=["14B", "1.3B"], 
                        help="Model size to use")
    # Removed quantize flags as we're now always using quantized models
    
    # Hardware/performance parameters
    parser.add_argument("--gpu", type=str, default="cuda", help="GPU device to use")
    parser.add_argument("--attention", type=str, default="auto", 
                        choices=["auto", "sdpa", "flash", "sage", "sage2"], 
                        help="Attention mechanism")
    parser.add_argument("--profile", type=int, default=4, choices=[1, 2, 3, 4, 5],
                        help="Memory profile (1=HighRAM_HighVRAM, 4=LowRAM_LowVRAM)")
    parser.add_argument("--preload", type=int, default=4096, 
                        help="Megabytes to preload into VRAM")
    parser.add_argument("--compile", action="store_true", 
                        help="Enable PyTorch compilation")
    parser.add_argument("--vae-tile-size", type=int, default=0, 
                        choices=[0, 128, 256, 512], 
                        help="VAE tiling size (0=auto)")
    
    # Advanced features
    parser.add_argument("--riflex", action="store_true", 
                        help="Enable RIFLEX for long videos")
    parser.add_argument("--tea-cache", type=float, default=0.0, 
                        help="TeaCache acceleration (0=disabled, 1.5-2.5=speedup factor)")
    parser.add_argument("--tea-cache-start", type=int, default=0, 
                        help="TeaCache starting point (percent of generation)")
    parser.add_argument("--verbose", type=int, default=1, choices=[0, 1, 2], 
                        help="Verbosity level")
    
    return parser.parse_args()

def load_model(args):
    """Load the Wan2.1 T2V model with quantized settings."""
    # Always use quantized models based on what's available
    model_size = args.model
    
    # Set appropriate files based on model size
    if model_size == "14B":
        # Use the quantized int8 model (which is already downloaded)
        transformer_filename = "ckpts/wan2.1_text2video_14B_quanto_int8.safetensors"
        cfg = WAN_CONFIGS['t2v-14B']
    else:  # 1.3B
        # For 1.3B, we'll use the non-quantized model since that's what's available
        transformer_filename = "ckpts/wan2.1_text2video_1.3B_bf16.safetensors"
        cfg = WAN_CONFIGS['t2v-1.3B']
    
    # Use the quantized text encoder that's already downloaded
    text_encoder_filename = "ckpts/models_t5_umt5-xxl-enc-quanto_int8.safetensors"
    
    # Check if files exist (replaces downloading)
    download_models(transformer_filename, text_encoder_filename)
    
    # Set device
    if len(args.gpu) > 0:
        torch.set_default_device(args.gpu)
    
    # Determine precision based on GPU capabilities
    major, minor = torch.cuda.get_device_capability(None)
    if major < 8:
        print("GPU architecture doesn't support bf16, using fp16")
        dtype = torch.float16
    else:
        dtype = torch.bfloat16
    
    # Load model
    print(f"Loading model: {transformer_filename}")
    wan_model = wan.WanT2V(
        config=cfg,
        checkpoint_dir="ckpts",
        model_filename=transformer_filename,
        text_encoder_filename=text_encoder_filename,
        quantizeTransformer=True,  # Always use quantized transformer
        dtype=dtype,
        VAE_dtype=torch.float32,
        mixed_precision_transformer=False
    )
    
    # FIX: Add the _interrupt attribute
    wan_model._interrupt = False
    
    # Setup memory management
    pipe = {
        "transformer": wan_model.model,
        "text_encoder": wan_model.text_encoder.model,
        "vae": wan_model.vae.model
    }
    
    # Setup offloading profile
    kwargs = {"extraModelsToQuantize": None}
    if args.profile == 2 or args.profile == 4:
        kwargs["budgets"] = {
            "transformer": args.preload if args.preload > 0 else 100,
            "text_encoder": 100,
            "*": 1000
        }
    elif args.profile == 3:
        kwargs["budgets"] = {"*": "70%"}
    
    # Configure attention mechanism
    effective_attention = args.attention
    if effective_attention == "auto":
        effective_attention = get_auto_attention()
    offload.shared_state["_attention"] = effective_attention
    print(f"Using attention mode: {effective_attention}")
    
    # Apply offloading profile
    offloadobj = offload.profile(
        pipe,
        profile_no=args.profile,
        compile="transformer" if args.compile else "",
        quantizeTransformer=True,  # Always use quantized transformer
        loras="transformer",
        coTenantsMap={},
        perc_reserved_mem_max=0,
        convertWeightsFloatTo=dtype,
        **kwargs
    )
    
    print("Model loaded successfully")
    return wan_model, offloadobj, transformer_filename

def generate_video(wan_model, args, model_filename):
    """Generate video from text prompt with the loaded model."""
    print(f"Generating video for prompt: {args.prompt}")
    
    # Process prompt template if any
    processed_prompt, errors = prompt_parser.process_template(args.prompt)
    if errors:
        raise ValueError(f"Prompt template error: {errors}")
    
    # Parse resolution
    width, height = map(int, args.resolution.split("x"))
    
    # Set random seed
    seed = args.seed
    if seed == -1:
        seed = random.randint(0, 999999999)
        print(f"Using random seed: {seed}")
    
    # Configure VAE tiling based on available VRAM
    vae_tile_size = args.vae_tile_size
    if vae_tile_size == 0:
        device_mem_capacity = torch.cuda.get_device_properties(None).total_memory / 1048576
        if device_mem_capacity >= 24000:
            vae_tile_size = 0
        elif device_mem_capacity >= 8000:
            vae_tile_size = 256
        else:
            vae_tile_size = 128
        print(f"Auto VAE tiling: {vae_tile_size or 'Disabled'} (VRAM: {device_mem_capacity:.0f}MB)")
    
    # Enable RIFLEX for longer videos by default if needed
    enable_RIFLEx = args.riflex or args.frames > (6 * 16)
    if enable_RIFLEx:
        print("RIFLEX enabled for long video generation")
    
    # Configure TeaCache
    model_type = "14B" if "14B" in model_filename else "1.3B"
    setup_teacache(
        wan_model.model,
        enabled=args.tea_cache > 0,
        multiplier=args.tea_cache,
        start_step_percent=args.tea_cache_start,
        steps=args.steps,
        model_type=model_type
    )
    
    # Adjust frames to be 4n+1
    frames = (args.frames // 4) * 4 + 1
    if frames != args.frames:
        print(f"Adjusted frames from {args.frames} to {frames} to be 4n+1")
    
    start_time = time.time()
    
    # Generate the video
    print(f"Starting generation with {frames} frames, {args.steps} steps...")
    
    # Try to add safety filter to prevent bad content
    try:
        # Add negative prompt for safety if not already specified
        if not args.negative_prompt:
            safety_negatives = "poor quality, low quality, bad anatomy, wrong anatomy, extra limbs, missing limbs, floating limbs, disconnected limbs, mutation, mutated, disfigured, deformed, poorly drawn hands, too many fingers, missing fingers, extra fingers, fused fingers, disproportionate body, bad proportions, malformed limbs, extra body parts, missing body parts, floating body parts, disfigured face, deformed face, ugly face, bad face anatomy, weird face, double face, two faces, multiple faces, displaced facial features, disproportionate face, unrealistic eyes, crossed eyes, wandering eyes, misaligned eyes, poorly drawn face, face artifacts, unnatural skin, plastic skin, shiny skin, weird expression, unnatural expression, back view, rear view, side view, obscured front, blocked view, cropped body, off-center subject, unbalanced composition, cluttered background, unwanted objects, overlapping figures, overexposed, underexposed, harsh shadows, blurry, grainy, pixelated, out of focus, dark image, washed-out colors, unnatural lighting, flat lighting, static pose, awkward pose, unnatural stance, stiff movement, jerky motion, lack of fluidity, exaggerated motion, unintended angles, uneven body contours, twisted limbs, unnatural joint bends, warped torso"
            # safety_negatives = "poor quality"
            args.negative_prompt = safety_negatives
            print(f"Added safety negative prompt: {safety_negatives}")
    except Exception as e:
        print(f"Warning: Could not add safety filter: {e}")
    
    samples = wan_model.generate(
        processed_prompt,
        frame_num=frames,
        size=(width, height),
        shift=args.flow_shift,
        sampling_steps=args.steps,
        guide_scale=args.guidance_scale,
        n_prompt=args.negative_prompt,
        seed=seed,
        enable_RIFLEx=enable_RIFLEx,
        VAE_tile_size=vae_tile_size,
        joint_pass=True  # This provides a speed boost
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save the video
    time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%Hh%Mm%Ss")
    safe_prompt = sanitize_file_name(args.prompt[:50]).strip()
    file_name = f"{time_flag}_seed{seed}_{safe_prompt}.mp4"
    video_path = os.path.join(args.output_dir, file_name)
    
    # Cache video to file
    samples = samples.to("cpu")
    cache_video(
        tensor=samples[None],
        save_file=video_path,
        fps=16,
        nrow=1,
        normalize=True,
        value_range=(-1, 1)
    )
    
    # Save metadata
    metadata = {
        "prompt": processed_prompt,
        "negative_prompt": args.negative_prompt,
        "seed": seed,
        "steps": args.steps,
        "frames": frames,
        "guidance_scale": args.guidance_scale,
        "flow_shift": args.flow_shift,
        "resolution": args.resolution,
        "model": model_type,
        "quantized": True,  # Always true now since we're using quantized models
        "tea_cache": args.tea_cache > 0,
        "riflex": enable_RIFLEx,
        "attention": offload.shared_state["_attention"],
        "generation_time": time.time() - start_time
    }
    
    metadata_path = video_path.replace(".mp4", ".json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    
    end_time = time.time()
    print(f"Video generated successfully in {end_time - start_time:.2f} seconds")
    print(f"Saved to: {video_path}")
    return video_path

def download_specific_models():
    """Download only the specified Wan2.1 model files if they don't exist locally."""
    repo_id = "DeepBeepMeep/Wan2.1"
    target_dir = "ckpts"
    os.makedirs(target_dir, exist_ok=True)
    files_to_download = [
        ("wan2.1_text2video_14B_quanto_int8.safetensors", ""),
        ("models_t5_umt5-xxl-enc-quanto_int8.safetensors", ""),
        ("Wan2.1_VAE.safetensors", "")
    ]
    print("Starting download of Wan2.1 model files...")
    for filename, subfolder in tqdm(files_to_download, desc="Downloading model files"):
        local_file_path = os.path.join(target_dir, filename)
        if os.path.isfile(local_file_path):
            print(f"\u2713 {filename} already exists, skipping download")
            continue
        try:
            print(f"Downloading {filename}...")
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=target_dir,
                subfolder=subfolder
            )
            print(f"\u2713 Successfully downloaded {filename}")
        except Exception as e:
            print(f"\u2717 Error downloading {filename}: {str(e)}")
            return False
    all_downloaded = True
    for filename, _ in files_to_download:
        local_file_path = os.path.join(target_dir, filename)
        if not os.path.isfile(local_file_path):
            print(f"\u2717 Failed to download {filename}")
            all_downloaded = False
    if all_downloaded:
        print("\nAll model files downloaded successfully!")
        return True
    else:
        print("\nSome files could not be downloaded. Please check the errors above.")
        return False

def main():
    # Download/check model files before anything else
    success = download_specific_models()
    if not success:
        print("Model files missing or failed to download. Exiting.")
        sys.exit(1)
    # Parse command line arguments
    args = parse_args()
    
    # Configure MMGP verbosity
    offload.default_verboseLevel = args.verbose
    
    try:
        # Load model
        wan_model, offloadobj, model_filename = load_model(args)
        
        # Generate video
        video_path = generate_video(wan_model, args, model_filename)
        
        print(f"\nGeneration complete! Video saved to: {video_path}")
        
    except Exception as e:
        print(f"Error during generation: {e}")
        traceback.print_exc()
        return 1
    finally:
        # Clean up resources
        if 'offloadobj' in locals() and offloadobj is not None:
            print("Releasing MMGP profile...")
            offloadobj.release()
        
        # Force garbage collection and clear CUDA cache
        gc.collect()
        torch.cuda.empty_cache()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())