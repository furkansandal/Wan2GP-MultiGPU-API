#!/usr/bin/env python3
"""
WGP API - LTX Video 0.9.7 Distilled 13B Inference API
Supports image-to-video generation with multiple aspect ratios
"""

import os
import sys
import asyncio
import logging
import torch
import gc
import time
import uuid
import tempfile
import shutil
from typing import Optional, Union
from datetime import datetime
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import argparse
import random
import cv2
import json

# FastAPI imports
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from contextlib import asynccontextmanager

# Import project modules
from mmgp import offload
from ltx_video.ltxv import LTXV as BaseLTXV
from wan.modules.attention import get_supported_attention_modes
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from ltx_video.pipelines import crf_compressor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('wgp_api.log')
    ]
)
logger = logging.getLogger(__name__)


class LTXV(BaseLTXV):
    """Wrapper class for LTXV that ensures compatibility"""
    
    def __init__(self, model_filepath, text_encoder_filepath, model_def, *args, **kwargs):
        # Call parent init with all required parameters
        super().__init__(model_filepath, text_encoder_filepath, model_def, *args, **kwargs)
        
        # Add interrupt flag for pipeline compatibility
        self._interrupt = False
    
    def generate(self, *args, **kwargs):
        # Pass all parameters directly to the base class
        return super().generate(*args, **kwargs)


# Model definitions - matching wgp.py structure
def get_model_def(model_type):
    """Get model definition from JSON files"""
    model_def_path = f"defaults/{model_type}.json"
    
    if not os.path.exists(model_def_path):
        # Try finetunes folder
        model_def_path = f"finetunes/{model_type}.json"
    
    if not os.path.exists(model_def_path):
        raise ValueError(f"Model definition not found for {model_type}")
    
    with open(model_def_path, "r", encoding="utf-8") as f:
        json_def = json.load(f)
    
    model_def = json_def["model"]
    model_def["path"] = model_def_path
    
    # Handle settings
    del json_def["model"]
    settings = json_def
    model_def["settings"] = settings
    
    return model_def


# Constants
MAX_FRAMES = 129  # User requested 129 frames
# LTXV recommended resolutions (width x height)
SUPPORTED_RESOLUTIONS = {
    "9:16": (704, 1216),   # LTXV portrait
    "16:9": (1216, 704),   # LTXV landscape (default)
    "1:1": (960, 960),     # Square
    "3:4": (912, 1216),    # Portrait
    "4:3": (1216, 912)     # Traditional
}

DEFAULT_NEGATIVE_PROMPT = "low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly"

# Global variables for model
model = None
device = None
model_lock = asyncio.Lock()
offloadobj = None
prompt_enhancer_image_caption_model = None
prompt_enhancer_image_caption_processor = None
prompt_enhancer_llm_model = None
prompt_enhancer_llm_tokenizer = None

# Task queue for video generation
generation_tasks = {}
task_queue = asyncio.Queue()
worker_task = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    logger.info("Starting WGP API server...")
    
    # Start model loading
    logger.info("Starting model loading...")
    load_model()
    
    # Start video generation worker
    global worker_task
    worker_task = asyncio.create_task(video_generation_worker())
    logger.info("Video generation worker started")
    
    # Start cleanup task
    cleanup_task = asyncio.create_task(cleanup_old_tasks())
    logger.info("Cleanup task started")
    
    logger.info("API server ready!")
    yield
    
    # Cleanup
    logger.info("Shutting down...")
    if worker_task:
        worker_task.cancel()


# Create FastAPI app
app = FastAPI(
    title="WGP API - LTX Video Generation",
    description="API for LTX Video 0.9.7 Distilled 13B model",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class VideoGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt for video generation")
    negative_prompt: str = Field(DEFAULT_NEGATIVE_PROMPT, description="Negative prompt")
    image_url: str = Field(..., description="URL of the image to animate")
    aspect_ratio: str = Field("16:9", description="Aspect ratio (9:16, 16:9, 1:1, 3:4, 4:3)")
    seed: int = Field(-1, description="Random seed (-1 for random)")
    enhance_prompt_with_llm: bool = Field(True, description="Whether to enhance prompt with LLM")


class VideoGenerationResponse(BaseModel):
    task_id: str
    status: str = "pending"
    message: str = "Task queued for processing"


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    video_url: Optional[str] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None
    queue_position: Optional[int] = None
    created_at: str
    updated_at: str


class TaskInfo:
    def __init__(self, task_id: str, request: VideoGenerationRequest):
        self.task_id = task_id
        self.request = request
        self.status = "pending"
        self.video_url = None
        self.error = None
        self.processing_time = None
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.temp_dir = None


# Argument parser
parser = argparse.ArgumentParser(description='WGP API Server')
parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
parser.add_argument('--gpu', type=str, default='cuda:0', help='GPU device to use')
parser.add_argument('--profile', type=int, default=1, help='Offload profile (0-5)')
parser.add_argument('--quantization', type=str, default='int8', help='Model quantization')
parser.add_argument('--fp16', action='store_true', help='Use fp16 precision')
parser.add_argument('--bf16', action='store_true', help='Use bf16 precision')
parser.add_argument('--vae-config', type=str, default='0', help='VAE configuration')
parser.add_argument('--attention', type=str, default='xformers', help='Attention mode')
parser.add_argument('--model', type=str, default='ltxv_13B', help='Model type to use')
parser.add_argument('--distilled', action='store_true', help='Use distilled model')

args = parser.parse_args()

# Update model type based on distilled flag
if args.distilled:
    args.model = 'ltxv_distilled'

# Global configuration
device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
profile = args.profile
transformer_quantization = args.quantization
text_encoder_quantization = args.quantization
attention_mode = args.attention


def get_transformer_dtype():
    """Get transformer dtype based on arguments"""
    if args.fp16:
        return torch.float16
    return torch.bfloat16


def get_model_filename(model_type, quantization='', dtype_policy=''):
    """Get model filename based on type and quantization"""
    # Get model definition
    model_def = get_model_def(model_type)
    
    # For ltxv_distilled, we need to get the base model
    if model_def.get("URLs") == "ltxv_13B":
        # Get base model definition
        base_model_def = get_model_def("ltxv_13B")
        urls = base_model_def.get("URLs", [])
    else:
        urls = model_def.get("URLs", [])
    
    if not urls:
        raise ValueError(f"No URLs found for model {model_type}")
    
    # Select appropriate URL based on quantization
    if quantization == "int8" and len(urls) > 1:
        # Use quantized version if available
        return urls[1]
    else:
        # Use base version
        return urls[0]


def get_ltxv_text_encoder_filename(quantization):
    """Get text encoder filename based on quantization"""
    # Check which text encoder files are available
    int8_path = "ckpts/T5_xxl_1.1/T5_xxl_1.1_enc_quanto_bf16_int8.safetensors"
    bf16_path = "ckpts/T5_xxl_1.1/T5_xxl_1.1_enc_bf16.safetensors"
    
    if quantization == "int8" and os.path.exists(int8_path):
        return int8_path
    else:
        # Fallback to bf16 version which is available
        return bf16_path


def load_image_to_tensor_with_resize_and_crop(
    image_input: Union[str, Image.Image],
    target_height: int = 512,
    target_width: int = 768,
    just_crop: bool = False,
) -> torch.Tensor:
    """Load and process an image into a tensor - matching ltxv.py exactly"""
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input
    else:
        raise ValueError("image_input must be either a file path or a PIL Image object")

    input_width, input_height = image.size
    aspect_ratio_target = target_width / target_height
    aspect_ratio_frame = input_width / input_height
    if aspect_ratio_frame > aspect_ratio_target:
        new_width = int(input_height * aspect_ratio_target)
        new_height = input_height
        x_start = (input_width - new_width) // 2
        y_start = 0
    else:
        new_width = input_width
        new_height = int(input_width / aspect_ratio_target)
        x_start = 0
        y_start = (input_height - new_height) // 2

    image = image.crop((x_start, y_start, x_start + new_width, y_start + new_height))
    if not just_crop:
        image = image.resize((target_width, target_height))

    image = np.array(image)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    frame_tensor = torch.from_numpy(image).float()
    frame_tensor = crf_compressor.compress(frame_tensor / 255.0) * 255.0
    frame_tensor = frame_tensor.permute(2, 0, 1)
    frame_tensor = (frame_tensor / 127.5) - 1.0
    # Create 5D tensor: (batch_size=1, channels=3, num_frames=1, height, width)
    return frame_tensor.unsqueeze(0).unsqueeze(2)


def load_model():
    """Load the LTX Video model and supporting models"""
    global model, device, offloadobj, transformer_quantization
    global prompt_enhancer_image_caption_model, prompt_enhancer_image_caption_processor
    global prompt_enhancer_llm_model, prompt_enhancer_llm_tokenizer
    
    try:
        # Set CUDA settings for reproducibility
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        logger.info("Set CUDA deterministic mode for reproducible results")
        
        # Set attention mode
        supported_modes = get_supported_attention_modes()
        logger.info(f"Available attention modes: {supported_modes}")
        
        # Use shared state for attention (matching wgp.py)
        offload.shared_state["_attention"] = attention_mode
        logger.info(f"Set attention mode to {attention_mode}")
        
        # Log GPU info
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"Using GPU: {gpu_name}")
            logger.info(f"VRAM available: {vram_gb:.1f} GB")
        
        # Get model definition
        model_type = args.model
        logger.info(f"Loading model type: {model_type}")
        
        model_def = get_model_def(model_type)
        
        # Get model filepath
        base_model_filename = get_model_filename(model_type, transformer_quantization)
        
        # Handle file paths
        if base_model_filename.startswith("http"):
            # For URLs, use just the filename
            model_filepath = os.path.join("ckpts", os.path.basename(base_model_filename))
        else:
            model_filepath = base_model_filename
        
        logger.info(f"Using model file: {model_filepath}")
        
        # Get LoRA files if defined
        lora_files = model_def.get("loras", [])
        if lora_files:
            # Download/locate LoRA files
            lora_filepaths = []
            for lora_url in lora_files:
                if lora_url.startswith("http"):
                    lora_path = os.path.join("ckpts", os.path.basename(lora_url))
                else:
                    lora_path = lora_url
                lora_filepaths.append(lora_path)
            logger.info(f"Found LoRA files: {lora_filepaths}")
        else:
            lora_filepaths = []
        
        # Check if files exist
        if not os.path.exists(model_filepath):
            raise FileNotFoundError(f"Model file not found: {model_filepath}")
        
        for lora_path in lora_filepaths:
            if not os.path.exists(lora_path):
                raise FileNotFoundError(f"LoRA file not found: {lora_path}")
        
        # Get text encoder file
        text_encoder_filepath = get_ltxv_text_encoder_filename(text_encoder_quantization)
        
        if not os.path.exists(text_encoder_filepath):
            raise FileNotFoundError(f"Text encoder file not found: {text_encoder_filepath}")
        
        logger.info(f"Found text encoder file: {text_encoder_filepath}")
        
        # Initialize LTXV model with model_def
        logger.info("Initializing LTXV model...")
        model = LTXV(
            model_filepath=[model_filepath],  # Pass as list
            text_encoder_filepath=text_encoder_filepath,
            model_def=model_def,  # Pass model definition
            dtype=torch.bfloat16,
            VAE_dtype=torch.float32,  # Match wgp.py - use float32 for VAE
            mixed_precision_transformer=False  # Match wgp.py default
        )
        
        # Get pipeline components
        pipeline = model.pipeline
        pipe = {
            "transformer": pipeline.video_pipeline.transformer,
            "vae": pipeline.vae,
            "text_encoder": pipeline.video_pipeline.text_encoder,
            "latent_upsampler": pipeline.latent_upsampler
        }
        
        # Load prompt enhancement models
        logger.info("Loading prompt enhancement models...")
        try:
            # Florence 2 for image captioning
            if os.path.exists("ckpts/Florence2"):
                prompt_enhancer_image_caption_model = AutoModelForCausalLM.from_pretrained(
                    "ckpts/Florence2", 
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    device_map=device
                )
                prompt_enhancer_image_caption_processor = AutoProcessor.from_pretrained(
                    "ckpts/Florence2", 
                    trust_remote_code=True
                )
                logger.info("Loaded Florence 2 for image captioning")
            else:
                logger.warning("Florence2 model not found, prompt enhancement disabled")
            
            # Llama 3.2 for prompt enhancement
            if os.path.exists("ckpts/Llama3_2/Llama3_2_quanto_bf16_int8.safetensors"):
                prompt_enhancer_llm_model = offload.fast_load_transformers_model(
                    "ckpts/Llama3_2/Llama3_2_quanto_bf16_int8.safetensors"
                )
                prompt_enhancer_llm_tokenizer = AutoTokenizer.from_pretrained("ckpts/Llama3_2")
                logger.info("Loaded Llama 3.2 for prompt enhancement")
            else:
                logger.warning("Llama3_2 model not found, LLM enhancement disabled")
                
        except Exception as e:
            logger.warning(f"Failed to load prompt enhancement models: {e}")
        
        # Add prompt enhancement models to pipe if loaded
        if prompt_enhancer_image_caption_model:
            pipe["prompt_enhancer_image_caption_model"] = prompt_enhancer_image_caption_model
            # Assign to pipeline for use in generation
            model.pipeline.video_pipeline.prompt_enhancer_image_caption_model = prompt_enhancer_image_caption_model
            model.pipeline.video_pipeline.prompt_enhancer_image_caption_processor = prompt_enhancer_image_caption_processor
        if prompt_enhancer_llm_model:
            pipe["prompt_enhancer_llm_model"] = prompt_enhancer_llm_model
            model.pipeline.video_pipeline.prompt_enhancer_llm_model = prompt_enhancer_llm_model
            model.pipeline.video_pipeline.prompt_enhancer_llm_tokenizer = prompt_enhancer_llm_tokenizer
        
        logger.info("Assigned prompt enhancement models to pipeline")
        
        # Profile configuration based on wgp.py
        kwargs = {}
        if profile in (2, 4, 5):
            preload = 40000  # 40GB preload
            kwargs["budgets"] = {"transformer": 100 if preload == 0 else preload, 
                               "text_encoder": 100 if preload == 0 else preload, 
                               "*": max(1000 if profile == 5 else 3000, preload)}
        elif profile == 3:
            kwargs["budgets"] = {"*": "70%"}
        
        # Apply offload profile
        try:
            offloadobj = offload.profile(
                pipe, 
                profile_no=profile, 
                compile="", 
                quantizeTransformer=False, 
                loras="transformer",  # Enable LoRA support
                coTenantsMap={}, 
                perc_reserved_mem_max=0.3,  # Default from wgp.py
                convertWeightsFloatTo=torch.bfloat16,
                **kwargs
            )
            logger.info(f"Set offload profile to HighRAM_HighVRAM (profile {profile})")
            
            # Now load LoRA if we have one (after offload profile is set)
            if lora_filepaths:
                try:
                    logger.info(f"Loading LoRA weights from: {lora_filepaths}")
                    lora_multipliers = model_def.get("loras_multipliers", [1.0] * len(lora_filepaths))
                    offload.load_loras_into_model(
                        pipe["transformer"],
                        lora_filepaths,
                        lora_multipliers,
                        activate_all_loras=True
                    )
                    logger.info(f"LoRA weights applied successfully")
                except Exception as e:
                    logger.error(f"Failed to apply LoRA weights: {e}")
                    
        except Exception as e:
            logger.warning(f"Failed to set offload profile: {e}. Continuing without profile optimization.")
            offloadobj = None
        
        logger.info("Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


async def video_generation_worker():
    """Background worker for processing video generation tasks"""
    while True:
        try:
            # Get task from queue
            task_id = await task_queue.get()
            
            if task_id is None:
                break
                
            task_info = generation_tasks.get(task_id)
            if not task_info:
                continue
                
            # Update status
            task_info.status = "processing"
            task_info.updated_at = datetime.now()
            
            # Process the task
            try:
                logger.info(f"Starting processing for task {task_id}")
                
                # Generate video
                await generate_video_for_task(task_id, task_info)
                
                # Update status
                task_info.status = "completed"
                
            except Exception as e:
                logger.error(f"Error processing task {task_id}: {str(e)}")
                task_info.status = "failed"
                task_info.error = str(e)
                
            task_info.updated_at = datetime.now()
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in video generation worker: {str(e)}")


async def generate_video_for_task(task_id: str, task_info: TaskInfo):
    """Generate video for a specific task"""
    try:
        start_time = time.time()
        
        # Download image
        input_image = await download_image(task_info.request.image_url)
        
        # Get resolution
        width, height = SUPPORTED_RESOLUTIONS[task_info.request.aspect_ratio]
        
        # Process image with same method as ltxv.py
        # This is crucial for maintaining quality throughout the video
        processed_image_tensor = load_image_to_tensor_with_resize_and_crop(
            input_image, 
            target_height=height, 
            target_width=width,
            just_crop=False
        )
        # Convert back to PIL for model.generate (it will process again internally)
        # For now, keep using the original image until we implement conditioning_items
        
        # Handle seed
        print(f"Pre defined seed: {task_info.request.seed}")
        seed = task_info.request.seed if task_info.request.seed != -1 else int(random.randint(0, 999999999))
        print(f"New seed: {seed}")
        
        # Enhance prompt if requested
        prompt_to_use = task_info.request.prompt
        if task_info.request.enhance_prompt_with_llm:
            prompt_to_use = await enhance_prompt(task_info.request.prompt, input_image)
        
        # Prepare generation parameters - matching wgp.py exactly
        generation_params = {
            "input_prompt": prompt_to_use,
            "n_prompt": task_info.request.negative_prompt,
            "image_start": input_image,
            "image_end": None,
            "input_video": None,  # wgp.py uses pre_video_guide here for LTXV
            "denoising_strength": 1.0,  # wgp.py uses this for LTXV
            "sampling_steps": 50,
            "image_cond_noise_scale": 0.0,  # Pipeline default, not ltxv.py default
            "seed": seed,
            "strength": 1.0,  # Image conditioning strength
            "height": height,
            "width": width,
            "frame_num": MAX_FRAMES,
            "frame_rate": 30,
            "fit_into_canvas": True,
            "device": str(device),
            "VAE_tile_size": (0, 0),
        }
        
        # Generate video
        start_time = time.time()
        async with model_lock:
            logger.info(f"Generating video for task {task_id}...")
            logger.info(f"Generation params: resolution={width}x{height}, frames={MAX_FRAMES}, seed={seed}")
            loop = asyncio.get_event_loop()
            try:
                video_tensor = await loop.run_in_executor(
                    None,
                    lambda: model.generate(**generation_params)
                )
            except Exception as e:
                logger.error(f"Error in model.generate: {str(e)}")
                logger.error(f"Error type: {type(e).__name__}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise
        
        if video_tensor is None:
            raise Exception("Video generation failed - model returned None")
        
        # Save video
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "video.mp4")
        save_video_tensor(video_tensor, output_path)
        
        # Update task info
        task_info.temp_dir = temp_dir
        task_info.video_url = f"/download/{task_id}"
        task_info.processing_time = time.time() - start_time
        
        logger.info(f"Task {task_id} completed in {task_info.processing_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Error generating video: {str(e)}")
        raise


async def download_image(url: str) -> Image.Image:
    """Download image from URL"""
    logger.info(f"Downloading image from: {url}")
    response = requests.get(url)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content)).convert("RGB")
    logger.info(f"Image downloaded successfully: {image.size}")
    return image


async def enhance_prompt(prompt: str, image: Image.Image) -> str:
    """Enhance prompt using image captioning and LLM"""
    try:
        # Generate image caption
        if prompt_enhancer_image_caption_model and prompt_enhancer_image_caption_processor:
            logger.info("Generating image caption...")
            
            # Prepare image for Florence
            inputs = prompt_enhancer_image_caption_processor(
                text="<MORE_DETAILED_CAPTION>", 
                images=image, 
                return_tensors="pt"
            ).to(device)
            
            # Generate caption
            with torch.no_grad():
                generated_ids = prompt_enhancer_image_caption_model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    early_stopping=False,
                    do_sample=False,
                    num_beams=3,
                )
            
            # Decode caption
            generated_text = prompt_enhancer_image_caption_processor.batch_decode(
                generated_ids, 
                skip_special_tokens=False
            )[0]
            
            # Parse Florence output
            parsed_answer = prompt_enhancer_image_caption_processor.post_process_generation(
                generated_text,
                task="<MORE_DETAILED_CAPTION>",
                image_size=image.size
            )
            
            image_caption = parsed_answer["<MORE_DETAILED_CAPTION>"]
            logger.info(f"Generated image caption: {image_caption}")
        else:
            image_caption = ""
        
        # Enhance with LLM
        if prompt_enhancer_llm_model and prompt_enhancer_llm_tokenizer and image_caption:
            logger.info("Enhancing prompt with LLM...")
            
            # Prepare prompt for Llama
            system_prompt = """You are a creative assistant that enhances video generation prompts. 
Your task is to take a simple prompt and an image description, then create a detailed, vivid prompt that will generate high-quality videos.
Focus on movement, atmosphere, lighting, and cinematic qualities. Keep the output under 150 words."""
            
            llm_input = f"""Image description: {image_caption}
User prompt: {prompt}

Create an enhanced video generation prompt that brings this scene to life with movement and atmosphere:"""
            
            # Tokenize
            inputs = prompt_enhancer_llm_tokenizer(
                llm_input,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(device)
            
            # Generate
            with torch.no_grad():
                outputs = prompt_enhancer_llm_model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.9,
                )
            
            # Decode
            enhanced_prompt = prompt_enhancer_llm_tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            # Clean up the output
            enhanced_prompt = enhanced_prompt.replace(llm_input, "").strip()
            
            logger.info(f"Enhanced prompt: {enhanced_prompt}")
            return enhanced_prompt
        
        # If enhancement failed, return original prompt
        return prompt
        
    except Exception as e:
        logger.error(f"Error enhancing prompt: {e}")
        return prompt


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "WGP API",
        "version": "1.0.0",
        "model": args.model,
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "queue_size": task_queue.qsize()
    }


@app.post("/generate", response_model=VideoGenerationResponse)
async def generate_video(request: VideoGenerationRequest):
    """Queue a video generation task"""
    
    # Validate aspect ratio
    if request.aspect_ratio not in SUPPORTED_RESOLUTIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid aspect ratio. Supported: {list(SUPPORTED_RESOLUTIONS.keys())}"
        )
    
    # Create task
    task_id = str(uuid.uuid4())
    task_info = TaskInfo(task_id, request)
    generation_tasks[task_id] = task_info
    
    # Queue for processing
    await task_queue.put(task_id)
    
    # Log queue status
    queue_size = task_queue.qsize()
    logger.info(f"Task {task_id} added to queue at position {queue_size}")
    
    return VideoGenerationResponse(
        task_id=task_id,
        status="pending",
        message=f"Task queued for processing (position: {queue_size})"
    )


@app.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Check the status of a video generation task"""
    
    task_info = generation_tasks.get(task_id)
    
    if not task_info:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Calculate queue position if still pending
    queue_position = None
    if task_info.status == "pending":
        # Count how many tasks are ahead in queue
        queue_items = list(task_queue._queue)
        if task_id in queue_items:
            queue_position = queue_items.index(task_id) + 1
    
    return TaskStatusResponse(
        task_id=task_id,
        status=task_info.status,
        video_url=task_info.video_url,
        error=task_info.error,
        processing_time=task_info.processing_time,
        queue_position=queue_position,
        created_at=task_info.created_at.isoformat(),
        updated_at=task_info.updated_at.isoformat()
    )


@app.get("/download/{task_id}")
async def download_video(task_id: str):
    """Download generated video"""
    task_info = generation_tasks.get(task_id)
    
    if not task_info:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task_info.status != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Video not ready. Current status: {task_info.status}"
        )
    
    if not task_info.temp_dir:
        raise HTTPException(status_code=404, detail="Video file not found")
    
    video_path = os.path.join(task_info.temp_dir, "video.mp4")
    
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"ltx_video_{task_id}.mp4"
    )


def save_video_tensor(tensor: torch.Tensor, output_path: str, fps: int = 30):
    """Save video tensor to file"""
    import cv2
    
    # Convert tensor to numpy array
    # First convert BFloat16 to Float32 as numpy doesn't support BFloat16
    # Tensor shape: (C, T, H, W) -> (T, H, W, C)
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    video_np = tensor.permute(1, 2, 3, 0).cpu().numpy()
    
    # Denormalize from [-1, 1] to [0, 255]
    video_np = ((video_np + 1) * 127.5).clip(0, 255).astype(np.uint8)
    
    # Get dimensions
    num_frames, height, width, channels = video_np.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames
    for frame in video_np:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
    
    writer.release()
    logger.info(f"Video saved: {num_frames} frames at {fps}fps")


@app.get("/queue")
async def get_queue_status():
    """Get current queue status"""
    pending_tasks = [task_id for task_id, task in generation_tasks.items() 
                     if task.status == "pending"]
    processing_tasks = [task_id for task_id, task in generation_tasks.items() 
                        if task.status == "processing"]
    completed_tasks = [task_id for task_id, task in generation_tasks.items() 
                       if task.status == "completed"]
    failed_tasks = [task_id for task_id, task in generation_tasks.items() 
                    if task.status == "failed"]
    
    return {
        "queue_size": task_queue.qsize(),
        "pending": len(pending_tasks),
        "processing": len(processing_tasks),
        "completed": len(completed_tasks),
        "failed": len(failed_tasks),
        "total_tasks": len(generation_tasks)
    }


# Cleanup old tasks periodically
async def cleanup_old_tasks():
    """Clean up old completed tasks and their temporary files"""
    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour
            
            current_time = datetime.now()
            tasks_to_remove = []
            
            for task_id, task_info in generation_tasks.items():
                # Remove tasks older than 24 hours
                if (current_time - task_info.created_at).total_seconds() > 86400:
                    tasks_to_remove.append(task_id)
                    
                    # Clean up temp directory
                    if task_info.temp_dir and os.path.exists(task_info.temp_dir):
                        shutil.rmtree(task_info.temp_dir)
            
            # Remove old tasks
            for task_id in tasks_to_remove:
                del generation_tasks[task_id]
                
            if tasks_to_remove:
                logger.info(f"Cleaned up {len(tasks_to_remove)} old tasks")
                
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")


if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )