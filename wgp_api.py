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
from typing import Optional
from datetime import datetime
import requests
from io import BytesIO
from PIL import Image
import numpy as np

# FastAPI imports
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import project modules
from mmgp import offload, profile_type
from ltx_video.ltxv import LTXV
from wan.modules.attention import get_supported_attention_modes
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

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

# Constants
MAX_FRAMES = 129
SUPPORTED_RESOLUTIONS = {
    "9:16": (405, 720),   # 720p portrait
    "16:9": (1280, 720),  # 720p landscape
    "1:1": (720, 720),    # Square
    "3:4": (540, 720),    # Portrait
    "4:3": (960, 720)     # Traditional
}

DEFAULT_NEGATIVE_PROMPT = "low quality, worst quality, deformed, distorted, disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, weird hand, ugly"

# Global variables for model
model = None
device = None
model_lock = asyncio.Lock()
prompt_enhancer_image_caption_model = None
prompt_enhancer_image_caption_processor = None
prompt_enhancer_llm_model = None
prompt_enhancer_llm_tokenizer = None

# Task queue for video generation
generation_tasks = {}
task_queue = asyncio.Queue()
worker_task = None

# FastAPI app
app = FastAPI(
    title="WGP LTX Video API",
    description="API for LTX Video 0.9.7 Distilled 13B inference",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class VideoGenerationRequest(BaseModel):
    """Request model for video generation"""
    prompt: str = Field(..., description="Text prompt for video generation")
    image_url: str = Field(..., description="URL of the input image")
    aspect_ratio: str = Field(default="16:9", description="Aspect ratio (9:16, 16:9, 1:1, 3:4, 4:3)")
    negative_prompt: Optional[str] = Field(default=DEFAULT_NEGATIVE_PROMPT, description="Negative prompt")
    seed: int = Field(default=-1, description="Random seed (-1 for random)")
    enhance_prompt_with_llm: bool = Field(default=True, description="Enhance prompt using LLM")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "A serene landscape with mountains",
                "image_url": "https://example.com/image.jpg",
                "aspect_ratio": "16:9",
                "seed": -1
            }
        }


class VideoGenerationResponse(BaseModel):
    """Response model for video generation"""
    task_id: str
    status: str
    message: Optional[str] = None
    queue_position: Optional[int] = None


class TaskStatusResponse(BaseModel):
    """Response model for task status check"""
    task_id: str
    status: str  # pending, processing, completed, failed
    video_url: Optional[str] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None
    queue_position: Optional[int] = None
    created_at: str
    updated_at: str


class TaskInfo:
    """Internal task information"""
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


async def download_image(url: str) -> Image.Image:
    """Download image from URL and return PIL Image"""
    try:
        logger.info(f"Downloading image from: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('RGB')
        logger.info(f"Image downloaded successfully: {img.size}")
        return img
    except Exception as e:
        logger.error(f"Error downloading image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")


async def enhance_prompt(prompt: str, image: Image.Image) -> str:
    """Enhance prompt using LLM based on both text and image"""
    if not prompt_enhancer_llm_model or not prompt_enhancer_image_caption_model:
        logger.warning("Prompt enhancement models not available")
        return prompt
    
    try:
        enhanced_prompt = prompt
        
        # First, get image caption using Florence 2
        if prompt_enhancer_image_caption_model and prompt_enhancer_image_caption_processor:
            logger.info("Generating image caption...")
            inputs = prompt_enhancer_image_caption_processor(
                text="<MORE_DETAILED_CAPTION>", 
                images=image, 
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                generated_ids = prompt_enhancer_image_caption_model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=1024,
                    num_beams=3
                )
            
            generated_text = prompt_enhancer_image_caption_processor.batch_decode(
                generated_ids, 
                skip_special_tokens=False
            )[0]
            
            # Extract caption
            parsed_answer = prompt_enhancer_image_caption_processor.post_process_generation(
                generated_text,
                task="<MORE_DETAILED_CAPTION>",
                image_size=(image.width, image.height)
            )
            image_caption = parsed_answer.get('<MORE_DETAILED_CAPTION>', '')
            logger.info(f"Generated image caption: {image_caption}")
        
        # Then enhance with Llama 3.2
        if prompt_enhancer_llm_model and prompt_enhancer_llm_tokenizer:
            logger.info("Enhancing prompt with LLM...")
            
            # Construct the enhancement prompt
            system_prompt = """You are a creative assistant that enhances video generation prompts. 
Given a user prompt and an image description, create a detailed, vivid prompt that will generate a compelling video.
Keep the enhanced prompt concise but descriptive, focusing on movement, atmosphere, and visual details."""
            
            enhancement_prompt = f"""System: {system_prompt}

Image Description: {image_caption}
User Prompt: {prompt}

Enhanced Prompt:"""
            
            inputs = prompt_enhancer_llm_tokenizer(
                enhancement_prompt, 
                return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                outputs = prompt_enhancer_llm_model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9
                )
            
            enhanced_prompt = prompt_enhancer_llm_tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            logger.info(f"Enhanced prompt: {enhanced_prompt}")
        
        return enhanced_prompt if enhanced_prompt else prompt
        
    except Exception as e:
        logger.error(f"Error enhancing prompt: {str(e)}")
        return prompt


def load_model():
    """Load LTX Video model with specified configurations"""
    global model, device, prompt_enhancer_image_caption_model, prompt_enhancer_image_caption_processor
    global prompt_enhancer_llm_model, prompt_enhancer_llm_tokenizer
    
    logger.info("Starting model loading...")
    
    # Set attention mode to xformers
    available_modes = get_supported_attention_modes()
    logger.info(f"Available attention modes: {available_modes}")
    if 'xformers' in available_modes:
        # Set attention mode using offload shared state
        offload.shared_state["_attention"] = 'xformers'
        logger.info("Set attention mode to xformers")
    else:
        # Use auto mode as fallback
        offload.shared_state["_attention"] = 'auto'
        logger.warning("xformers not available, using auto attention mode")
    
    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        logger.info(f"VRAM available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available, using CPU (will be slow)")
    
    try:
        # Model paths
        model_files = [
            "ckpts/ltxv_0.9.7_13B_distilled_bf16.safetensors",
            "ckpts/ltxv_0.9.7_13B_distilled_quanto_bf16_int8.safetensors",
            "ckpts/ltxv_0.9.7_13B_distilled_lora128_bf16.safetensors"
        ]
        
        text_encoder_files = [
            "ckpts/T5_xxl_1.1/T5_xxl_1.1_enc_bf16.safetensors",
            "ckpts/T5_xxl_1.1/T5_xxl_1.1_enc_quanto_bf16_int8.safetensors"
        ]
        
        # Check which model files exist
        model_filepath = None
        for file in model_files:
            if os.path.exists(file):
                model_filepath = [file]
                logger.info(f"Found model file: {file}")
                break
        
        if not model_filepath:
            logger.error(f"No model files found. Checked: {model_files}")
            raise FileNotFoundError("Model files not found")
        
        # Check text encoder
        text_encoder_filepath = None
        for file in text_encoder_files:
            if os.path.exists(file):
                text_encoder_filepath = file
                logger.info(f"Found text encoder file: {file}")
                break
        
        if not text_encoder_filepath:
            logger.error(f"No text encoder files found. Checked: {text_encoder_files}")
            raise FileNotFoundError("Text encoder files not found")
        
        # Initialize model with BF16 precision
        logger.info("Initializing LTXV model...")
        model = LTXV(
            model_filepath=model_filepath,
            text_encoder_filepath=text_encoder_filepath,
            dtype=torch.bfloat16,
            VAE_dtype=torch.bfloat16,
            mixed_precision_transformer=True  # 16-bit transformer calculation
        )
        
        # Load prompt enhancement models
        logger.info("Loading prompt enhancement models...")
        try:
            # Florence 2 for image captioning
            if os.path.exists("ckpts/Florence2"):
                prompt_enhancer_image_caption_model = AutoModelForCausalLM.from_pretrained(
                    "ckpts/Florence2", 
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16
                ).to(device)
                prompt_enhancer_image_caption_processor = AutoProcessor.from_pretrained(
                    "ckpts/Florence2", 
                    trust_remote_code=True
                )
                logger.info("Loaded Florence 2 for image captioning")
            else:
                logger.warning("Florence2 model not found, prompt enhancement disabled")
            
            # Llama 3.2 for text enhancement
            if os.path.exists("ckpts/Llama3_2/Llama3_2_quanto_bf16_int8.safetensors"):
                prompt_enhancer_llm_model = offload.fast_load_transformers_model(
                    "ckpts/Llama3_2/Llama3_2_quanto_bf16_int8.safetensors"
                )
                prompt_enhancer_llm_tokenizer = AutoTokenizer.from_pretrained("ckpts/Llama3_2")
                logger.info("Loaded Llama 3.2 for prompt enhancement")
            else:
                logger.warning("Llama 3.2 model not found, LLM enhancement disabled")
                
        except Exception as e:
            logger.warning(f"Failed to load prompt enhancement models: {e}")
        
        # Configure offload settings for H100 80GB
        if hasattr(offload, 'set_max_vram_budget'):
            offload.set_max_vram_budget(40000)  # 40GB as specified
            logger.info("Set VRAM budget to 40GB")
        
        # Set up offload profile for HighRAM_HighVRAM
        offload.set_profile_type(profile_type.HighRAM_HighVRAM)
        logger.info("Set offload profile to HighRAM_HighVRAM")
        
        logger.info("Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


async def video_generation_worker():
    """Background worker for processing video generation tasks"""
    global generation_tasks
    
    while True:
        try:
            # Get task from queue
            task_id = await task_queue.get()
            task_info = generation_tasks.get(task_id)
            
            if not task_info:
                logger.error(f"Task {task_id} not found in generation_tasks")
                continue
            
            # Update status to processing
            task_info.status = "processing"
            task_info.updated_at = datetime.now()
            logger.info(f"Starting processing for task {task_id}")
            
            try:
                # Download image
                input_image = await download_image(task_info.request.image_url)
                
                # Get resolution
                width, height = SUPPORTED_RESOLUTIONS[task_info.request.aspect_ratio]
                
                # Handle seed
                seed = task_info.request.seed if task_info.request.seed != -1 else int(time.time())
                
                # Enhance prompt if requested
                prompt_to_use = task_info.request.prompt
                if task_info.request.enhance_prompt_with_llm:
                    prompt_to_use = await enhance_prompt(task_info.request.prompt, input_image)
                
                # Prepare generation parameters
                generation_params = {
                    "input_prompt": prompt_to_use,
                    "n_prompt": task_info.request.negative_prompt,
                    "image_start": input_image,
                    "image_end": None,
                    "input_video": None,
                    "sampling_steps": 7,
                    "image_cond_noise_scale": 0.15,
                    "seed": seed,
                    "height": height,
                    "width": width,
                    "frame_num": MAX_FRAMES,
                    "frame_rate": 30,
                    "fit_into_canvas": True,
                    "device": device,
                    "VAE_tile_size": None,
                }
                
                # Generate video
                start_time = time.time()
                async with model_lock:
                    logger.info(f"Generating video for task {task_id}...")
                    loop = asyncio.get_event_loop()
                    video_tensor = await loop.run_in_executor(
                        None,
                        lambda: model.generate(**generation_params)
                    )
                
                if video_tensor is None:
                    raise Exception("Video generation failed")
                
                # Save video
                temp_dir = tempfile.mkdtemp()
                output_path = os.path.join(temp_dir, "video.mp4")
                save_video_tensor(video_tensor, output_path, fps=30)
                
                # Update task info
                task_info.temp_dir = temp_dir
                task_info.video_url = f"/download/{task_id}"
                task_info.status = "completed"
                task_info.processing_time = time.time() - start_time
                task_info.updated_at = datetime.now()
                
                logger.info(f"Task {task_id} completed in {task_info.processing_time:.2f}s")
                
                # Schedule cleanup
                asyncio.create_task(cleanup_task(task_id))
                
            except Exception as e:
                logger.error(f"Error processing task {task_id}: {str(e)}")
                task_info.status = "failed"
                task_info.error = str(e)
                task_info.updated_at = datetime.now()
                
                # Schedule cleanup for failed tasks too
                asyncio.create_task(cleanup_task(task_id))
                
        except Exception as e:
            logger.error(f"Worker error: {str(e)}")
            await asyncio.sleep(1)


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global worker_task
    
    logger.info("Starting WGP API server...")
    load_model()
    
    # Start background worker
    worker_task = asyncio.create_task(video_generation_worker())
    logger.info("Video generation worker started")
    
    logger.info("API server ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global worker_task
    
    logger.info("Shutting down API server...")
    
    # Cancel worker task
    if worker_task:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
    
    # Cleanup model
    if model:
        del model
        torch.cuda.empty_cache()
        gc.collect()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "WGP LTX Video API",
        "version": "1.0.0",
        "model": "LTX Video 0.9.7 Distilled 13B",
        "status": "running" if model is not None else "model not loaded"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "model_loaded": model is not None
    }


@app.post("/generate", response_model=VideoGenerationResponse)
async def generate_video(request: VideoGenerationRequest):
    """Submit a video generation request"""
    
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate aspect ratio
    if request.aspect_ratio not in SUPPORTED_RESOLUTIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported aspect ratio. Supported: {list(SUPPORTED_RESOLUTIONS.keys())}"
        )
    
    # Generate task ID
    task_id = str(uuid.uuid4())
    
    # Create task info
    task_info = TaskInfo(task_id, request)
    generation_tasks[task_id] = task_info
    
    # Add to queue
    await task_queue.put(task_id)
    
    # Calculate queue position
    queue_position = task_queue.qsize()
    
    logger.info(f"Task {task_id} added to queue at position {queue_position}")
    
    return VideoGenerationResponse(
        task_id=task_id,
        status="pending",
        message="Task queued for processing",
        queue_position=queue_position
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
    # Tensor shape: (C, T, H, W) -> (T, H, W, C)
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


async def cleanup_task(task_id: str, delay: int = 600):
    """Clean up task data and files after delay"""
    await asyncio.sleep(delay)  # Wait 10 minutes
    
    task_info = generation_tasks.get(task_id)
    if task_info and task_info.temp_dir:
        try:
            shutil.rmtree(task_info.temp_dir)
            logger.info(f"Cleaned up temporary directory for task {task_id}")
        except Exception as e:
            logger.warning(f"Failed to clean up {task_info.temp_dir}: {str(e)}")
    
    # Remove task from memory
    generation_tasks.pop(task_id, None)
    logger.info(f"Removed task {task_id} from memory")


if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "wgp_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )