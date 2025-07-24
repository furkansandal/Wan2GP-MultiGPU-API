# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import binascii
import os
import os.path as osp
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import cv2
import tempfile
import imageio
import torch
import decord
import torchvision
from PIL import Image
import numpy as np
from rembg import remove, new_session
import random
import ffmpeg
import os
import tempfile

__all__ = ['cache_video', 'cache_image', 'str2bool']



from PIL import Image

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def expand_slist(slist, num_inference_steps ):
    new_slist= []
    inc =  len(slist) / num_inference_steps 
    pos = 0
    for i in range(num_inference_steps):
        new_slist.append(slist[ int(pos)])
        pos += inc
    return new_slist

def update_loras_slists(trans, slists, num_inference_steps ):
    from mmgp import offload
    slists = [ expand_slist(slist, num_inference_steps ) if isinstance(slist, list) else slist for slist in slists ]
    nos = [str(l) for l in range(len(slists))]
    offload.activate_loras(trans, nos, slists ) 

def resample(video_fps, video_frames_count, max_target_frames_count, target_fps, start_target_frame ):
    import math

    video_frame_duration = 1 /video_fps
    target_frame_duration = 1 / target_fps 
    
    target_time = start_target_frame * target_frame_duration
    frame_no = math.ceil(target_time / video_frame_duration)  
    cur_time = frame_no * video_frame_duration
    frame_ids =[]
    while True:
        if max_target_frames_count != 0 and len(frame_ids) >= max_target_frames_count :
            break
        diff = round( (target_time -cur_time) / video_frame_duration , 5)
        add_frames_count = math.ceil( diff)
        frame_no += add_frames_count
        if frame_no >= video_frames_count:             
            break
        frame_ids.append(frame_no)
        cur_time += add_frames_count * video_frame_duration
        target_time += target_frame_duration
    frame_ids = frame_ids[:max_target_frames_count]
    return frame_ids

import os
from datetime import datetime

def get_file_creation_date(file_path):
    # On Windows
    if os.name == 'nt':
        return datetime.fromtimestamp(os.path.getctime(file_path))
    # On Unix/Linux/Mac (gets last status change, not creation)
    else:
        stat = os.stat(file_path)
    return datetime.fromtimestamp(stat.st_birthtime if hasattr(stat, 'st_birthtime') else stat.st_mtime)

def truncate_for_filesystem(s, max_bytes=255):
    if len(s.encode('utf-8')) <= max_bytes: return s
    l, r = 0, len(s)
    while l < r:
        m = (l + r + 1) // 2
        if len(s[:m].encode('utf-8')) <= max_bytes: l = m
        else: r = m - 1
    return s[:l]

def get_video_info(video_path):
    import cv2
    cap = cv2.VideoCapture(video_path)
    
    # Get FPS
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    
    # Get resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
    cap.release()
    
    return fps, width, height, frame_count

def get_video_frame(file_name, frame_no):
    decord.bridge.set_bridge('torch')
    reader = decord.VideoReader(file_name)

    frame = reader.get_batch([frame_no]).squeeze(0)
    img = Image.fromarray(frame.numpy().astype(np.uint8))
    return img

def convert_image_to_video(image):
    if image is None:
        return None
    
    # Convert PIL/numpy image to OpenCV format if needed
    if isinstance(image, np.ndarray):
        # Gradio images are typically RGB, OpenCV expects BGR
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        # Handle PIL Image
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    height, width = img_bgr.shape[:2]
    
    # Create temporary video file (auto-cleaned by Gradio)
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video.name, fourcc, 30.0, (width, height))
        out.write(img_bgr)
        out.release()
        return temp_video.name
    
def resize_lanczos(img, h, w):
    img = Image.fromarray(np.clip(255. * img.movedim(0, -1).cpu().numpy(), 0, 255).astype(np.uint8))
    img = img.resize((w,h), resample=Image.Resampling.LANCZOS) 
    return torch.from_numpy(np.array(img).astype(np.float32) / 255.0).movedim(-1, 0)


def remove_background(img, session=None):
    if session ==None:
        session = new_session() 
    img = Image.fromarray(np.clip(255. * img.movedim(0, -1).cpu().numpy(), 0, 255).astype(np.uint8))
    img = remove(img, session=session, alpha_matting = True, bgcolor=[255, 255, 255, 0]).convert('RGB')
    return torch.from_numpy(np.array(img).astype(np.float32) / 255.0).movedim(-1, 0)

def convert_tensor_to_image(t, frame_no = -1):    
    t = t[:, frame_no] if frame_no >= 0 else t
    return Image.fromarray(t.clone().add_(1.).mul_(127.5).permute(1,2,0).to(torch.uint8).cpu().numpy())

def save_image(tensor_image, name, frame_no = -1):
    convert_tensor_to_image(tensor_image, frame_no).save(name)

def get_outpainting_full_area_dimensions(frame_height,frame_width, outpainting_dims):
    outpainting_top, outpainting_bottom, outpainting_left, outpainting_right= outpainting_dims
    frame_height = int(frame_height * (100 + outpainting_top + outpainting_bottom) / 100)
    frame_width =  int(frame_width * (100 + outpainting_left + outpainting_right) / 100)
    return frame_height, frame_width  

def  get_outpainting_frame_location(final_height, final_width,  outpainting_dims, block_size = 8):
    outpainting_top, outpainting_bottom, outpainting_left, outpainting_right= outpainting_dims
    raw_height = int(final_height / ((100 + outpainting_top + outpainting_bottom) / 100))
    height = int(raw_height / block_size) * block_size
    extra_height = raw_height - height
          
    raw_width = int(final_width / ((100 + outpainting_left + outpainting_right) / 100)) 
    width = int(raw_width / block_size) * block_size
    extra_width = raw_width - width  
    margin_top = int(outpainting_top/(100 + outpainting_top + outpainting_bottom) * final_height)
    if extra_height != 0 and (outpainting_top + outpainting_bottom) != 0:
        margin_top += int(outpainting_top / (outpainting_top + outpainting_bottom) * extra_height)
    if (margin_top + height) > final_height or outpainting_bottom == 0: margin_top = final_height - height
    margin_left = int(outpainting_left/(100 + outpainting_left + outpainting_right) * final_width)
    if extra_width != 0 and (outpainting_left + outpainting_right) != 0:
        margin_left += int(outpainting_left / (outpainting_left + outpainting_right) * extra_height)
    if (margin_left + width) > final_width or outpainting_right == 0: margin_left = final_width - width
    return height, width, margin_top, margin_left

def calculate_new_dimensions(canvas_height, canvas_width, height, width, fit_into_canvas, block_size = 16):
    if fit_into_canvas == None:
        return height, width
    if fit_into_canvas:
        scale1  = min(canvas_height / height, canvas_width / width)
        scale2  = min(canvas_width / height, canvas_height / width)
        scale = max(scale1, scale2) 
    else:
        scale = (canvas_height * canvas_width / (height * width))**(1/2)

    new_height = round( height * scale / block_size) * block_size
    new_width = round( width * scale / block_size) * block_size
    return new_height, new_width

def resize_and_remove_background(img_list, budget_width, budget_height, rm_background, ignore_first, fit_into_canvas = False ):
    if rm_background:
        session = new_session() 

    output_list =[]
    for i, img in enumerate(img_list):
        width, height =  img.size 

        if fit_into_canvas:
            white_canvas = np.ones((budget_height, budget_width, 3), dtype=np.uint8) * 255 
            scale = min(budget_height / height, budget_width / width)
            new_height = int(height * scale)
            new_width = int(width * scale)
            resized_image= img.resize((new_width,new_height), resample=Image.Resampling.LANCZOS) 
            top = (budget_height - new_height) // 2
            left = (budget_width - new_width) // 2
            white_canvas[top:top + new_height, left:left + new_width] = np.array(resized_image)            
            resized_image = Image.fromarray(white_canvas)  
        else:
            scale = (budget_height * budget_width / (height * width))**(1/2)
            new_height = int( round(height * scale / 16) * 16)
            new_width = int( round(width * scale / 16) * 16)
            resized_image= img.resize((new_width,new_height), resample=Image.Resampling.LANCZOS) 
        if rm_background  and not (ignore_first and i == 0) :
            # resized_image = remove(resized_image, session=session, alpha_matting_erode_size = 1,alpha_matting_background_threshold = 70, alpha_foreground_background_threshold = 100, alpha_matting = True, bgcolor=[255, 255, 255, 0]).convert('RGB')
            resized_image = remove(resized_image, session=session, alpha_matting_erode_size = 1, alpha_matting = True, bgcolor=[255, 255, 255, 0]).convert('RGB')
        output_list.append(resized_image) #alpha_matting_background_threshold = 30, alpha_foreground_background_threshold = 200,
    return output_list


def rand_name(length=8, suffix=''):
    name = binascii.b2a_hex(os.urandom(length)).decode('utf-8')
    if suffix:
        if not suffix.startswith('.'):
            suffix = '.' + suffix
        name += suffix
    return name


def cache_video(tensor,
                save_file=None,
                fps=30,
                suffix='.mp4',
                nrow=8,
                normalize=True,
                value_range=(-1, 1),
                retry=5):
    # cache file
    cache_file = osp.join('/tmp', rand_name(
        suffix=suffix)) if save_file is None else save_file

    # save to cache
    error = None
    for _ in range(retry):
        try:
            # preprocess
            tensor = tensor.clamp(min(value_range), max(value_range))
            tensor = torch.stack([
                torchvision.utils.make_grid(
                    u, nrow=nrow, normalize=normalize, value_range=value_range)
                for u in tensor.unbind(2)
            ],
                                 dim=1).permute(1, 2, 3, 0)
            tensor = (tensor * 255).type(torch.uint8).cpu()

            # write video
            writer = imageio.get_writer(
                cache_file, fps=fps, codec='libx264', quality=8)
            for frame in tensor.numpy():
                writer.append_data(frame)
            writer.close()
            return cache_file
        except Exception as e:
            error = e
            continue
    else:
        print(f'cache_video failed, error: {error}', flush=True)
        return None


def cache_image(tensor,
                save_file,
                nrow=8,
                normalize=True,
                value_range=(-1, 1),
                retry=5):
    # cache file
    suffix = osp.splitext(save_file)[1]
    if suffix.lower() not in [
            '.jpg', '.jpeg', '.png', '.tiff', '.gif', '.webp'
    ]:
        suffix = '.png'

    # save to cache
    error = None
    for _ in range(retry):
        try:
            tensor = tensor.clamp(min(value_range), max(value_range))
            torchvision.utils.save_image(
                tensor,
                save_file,
                nrow=nrow,
                normalize=normalize,
                value_range=value_range)
            return save_file
        except Exception as e:
            error = e
            continue


def str2bool(v):
    """
    Convert a string to a boolean.

    Supported true values: 'yes', 'true', 't', 'y', '1'
    Supported false values: 'no', 'false', 'f', 'n', '0'

    Args:
        v (str): String to convert.

    Returns:
        bool: Converted boolean value.

    Raises:
        argparse.ArgumentTypeError: If the value cannot be converted to boolean.
    """
    if isinstance(v, bool):
        return v
    v_lower = v.lower()
    if v_lower in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v_lower in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected (True/False)')


import sys, time

# Global variables to track download progress
_start_time = None
_last_time = None
_last_downloaded = 0
_speed_history = []
_update_interval = 0.5  # Update speed every 0.5 seconds

def progress_hook(block_num, block_size, total_size, filename=None):
    """
    Simple progress bar hook for urlretrieve
    
    Args:
        block_num: Number of blocks downloaded so far
        block_size: Size of each block in bytes
        total_size: Total size of the file in bytes
        filename: Name of the file being downloaded (optional)
    """
    global _start_time, _last_time, _last_downloaded, _speed_history, _update_interval
    
    current_time = time.time()
    downloaded = block_num * block_size
    
    # Initialize timing on first call
    if _start_time is None or block_num == 0:
        _start_time = current_time
        _last_time = current_time
        _last_downloaded = 0
        _speed_history = []
    
    # Calculate download speed only at specified intervals
    speed = 0
    if current_time - _last_time >= _update_interval:
        if _last_time > 0:
            current_speed = (downloaded - _last_downloaded) / (current_time - _last_time)
            _speed_history.append(current_speed)
            # Keep only last 5 speed measurements for smoothing
            if len(_speed_history) > 5:
                _speed_history.pop(0)
            # Average the recent speeds for smoother display
            speed = sum(_speed_history) / len(_speed_history)
        
        _last_time = current_time
        _last_downloaded = downloaded
    elif _speed_history:
        # Use the last calculated average speed
        speed = sum(_speed_history) / len(_speed_history)
    # Format file sizes and speed
    def format_bytes(bytes_val):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_val < 1024:
                return f"{bytes_val:.1f}{unit}"
            bytes_val /= 1024
        return f"{bytes_val:.1f}TB"
    
    file_display = filename if filename else "Unknown file"
    
    if total_size <= 0:
        # If total size is unknown, show downloaded bytes
        speed_str = f" @ {format_bytes(speed)}/s" if speed > 0 else ""
        line = f"\r{file_display}: {format_bytes(downloaded)}{speed_str}"
        # Clear any trailing characters by padding with spaces
        sys.stdout.write(line.ljust(80))
        sys.stdout.flush()
        return
    
    downloaded = block_num * block_size
    percent = min(100, (downloaded / total_size) * 100)
    
    # Create progress bar (40 characters wide to leave room for other info)
    bar_length = 40
    filled = int(bar_length * percent / 100)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    # Format file sizes and speed
    def format_bytes(bytes_val):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_val < 1024:
                return f"{bytes_val:.1f}{unit}"
            bytes_val /= 1024
        return f"{bytes_val:.1f}TB"
    
    speed_str = f" @ {format_bytes(speed)}/s" if speed > 0 else ""
    
    # Display progress with filename first
    line = f"\r{file_display}: [{bar}] {percent:.1f}% ({format_bytes(downloaded)}/{format_bytes(total_size)}){speed_str}"
    # Clear any trailing characters by padding with spaces
    sys.stdout.write(line.ljust(100))
    sys.stdout.flush()
    
    # Print newline when complete
    if percent >= 100:
        print()

# Wrapper function to include filename in progress hook
def create_progress_hook(filename):
    """Creates a progress hook with the filename included"""
    global _start_time, _last_time, _last_downloaded, _speed_history
    # Reset timing variables for new download
    _start_time = None
    _last_time = None
    _last_downloaded = 0
    _speed_history = []
    
    def hook(block_num, block_size, total_size):
        return progress_hook(block_num, block_size, total_size, filename)
    return hook

import ffmpeg
import os
import tempfile

def extract_audio_tracks(source_video, verbose=False, query_only= False):
    """
    Extract all audio tracks from source video to temporary files.
    
    Args:
        source_video: Path to video with audio to extract
        verbose: Enable verbose output (default: False)
        
    Returns:
        List of temporary audio file paths, or empty list if no audio tracks
    """
    try:
        # Check if source video has audio
        probe = ffmpeg.probe(source_video)
        audio_streams = [s for s in probe['streams'] if s['codec_type'] == 'audio']
        
        if not audio_streams:
            if query_only: return 0
            if verbose:
                print(f"No audio track found in {source_video}")
            return []
        if query_only: return len(audio_streams)
        if verbose:
            print(f"Found {len(audio_streams)} audio track(s)")
        
        # Create temporary audio files for each track
        temp_audio_files = []
        for i in range(len(audio_streams)):
            fd, temp_path = tempfile.mkstemp(suffix=f'_track{i}.aac', prefix='audio_')
            os.close(fd)  # Close file descriptor immediately
            temp_audio_files.append(temp_path)
        
        # Extract each audio track
        for i, temp_path in enumerate(temp_audio_files):
            (ffmpeg
             .input(source_video)
             .output(temp_path, **{f'map': f'0:a:{i}', 'acodec': 'aac'})
             .overwrite_output()
             .run(quiet=not verbose))
        
        return temp_audio_files
        
    except ffmpeg.Error as e:
        print(f"FFmpeg error during audio extraction: {e}")
        return 0 if query_only else []
    except Exception as e:
        print(f"Error during audio extraction: {e}")
        return 0 if query_only else []

def combine_video_with_audio_tracks(target_video, audio_tracks, output_video, verbose=False):
    """
    Combine video with audio tracks. Output duration matches video length exactly.
    
    Args:
        target_video: Path to video to receive the audio
        audio_tracks: List of audio file paths to combine
        output_video: Path for the output video
        verbose: Enable verbose output (default: False)
        
    Returns:
        True if successful, False otherwise
    """
    if not audio_tracks:
        if verbose:
            print("No audio tracks to combine")
        return False
    
    try:
        # Get video duration to ensure exact alignment
        video_probe = ffmpeg.probe(target_video)
        video_duration = float(video_probe['streams'][0]['duration'])
        
        if verbose:
            print(f"Target video duration: {video_duration:.3f} seconds")
        
        # Combine target video with all audio tracks, force video duration
        video = ffmpeg.input(target_video).video
        audio_inputs = [ffmpeg.input(audio_path).audio for audio_path in audio_tracks]
        
        # Create output with video duration as master timing
        inputs = [video] + audio_inputs
        (ffmpeg
         .output(*inputs, output_video, 
                vcodec='copy', 
                acodec='copy', 
                t=video_duration)  # Force exact video duration
         .overwrite_output()
         .run(quiet=not verbose))
        
        if verbose:
            print(f"Successfully created {output_video} with {len(audio_tracks)} audio track(s) aligned to video duration")
        return True
        
    except ffmpeg.Error as e:
        print(f"FFmpeg error during video combination: {e}")
        return False
    except Exception as e:
        print(f"Error during video combination: {e}")
        return False

def cleanup_temp_audio_files(audio_tracks, verbose=False):
    """
    Clean up temporary audio files.
    
    Args:
        audio_tracks: List of audio file paths to delete
        verbose: Enable verbose output (default: False)
        
    Returns:
        Number of files successfully deleted
    """
    deleted_count = 0
    
    for audio_path in audio_tracks:
        try:
            if os.path.exists(audio_path):
                os.unlink(audio_path)
                deleted_count += 1
                if verbose:
                    print(f"Cleaned up {audio_path}")
        except PermissionError:
            print(f"Warning: Could not delete {audio_path} (file may be in use)")
        except Exception as e:
            print(f"Warning: Error deleting {audio_path}: {e}")
    
    if verbose and deleted_count > 0:
        print(f"Successfully deleted {deleted_count} temporary audio file(s)")
    
    return deleted_count

