import os
import re
import time
from dataclasses import dataclass
from glob import iglob
from mmgp import offload as offload
import torch
from wan.utils.utils import calculate_new_dimensions
from flux.sampling import denoise, get_schedule, prepare_kontext, unpack
from flux.modules.layers import get_linear_split_map
from flux.util import (
    aspect_ratio_to_height_width,
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
    save_image,
)

class model_factory:
    def __init__(
        self,
        checkpoint_dir,
        model_filename = None,
        model_type = None, 
        base_model_type = None,
        text_encoder_filename = None,
        quantizeTransformer = False,
        save_quantized = False,
        dtype = torch.bfloat16,
        VAE_dtype = torch.float32,
        mixed_precision_transformer = False
    ):
        self.device = torch.device(f"cuda")
        self.VAE_dtype = VAE_dtype
        self.dtype = dtype
        torch_device = "cpu"

        self.t5 = load_t5(torch_device, text_encoder_filename, max_length=512)
        self.clip = load_clip(torch_device)
        self.name= "flux-dev-kontext"
        self.model = load_flow_model(self.name, model_filename[0], torch_device)

        self.vae = load_ae(self.name, device=torch_device)

        # offload.change_dtype(self.model, dtype, True)
        if save_quantized:
            from wgp import save_quantized_model
            save_quantized_model(self.model, model_type, model_filename[0], dtype, None)

        split_linear_modules_map = get_linear_split_map()
        self.model.split_linear_modules_map = split_linear_modules_map
        offload.split_linear_modules(self.model, split_linear_modules_map )

    
    def generate(
            self,
            seed: int | None = None,
            input_prompt: str = "replace the logo with the text 'Black Forest Labs'",
            sampling_steps: int = 20,
            input_ref_images = None,
            width= 832,
            height=480,
            guide_scale: float = 2.5,
            fit_into_canvas = None,
            callback = None,
            loras_slists = None,
            batch_size = 1,
            **bbargs
    ):
            
            if self._interrupt:
                return None

            rng = torch.Generator(device="cuda")
            if seed is None:
                seed = rng.seed()

            if input_ref_images != None and len(input_ref_images) > 0: 
                image_ref = input_ref_images[0]
                w, h = image_ref.size
                height, width = calculate_new_dimensions(height, width, h, w, fit_into_canvas)

            inp, height, width = prepare_kontext(
                t5=self.t5,
                clip=self.clip,
                prompt=input_prompt,
                ae=self.vae,
                img_cond=image_ref,
                target_width=width,
                target_height=height,
                bs=batch_size,
                seed=seed,
                device="cuda",
            )

            inp.pop("img_cond_orig")
            timesteps = get_schedule(sampling_steps, inp["img"].shape[1], shift=(self.name != "flux-schnell"))
            def unpack_latent(x):
                return unpack(x.float(), height, width) 
            # denoise initial noise
            x = denoise(self.model, **inp, timesteps=timesteps, guidance=guide_scale, callback=callback, pipeline=self, loras_slists= loras_slists, unpack_latent = unpack_latent)
            if x==None: return None
            # decode latents to pixel space
            x = unpack_latent(x)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                x = self.vae.decode(x)

            x = x.clamp(-1, 1)
            x = x.transpose(0, 1)
            return x

