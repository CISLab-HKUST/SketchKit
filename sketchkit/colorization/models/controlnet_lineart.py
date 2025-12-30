"""
ControlNet Lineart Model for Sketch Colorization
"""

import torch
from diffusers import StableDiffusionControlNetPipeline
from diffusers import ControlNetModel as DiffusersControlNetModel
from PIL import Image
from typing import Tuple, Union, Dict, Any, Optional
import warnings


class ControlNetLineartModel:
    """ControlNet model with lineart conditioning for sketch colorization."""
    
    def __init__(
        self,
        controlnet_model: str = "lllyasviel/control_v11p_sd15_lineart",
        sd_model: str = "runwayml/stable-diffusion-v1-5",
        torch_dtype: torch.dtype = torch.float16,
        safety_checker: bool = False,
        **kwargs: Any,
    ):
        print(f"Loading ControlNet Lineart model: {controlnet_model}")
        
        controlnet = DiffusersControlNetModel.from_pretrained(
            controlnet_model,
            torch_dtype=torch_dtype,
            **kwargs
        )
        
        if safety_checker:
            self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                sd_model,
                controlnet=controlnet,
                torch_dtype=torch_dtype,
                **kwargs
            )
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                    sd_model,
                    controlnet=controlnet,
                    torch_dtype=torch_dtype,
                    safety_checker=None,
                    requires_safety_checker=False,
                    **kwargs
                )
        
        if torch.cuda.is_available():
            self.pipeline = self.pipeline.to("cuda")
        
        if hasattr(self.pipeline, 'enable_attention_slicing'):
            self.pipeline.enable_attention_slicing()
        
        self.controlnet_model = controlnet_model
        self.sd_model = sd_model
        self.torch_dtype = torch_dtype
        
        print("✅ ControlNet Lineart model loaded successfully")
    
    def generate(
        self,
        control_image: Image.Image,
        prompt: str,
        size: Union[int, Tuple[int, int]] = 512,
        negative_prompt: str = "blurry, low quality, distorted",
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Image.Image:
        """Generate colorized image from lineart sketch."""
        # 从kwargs中移除已显式处理的参数
        kwargs_copy = kwargs.copy()
        kwargs_copy.pop('num_steps', None)
        kwargs_copy.pop('guidance_scale', None)
        kwargs_copy.pop('controlnet_scale', None)
        kwargs_copy.pop('generator', None)  # 移除可能重复的generator
        
        generator = None
        if seed is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=device).manual_seed(seed)
        
        return self.pipeline(
            prompt=prompt,
            image=control_image,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator,
            height=size if isinstance(size, int) else size[1],
            width=size if isinstance(size, int) else size[0],
            **kwargs_copy,
        ).images[0]