"""
T2I-Adapter Model Implementation
"""

import torch
from diffusers import StableDiffusionAdapterPipeline, T2IAdapter
from PIL import Image
from typing import Tuple, Union, Dict, Any, Optional
import warnings


class T2IAdapterModel:
    """T2I-Adapter model for edge-conditioned sketch-to-image conversion."""
    
    def __init__(
        self,
        adapter_model: str = "TencentARC/t2iadapter_canny_sd15v2",
        sd_model: str = "runwayml/stable-diffusion-v1-5",
        torch_dtype: torch.dtype = torch.float16,
        safety_checker: bool = False,
        **kwargs: Any,
    ):
        print(f"Loading T2I-Adapter model: {adapter_model}")
        
        try:
            adapter = T2IAdapter.from_pretrained(
                adapter_model,
                torch_dtype=torch_dtype,
                **kwargs
            )
            
            if safety_checker:
                self.pipeline = StableDiffusionAdapterPipeline.from_pretrained(
                    sd_model,
                    adapter=adapter,
                    torch_dtype=torch_dtype,
                    **kwargs
                )
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.pipeline = StableDiffusionAdapterPipeline.from_pretrained(
                        sd_model,
                        adapter=adapter,
                        torch_dtype=torch_dtype,
                        safety_checker=None,
                        **kwargs
                    )
            
            if hasattr(self.pipeline, 'enable_attention_slicing'):
                self.pipeline.enable_attention_slicing()
            
            if torch.cuda.is_available():
                self.pipeline = self.pipeline.to("cuda")
            
            self.adapter_model = adapter_model
            self.sd_model = sd_model
            self.torch_dtype = torch_dtype
            
            print(f"✅ T2I-Adapter model loaded successfully")
            print(f"   - Adapter: {adapter_model}")
            print(f"   - Base model: {sd_model}")
            
        except ImportError as e:
            raise ImportError(
                "T2I-Adapter requires diffusers>=0.14.0. "
                "Install with: pip install diffusers transformers accelerate"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load T2I-Adapter model: {e}") from e
    
    def generate(
        self,
        control_image: Image.Image,
        prompt: str,
        size: Union[int, Tuple[int, int]] = 512,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        adapter_conditioning_scale: float = 1.0,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> Image.Image:
        """Generate image from edge-conditioned sketch."""
        # 从kwargs中移除已显式处理的参数和diffusers 0.35.2不支持的参数
        kwargs_copy = kwargs.copy()
        kwargs_copy.pop('num_steps', None)
        kwargs_copy.pop('guidance_scale', None)
        kwargs_copy.pop('adapter_scale', None)
        kwargs_copy.pop('adapter_conditioning_factor', None)  # 移除不支持的参数
        
        if control_image.mode != 'L':
            raise ValueError(
                f"T2I-Adapter requires grayscale edge images (mode='L'), "
                f"got mode='{control_image.mode}'"
            )
        
        generator = None
        if seed is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=device).manual_seed(seed)
        
        try:
            result = self.pipeline(
                prompt=prompt,
                image=control_image,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                adapter_conditioning_scale=adapter_conditioning_scale,
                generator=generator,
                height=size if isinstance(size, int) else size[1],
                width=size if isinstance(size, int) else size[0],
                **kwargs_copy,
            )
            
            return result.images[0]
            
        except Exception as e:
            raise RuntimeError(f"T2I-Adapter generation failed: {e}") from e