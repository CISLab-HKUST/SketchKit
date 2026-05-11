from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageEnhance
import cv2

import torch
from torch.amp import autocast
import torch.nn.functional as F

from .network.line_extractor import LineExtractor

ImageInput = Union[Image.Image, np.ndarray, str, os.PathLike]
ImageSize = Union[int, Tuple[int, int]]

def increase_sharpness(img: np.ndarray, factor: float = 6.0) -> np.ndarray:
    image = Image.fromarray(img)
    enhancer = ImageEnhance.Sharpness(image)
    return np.array(enhancer.enhance(factor))

class AniLinesModel:
    """High level wrapper around the AniLines LineExtractor."""

    DEFAULT_CHECKPOINT_DIR = (
        Path.home() / ".cache" / "sketchkit" / "image2sketch" / "anilines"
    )

    def __init__(
        self,
        *,
        checkpoint_dir: Optional[Union[str, os.PathLike]] = None,
        mode: str = "basic",
        device: Optional[Union[str, torch.device]] = None,
        binarize: float = -1.0,
        fp16: bool = True,
    ) -> None:
        self._checkpoint_dir = (
            Path(checkpoint_dir)
            if checkpoint_dir is not None
            else self.DEFAULT_CHECKPOINT_DIR
        )
        if mode not in ("basic", "detail"):
            raise ValueError("mode must be either 'basic' or 'detail'")
        self.mode = mode
        self.binarize = binarize
        self.fp16 = fp16

        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        self._model: Optional[torch.nn.Module] = None

        self.MODEL_URLS = {
                "basic": "https://drive.google.com/uc?export=download&id=14Bp8mbQAbiR1rQrEsFp-uNdOou8hoCFr",
                "detail": "https://drive.google.com/uc?export=download&id=12U1Mwlonoipk2Yvr12mNaFB30foy420o"
            }

    def _resolve_model_path(self) -> Path:
        p = self._checkpoint_dir / f"{self.mode}.pth"
        if not p.exists():
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
            try:
                import gdown
                url = self.MODEL_URLS[self.mode]
                print(f"Downloading {self.mode}.pth from Google Drive...")
                gdown.download(url, str(p), quiet=False)
            except ImportError:
                raise ImportError("The 'gdown' package is required to download the weights automatically. Please install it with 'pip install gdown'.")
        return p

    def _initialise_net(self) -> torch.nn.Module:
        if self._model is not None:
            return self._model

        model_path = self._resolve_model_path()
        print("model path: ", model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"AniLines weights missing at {model_path}. Please place downloaded '{self.mode}.pth' in {self._checkpoint_dir}."
            )

        if self.mode == 'basic':
            model = LineExtractor(3, 1, True).to(self._device)
        else:
            model = LineExtractor(2, 1, True).to(self._device)
            
        model.load_state_dict(torch.load(str(model_path), map_location=self._device, weights_only=True))
        for param in model.parameters():
            param.requires_grad = False
        model.eval()
        self._model = model
        return model

    def _to_pil_image(self, image: ImageInput) -> Image.Image:
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, (str, os.PathLike)):
            return Image.open(image)
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                return Image.fromarray(image, "L")
            if image.ndim == 3 and image.shape[2] == 3:
                return Image.fromarray(image, "RGB")
            if image.ndim == 3 and image.shape[2] == 4:
                return Image.fromarray(image, "RGBA").convert("RGB")
        raise TypeError("Unsupported image input. Expected PIL.Image, numpy array, or a file path.")

    def _infer_single(self, image: Image.Image, size: Optional[ImageSize] = None) -> Image.Image:
        model = self._initialise_net()
        img_np = np.array(image.convert("RGB"))
        # CV2 uses BGR internally mostly, let's just make sure
        
        # logic from infer.py
        if self.mode == 'basic':
            # expects RGB
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) # convert to BGR first
            # img_rgb2 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # convert back to RGB ?? (that's what infer.py did: img = cv2.cvtColor(np.array(Image.open(path_in)), cv2.COLOR_RGB2BGR); then in inference: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # Just keepRGB
            img_sharp = increase_sharpness(img_np)
            t_img = torch.from_numpy(img_sharp).permute(2, 0, 1).unsqueeze(0).float().to(self._device) / 255.
            x_in = t_img
        else:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel = cv2.magnitude(sobelx, sobely)
            sobel = 255 - cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        
            t_img = torch.from_numpy(img_gray).unsqueeze(0).unsqueeze(0).float().to(self._device) / 255.
            t_sobel = torch.from_numpy(sobel).unsqueeze(0).unsqueeze(0).float().to(self._device) / 255.
        
            x_in = torch.cat([t_img, t_sobel], dim=1)

        B, C, H, W = x_in.shape
        pad_h = 8 - (H % 8)
        pad_w = 8 - (W % 8)
        x_in = F.pad(x_in, (0, pad_w, 0, pad_h), mode='reflect')

        with torch.no_grad():
            if self.fp16 and getattr(torch.cuda, "amp", None) and self._device.type == "cuda":
                with autocast(device_type='cuda'):
                    pred = model(x_in)
            else:
                pred = model(x_in)
                
        pred = pred[:, :, :H, :W]
        if self.binarize != -1:
            pred = (pred > self.binarize).float()
        
        out_np = np.clip((pred[0, 0].cpu().numpy() * 255) + 0.5, 0, 255).astype(np.uint8)
        out_pil = Image.fromarray(out_np, mode="L")
        
        if size is not None:
            if isinstance(size, int):
                size = (size, size)
            if out_pil.size != size:
                out_pil = out_pil.resize(size, Image.BILINEAR)
                
        return out_pil

    def generate(
        self,
        image: Union[ImageInput, Sequence[ImageInput]],
        *,
        size: Optional[ImageSize] = None,
    ) -> Union[Image.Image, List[Image.Image]]:
        images: List[ImageInput]
        if isinstance(image, Sequence) and not isinstance(image, (str, bytes, os.PathLike, np.ndarray, Image.Image)):
            images = list(image)
        else:
            images = [image]

        results = []
        for src in images:
            pil_img = self._to_pil_image(src)
            results.append(self._infer_single(pil_img, size=size))

        if len(results) == 1 and not isinstance(image, Sequence):
            return results[0]
        # if input was array or pil image, return single
        if len(results) == 1 and isinstance(image, (str, bytes, os.PathLike, np.ndarray, Image.Image)):
            return results[0]
        return results

