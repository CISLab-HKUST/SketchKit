"""HED (Holistically Nested Edge Detection) integration.

This module wraps the original Caffe implementation using OpenCV's DNN loader.
It mirrors the public API of other ``image2sketch`` backends so callers can
request HED outputs via ``image2sketch(..., method="hed")``.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
from typing import Any, List, Optional, Sequence, Tuple, Union
import urllib.request

import numpy as np
from PIL import Image

try:  # pragma: no cover - exercised in environments without OpenCV
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore

ImageInput = Union[Image.Image, np.ndarray, str, os.PathLike]
ImageSize = Union[int, Tuple[int, int]]

from .postprocess import DEFAULT_POSTPROCESS_PARAMS, PostprocessParams, apply_postprocess


def _resolve_size(size: ImageSize) -> Tuple[int, int]:
    if isinstance(size, int):
        if size <= 0:
            raise ValueError("size must be a positive integer.")
        return size, size

    if len(size) != 2:
        raise ValueError("size tuple must contain exactly two integers (width, height).")
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("size dimensions must be positive integers.")
    return width, height


class HEDModel:
    """High level wrapper around the original HED Caffe model."""

    MODEL_URL = "https://vcl.ucsd.edu/hed/hed_pretrained_bsds.caffemodel"
    MODEL_FILENAME = "hed_pretrained_bsds.caffemodel"
    MODEL_SHA1 = "2c5d7842f25f880eec62fc610b500c5cf2aa351d"

    MEAN_BGR = (104.00698793, 116.66876762, 122.67891434)

    def __init__(
        self,
        *,
        checkpoint_dir: Optional[Union[str, os.PathLike]] = None,
        auto_download: bool = True,
        postprocess: bool = True,
        postprocess_threshold: float = DEFAULT_POSTPROCESS_PARAMS.threshold,
        postprocess_small_edge: int = DEFAULT_POSTPROCESS_PARAMS.small_edge,
    ) -> None:
        self._auto_download = auto_download
        self._checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir is not None else Path.home() / ".cache" / "sketchkit" / "image2sketch" / "hed"
        )
        self._asset_dir = Path(__file__).resolve().parent / "assets"
        self._deploy_path = self._asset_dir / "deploy.prototxt"
        self._weights_path = self._checkpoint_dir / self.MODEL_FILENAME
        self._local_asset_path = self._asset_dir / self.MODEL_FILENAME

        if not self._deploy_path.exists():
            raise FileNotFoundError(f"HED deploy prototxt missing: {self._deploy_path}")

        self._net: Optional[Any] = None
        self._postprocess_enabled = postprocess
        self._postprocess_params = PostprocessParams(
            threshold=postprocess_threshold,
            small_edge=postprocess_small_edge,
        )

    # ------------------------------------------------------------------
    # public helpers
    # ------------------------------------------------------------------
    @property
    def checkpoint_dir(self) -> Path:
        return self._checkpoint_dir

    def ensure_assets(self) -> None:
        """Ensure the pretrained weights are available locally."""

        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self._weights_path.exists() and self._validate_sha1(self._weights_path, self.MODEL_SHA1):
            return

        if self._local_asset_path.exists():
            if not self._validate_sha1(self._local_asset_path, self.MODEL_SHA1):
                raise RuntimeError(
                    f"Local HED weights at {self._local_asset_path} failed integrity verification."
                )
            shutil.copy2(self._local_asset_path, self._weights_path)
            return

        if not self._auto_download:
            raise FileNotFoundError(
                f"HED weights missing at {self._weights_path}. "
                "Enable auto_download or download manually from "
                f"{self.MODEL_URL}."
            )

        self._download_weights(self.MODEL_URL, self._weights_path, self.MODEL_SHA1)

    def generate(
        self,
        image: Union[ImageInput, Sequence[ImageInput]],
        *,
        size: ImageSize = 512,
    ) -> Union[Image.Image, List[Image.Image]]:
        """Generate edge maps for the provided image(s)."""

        images: List[ImageInput]
        if isinstance(image, Sequence) and not isinstance(image, (str, bytes, os.PathLike, np.ndarray, Image.Image)):
            images = list(image)
            if not images:
                return []
        else:
            images = [image]

        net = self._initialise_net()

        results: List[Image.Image] = []
        width, height = _resolve_size(size)
        for source in images:
            pil_image = self._to_pil_image(source)
            edge_map = self._infer_single(net, pil_image, (width, height))
            results.append(edge_map)

        return results if len(results) > 1 else results[0]

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _initialise_net(self) -> Any:
        if cv2 is None:
            raise RuntimeError(
                "HEDModel requires the 'opencv-python' package. Install it to enable the method."
            )

        if self._net is not None:
            return self._net

        self.ensure_assets()
        if not self._weights_path.exists():
            raise FileNotFoundError(
                f"HED weights not found at {self._weights_path}. "
                "Download from the official repository if auto-download is disabled."
            )

        net = cv2.dnn.readNetFromCaffe(str(self._deploy_path), str(self._weights_path))
        self._net = net
        return net

    def _infer_single(
        self,
        net: Any,
        image: Image.Image,
        size: Tuple[int, int],
    ) -> Image.Image:
        if cv2 is None:
            raise RuntimeError(
                "HEDModel requires the 'opencv-python' package. Install it to enable the method."
            )

        image_rgb = image.convert("RGB")
        image_array = np.asarray(image_rgb).astype(np.float32)

        # Official HED workflow expects 500x500 input tensors.
        network_width, network_height = 500, 500
        resized_array = cv2.resize(image_array, (network_width, network_height), interpolation=cv2.INTER_LINEAR)

        blob = cv2.dnn.blobFromImage(
            resized_array,
            scalefactor=1.0,
            mean=self.MEAN_BGR,
            swapRB=True,
            crop=False,
        )
        net.setInput(blob)
        out = net.forward()  # shape: (1, 1, H, W)

        edge_map = out[0, 0]
        # Apply crop strategy discussed at https://github.com/s9xie/hed/issues/33
        crop = int(np.ceil((edge_map.shape[0] - network_height) / 2.0))
        if crop > 0:
            edge_map = edge_map[crop:-crop or None, crop:-crop or None]

        if edge_map.shape[:2] != (network_height, network_width):
            edge_map = cv2.resize(edge_map, (network_width, network_height), interpolation=cv2.INTER_LINEAR)

        edge_map = np.clip(edge_map, 0.0, 1.0)
        if self._postprocess_enabled:
            edge_map = apply_postprocess(edge_map, params=self._postprocess_params)

        edge_uint8 = (edge_map * 255).astype(np.uint8)
        pil_edges = Image.fromarray(edge_uint8, mode="L")

        target_w, target_h = size
        if (target_w, target_h) != (network_width, network_height):
            pil_edges = pil_edges.resize(size, Image.BILINEAR)
        return pil_edges

    @staticmethod
    def _to_pil_image(image: ImageInput) -> Image.Image:
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, (str, os.PathLike)):
            return Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            array = image
            if array.ndim == 2:
                array = np.stack([array] * 3, axis=-1)
            if array.ndim != 3 or array.shape[2] not in (3, 4):
                raise ValueError("numpy image inputs must have shape (H, W, 3) or (H, W, 4).")
            if array.shape[2] == 4:
                array = array[:, :, :3]
            if array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)
            return Image.fromarray(array, mode="RGB")
        raise TypeError(
            "Unsupported image input. Expected PIL.Image, numpy array, or a file path."
        )

    @staticmethod
    def _validate_sha1(path: Path, expected_hex: str) -> bool:
        sha1 = hashlib.sha1()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                sha1.update(chunk)
        return sha1.hexdigest() == expected_hex

    @staticmethod
    def _download_weights(url: str, destination: Path, expected_sha1: str) -> None:
        tmp_path = destination.with_suffix(".tmp")
        try:
            with urllib.request.urlopen(url) as response, tmp_path.open("wb") as fh:
                while True:
                    chunk = response.read(16 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

        if not HEDModel._validate_sha1(tmp_path, expected_sha1):
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(
                "Downloaded HED weights failed integrity check. "
                "Please retry or download manually."
            )

        tmp_path.replace(destination)


__all__ = ["HEDModel"]
