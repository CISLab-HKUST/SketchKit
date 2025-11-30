"""PhotoSketch programmatic API.

This module exposes a :class:`PhotoSketchModel` that mirrors the high level
contract used by other ``image2sketch`` backends. It wraps the original
PhotoSketch Pix2Pix generator while managing checkpoint discovery and
downloading automatically.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

import torch
from torchvision.transforms import functional as TF

try:  # pragma: no cover - exercised when gdown is not installed
    import gdown  # type: ignore
except ImportError:  # pragma: no cover
    gdown = None  # type: ignore

from .models import networks

ImageInput = Union[Image.Image, np.ndarray, str, os.PathLike]
ImageSize = Union[int, Tuple[int, int]]

_BICUBIC = getattr(Image, "Resampling", Image).BICUBIC  # type: ignore[attr-defined]


class PhotoSketchModel:
    """High level wrapper around the PhotoSketch Pix2Pix generator."""

    MODEL_DRIVE_ID = "1TQf-LyS8rRDDapdcTnEgWzYJllPgiXdj"
    MODEL_DOWNLOAD_URL = (
        f"https://drive.google.com/uc?export=download&id={MODEL_DRIVE_ID}"
    )
    MODEL_ARCHIVE_NAME = "photosketch_pretrained.zip"
    GENERATOR_FILENAME = "latest_net_G.pth"
    GENERATOR_SHA1 = "5968e8f007c650008a265c11f2d2a3887e5840d4"

    INPUT_CHANNELS = 3
    OUTPUT_CHANNELS = 1
    NUM_FILTERS = 64
    RESNET_BLOCKS = 9

    DEFAULT_MODEL_NAME = "pretrained"
    DEFAULT_CHECKPOINT_DIR = (
        Path.home() / ".cache" / "sketchkit" / "image2sketch" / "photosketch"
    )
    DEFAULT_INPUT_SIZE = 256

    def __init__(
        self,
        *,
        checkpoint_dir: Optional[Union[str, os.PathLike]] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        auto_download: bool = True,
        device: Optional[Union[str, torch.device]] = None,
        input_size: int = DEFAULT_INPUT_SIZE,
    ) -> None:
        self._checkpoint_root = (
            Path(checkpoint_dir)
            if checkpoint_dir is not None
            else self.DEFAULT_CHECKPOINT_DIR
        )
        self._model_dir = self._checkpoint_root / model_name
        self._auto_download = auto_download
        self._device = self._resolve_device(device)
        self._input_size = int(input_size)
        if self._input_size <= 0:
            raise ValueError("input_size must be a positive integer.")

        module_dir = Path(__file__).resolve().parent
        self._local_generator_paths = [
            module_dir / self.GENERATOR_FILENAME,
            module_dir / "assets" / self.GENERATOR_FILENAME,
        ]
        self._local_archive_paths = [
            module_dir / self.MODEL_ARCHIVE_NAME,
            module_dir / "assets" / self.MODEL_ARCHIVE_NAME,
        ]

        self._generator_path = self._model_dir / self.GENERATOR_FILENAME
        self._generator: Optional[torch.nn.Module] = None

        self._mean = (0.5, 0.5, 0.5)
        self._std = (0.5, 0.5, 0.5)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    @property
    def checkpoint_dir(self) -> Path:
        return self._checkpoint_root

    def ensure_assets(self) -> None:
        """Ensure that the pretrained PhotoSketch weights exist locally."""

        self._model_dir.mkdir(parents=True, exist_ok=True)
        if self._generator_path.exists() and self._validate_sha1(
            self._generator_path, self.GENERATOR_SHA1
        ):
            return

        if self._install_from_local_file():
            return

        if self._install_from_local_archive():
            return

        if not self._auto_download:
            raise FileNotFoundError(
                f"PhotoSketch generator weights missing at {self._generator_path}. "
                "Enable auto_download or provide the checkpoint manually."
            )

        self._download_and_extract()

        if not self._generator_path.exists():
            raise FileNotFoundError(
                f"Failed to obtain PhotoSketch generator at {self._generator_path}."
            )

        if not self._validate_sha1(self._generator_path, self.GENERATOR_SHA1):
            raise RuntimeError(
                "Downloaded PhotoSketch generator failed integrity verification."
            )

    def generate(
        self,
        image: Union[ImageInput, Sequence[ImageInput]],
        *,
        size: ImageSize = 512,
    ) -> Union[Image.Image, List[Image.Image]]:
        """Generate sketch images for the provided image input(s)."""

        if isinstance(image, Sequence) and not isinstance(
            image, (str, bytes, os.PathLike, np.ndarray, Image.Image)
        ):
            images = list(image)
            if not images:
                return []
        else:
            images = [image]

        results = self.generate_batch(images, size=size)
        return results if len(images) > 1 else results[0]

    def generate_batch(
        self,
        images: Sequence[ImageInput],
        *,
        size: ImageSize = 512,
    ) -> List[Image.Image]:
        """Generate sketches for a batch of images."""

        if not images:
            return []

        width, height = self._resolve_size(size)
        generator = self._initialise_generator()

        outputs: List[Image.Image] = []
        with torch.inference_mode():
            for source in images:
                pil_image = self._to_pil_image(source)
                tensor = self._prepare_tensor(pil_image)
                prediction = generator(tensor)[0]
                sketch = self._tensor_to_image(prediction, (width, height))
                outputs.append(sketch)

        return outputs

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _initialise_generator(self) -> torch.nn.Module:
        if self._generator is not None:
            return self._generator

        self.ensure_assets()
        if not self._generator_path.exists():
            raise FileNotFoundError(
                f"PhotoSketch generator not found at {self._generator_path}."
            )

        state_dict = torch.load(self._generator_path, map_location=self._device)

        norm_layer = networks.get_norm_layer("batch")
        generator = networks.ResnetGenerator(
            self.INPUT_CHANNELS,
            self.OUTPUT_CHANNELS,
            ngf=self.NUM_FILTERS,
            norm_layer=norm_layer,
            use_dropout=False,
            n_blocks=self.RESNET_BLOCKS,
        )
        generator.load_state_dict(state_dict)
        generator.to(self._device)
        generator.eval()
        generator.requires_grad_(False)

        self._generator = generator
        return generator

    def _prepare_tensor(self, image: Image.Image) -> torch.Tensor:
        resized = image.resize((self._input_size, self._input_size), _BICUBIC)
        tensor = TF.to_tensor(resized)
        tensor = TF.normalize(tensor, self._mean, self._std)
        return tensor.unsqueeze(0).to(self._device)

    def _tensor_to_image(
        self,
        tensor: torch.Tensor,
        size: Tuple[int, int],
    ) -> Image.Image:
        array = tensor.squeeze().detach().cpu().numpy()
        array = np.clip((array + 1.0) * 0.5 * 255.0, 0, 255).astype(np.uint8)
        image = Image.fromarray(array, mode="L")
        if image.size != size:
            image = image.resize(size, _BICUBIC)
        return image

    def _download_and_extract(self) -> None:
        if gdown is None:
            raise ImportError(
                "PhotoSketch auto-download requires the 'gdown' package. "
                "Install it or provide the weights manually."
            )

        self._model_dir.mkdir(parents=True, exist_ok=True)

        tmp_dir = Path(tempfile.mkdtemp(prefix="photosketch-"))
        try:
            archive_path = tmp_dir / self.MODEL_ARCHIVE_NAME
            gdown.download(self.MODEL_DOWNLOAD_URL, str(archive_path), quiet=False)

            if not archive_path.exists():
                raise FileNotFoundError(
                    f"Failed to download PhotoSketch archive to {archive_path}."
                )

            self._extract_archive(archive_path)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _install_from_local_file(self) -> bool:
        for candidate in self._local_generator_paths:
            if not candidate.exists():
                continue
            if not self._validate_sha1(candidate, self.GENERATOR_SHA1):
                raise RuntimeError(
                    f"Local PhotoSketch generator at {candidate} failed integrity verification."
                )
            shutil.copy2(candidate, self._generator_path)
            return True
        return False

    def _install_from_local_archive(self) -> bool:
        for archive in self._local_archive_paths:
            if not archive.exists():
                continue
            self._extract_archive(archive)
            if self._generator_path.exists() and self._validate_sha1(
                self._generator_path, self.GENERATOR_SHA1
            ):
                return True
            raise RuntimeError(
                f"Extracted PhotoSketch weights from {archive} but integrity verification failed."
            )
        return False

    def _extract_archive(self, archive_path: Path) -> None:
        with zipfile.ZipFile(archive_path) as zf:
            try:
                with zf.open(self.GENERATOR_FILENAME) as src, open(
                    self._generator_path, "wb"
                ) as dst:
                    shutil.copyfileobj(src, dst)
            except KeyError as exc:  # pragma: no cover - defensive guard
                raise FileNotFoundError(
                    f"Generator checkpoint '{self.GENERATOR_FILENAME}' not found inside archive."
                ) from exc

    @staticmethod
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
        return int(width), int(height)

    @staticmethod
    def _to_pil_image(image: ImageInput) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, (str, os.PathLike)):
            return Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            array = image
            if array.ndim == 2:
                array = np.stack([array] * 3, axis=-1)
            if array.ndim == 3 and array.shape[2] == 4:
                array = array[:, :, :3]
            if array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)
            return Image.fromarray(array).convert("RGB")
        raise TypeError(
            "Unsupported image input. Expected PIL.Image, numpy array, or a file path."
        )

    @staticmethod
    def _validate_sha1(path: Path, expected: str) -> bool:
        sha1 = hashlib.sha1()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                sha1.update(chunk)
        return sha1.hexdigest() == expected

    @staticmethod
    def _resolve_device(
        device: Optional[Union[str, torch.device]]
    ) -> torch.device:
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(device, torch.device):
            return device
        return torch.device(device)


__all__ = ["PhotoSketchModel"]
