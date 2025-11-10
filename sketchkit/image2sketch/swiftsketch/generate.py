"""SwiftSketch programmatic API.

This module exposes the :class:`SwiftSketchModel` that powers the
``image2sketch(..., method="swiftsketch")`` workflow.  The implementation wraps the
original SwiftSketch diffusion pipeline while providing a higher level API
that handles checkpoint management and preprocessing automatically.
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from pathlib import Path as FsPath
from types import SimpleNamespace
from typing import Callable, List, Optional, Sequence, Tuple, Union
import zipfile

import gdown
import numpy as np
import torch
from PIL import Image

from sketchkit.core import Sketch, Path as SketchPath, Curve, Vertex, Point
from .utils.fixseed import fixseed
from .utils import dist_util, sketch_utils
from .utils.model_util import create_model_and_diffusion, load_model_wo_clip, create_model
from .model.cfg_sampler import ClassifierFreeSampleModel
from .model import image_features_models as image_feature_models
from transformers import AutoModelForImageSegmentation


__all__ = ["SwiftSketchModel"]


MODEL_ARCHIVE_URL = "https://drive.google.com/uc?export=download&id=19FryO99dCmz-Dw1jzeZITUI0uuksiOA-"
REFINE_ARCHIVE_URL = "https://drive.google.com/uc?export=download&id=1OrLzwaJXZ4SlDw3hqn71Yg1L01ytLv2x"
MODEL_ARCHIVE_NAME = "sketch-diffusion.zip"
REFINE_ARCHIVE_NAME = "refinement-network.zip"
MODEL_ARCHIVE_FOLDER = "sketch-diffusion"
REFINE_ARCHIVE_FOLDER = "refinement-network"
DEFAULT_CACHE_DIR = FsPath.home() / ".cache" / "sketchkit" / "image2sketch" / "swiftsketch"


ImageInput = Union[Image.Image, np.ndarray, str, os.PathLike]


@dataclass
class SwiftSketchArtifacts:
    """Auxiliary outputs generated alongside a SwiftSketch `Sketch`."""

    svg: str
    image: Image.Image
    control_points: torch.Tensor
    diffusion_control_points: torch.Tensor
    compute_device: torch.device


SWIFTSKETCH_METADATA_ATTR = "_swift_artifacts"


class SwiftSketchModel:
    """High level wrapper around the SwiftSketch diffusion pipeline."""

    def __init__(
        self,
        *,
        device: Optional[Union[str, torch.device]] = None,
        use_refine: bool = True,
        guidance_param: float = 2.5,
        fix_scale: bool = False,
        checkpoint_dir: Optional[Union[str, os.PathLike]] = None,
        auto_download: bool = True,
        mask_model_factory: Optional[Callable[[torch.device], torch.nn.Module]] = None,
        feature_extractor_factory: Optional[
            Callable[[torch.device, str], torch.nn.Module]
        ] = None,
        diffusion_factory: Optional[
            Callable[[SimpleNamespace], Tuple[torch.nn.Module, object]]
        ] = None,
        refine_model_factory: Optional[Callable[[SimpleNamespace], torch.nn.Module]] = None,
    ) -> None:
        self._device = self._resolve_device(device)
        self.use_refine = use_refine
        self.guidance_param = guidance_param
        self.fix_scale = fix_scale
        self._auto_download = auto_download
        self._checkpoint_dir = FsPath(checkpoint_dir) if checkpoint_dir is not None else DEFAULT_CACHE_DIR

        self._mask_model_factory = mask_model_factory or self._default_mask_model_factory
        self._feature_extractor_factory = (
            feature_extractor_factory or self._default_feature_extractor_factory
        )
        self._diffusion_factory = diffusion_factory or self._default_diffusion_factory
        self._refine_model_factory = refine_model_factory or create_model

        self._model: Optional[torch.nn.Module] = None
        self._diffusion = None
        self._refine_model: Optional[torch.nn.Module] = None
        self._features_model: Optional[torch.nn.Module] = None
        self._mask_model: Optional[torch.nn.Module] = None

        self._model_args: Optional[SimpleNamespace] = None
        self._refine_args: Optional[SimpleNamespace] = None
        self._num_paths: Optional[int] = None
        self._canvas_width: Optional[int] = None
        self._canvas_height: Optional[int] = None
        self._scaling_factor: Optional[float] = None

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def checkpoint_dir(self) -> FsPath:
        return self._checkpoint_dir

    def ensure_assets(self) -> None:
        """Ensure that the model archives are available locally."""

        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_archive(self.model_archive_path, MODEL_ARCHIVE_URL)
        if self.use_refine:
            self._ensure_archive(self.refine_archive_path, REFINE_ARCHIVE_URL)

    def generate(
        self,
        image: Union[ImageInput, Sequence[ImageInput]],
        size: Union[int, Tuple[int, int]] = 512,
    ) -> Union[Sketch, List[Sketch]]:
        """Generate sketches for the provided image(s)."""

        if isinstance(image, Sequence) and not isinstance(image, (str, bytes, np.ndarray, Image.Image)):
            images = list(image)
            if not images:
                return []
        else:
            images = [image]

        results = self.generate_batch(images)
        return results if len(images) > 1 else results[0]

    def generate_batch(self, images: Sequence[ImageInput]) -> List[Sketch]:
        if not images:
            return []

        pil_images = [self._to_pil_image(img) for img in images]
        self._initialise_pipeline()

        image_features: List[torch.Tensor] = []
        with torch.no_grad():
            for pil_image in pil_images:
                mask = sketch_utils.get_mask(pil_image, self.device, self._mask_model)
                masked_image = sketch_utils.create_masked_image(pil_image, mask)
                if self.fix_scale:
                    masked_image = sketch_utils.fix_image_scale(masked_image)
                features = self._features_model(masked_image)
                image_features.append(features.to(self.device))

        image_features_tensor = torch.stack(image_features).to(self.device).float()
        final_batch_size = image_features_tensor.shape[0]

        scale: Optional[torch.Tensor] = None
        if self.guidance_param != 1:
            scale = torch.ones(final_batch_size, device=self.device) * self.guidance_param

        sample_shape = (
            final_batch_size,
            self._num_paths,
            self._model.ncpoints,
            self._model.nfeats,
        )

        with torch.no_grad():
            sample = self._diffusion.p_sample_loop(
                self._model,
                sample_shape,
                noise=None,
                clip_denoised=False,
                image_features=image_features_tensor,
                scale=scale,
                progress=False,
                skip_timesteps=0,
                init_image=None,
                dump_steps=None,
                const_noise=False,
            )

            diffusion_control_points = sample.clone()

            if self.use_refine and self._refine_model is not None:
                t = torch.zeros(final_batch_size, dtype=torch.long, device=self.device)
                refined = self._refine_model(
                    x=diffusion_control_points,
                    timesteps=t,
                    image_features=image_features_tensor,
                )
                control_points = refined
            else:
                control_points = diffusion_control_points

            control_points = sketch_utils.denormalize_points(
                control_points, self._scaling_factor, self._canvas_width
            )

        control_points_cpu = control_points.detach().cpu()
        diffusion_cpu = diffusion_control_points.detach().cpu()

        _, svg_content_list = sketch_utils.rander_image_from_points(
            control_points_cpu, self._canvas_width, self._canvas_height, return_svg_content=True
        )

        pil_images = [sketch_utils.svg_to_pil(svg).convert("RGB") for svg in svg_content_list]

        sketches: List[Sketch] = []
        for ctrl, diff_ctrl, svg, pil_img in zip(control_points_cpu, diffusion_cpu, svg_content_list, pil_images):
            sketch = self._control_points_to_sketch(ctrl)
            setattr(
                sketch,
                SWIFTSKETCH_METADATA_ATTR,
                SwiftSketchArtifacts(
                    svg=svg,
                    image=pil_img,
                    control_points=ctrl,
                    diffusion_control_points=diff_ctrl,
                    compute_device=self.device,
                ),
            )
            sketches.append(sketch)

        return sketches

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _control_points_to_sketch(control_points: torch.Tensor) -> Sketch:
        if control_points.ndim != 3 or control_points.shape[-1] != 2:
            raise ValueError(
                "SwiftSketch expected control points with shape (num_paths, 4, 2)."
            )

        paths: List[SketchPath] = []
        for path_points in control_points:
            points = path_points.detach().cpu().numpy()
            if points.shape[0] != 4:
                raise ValueError(
                    "SwiftSketch currently only supports cubic Bézier curves with four control points."
                )

            start = Vertex(float(points[0][0]), float(points[0][1]))
            ctrl1 = Point(float(points[1][0]), float(points[1][1]))
            ctrl2 = Point(float(points[2][0]), float(points[2][1]))
            end = Vertex(float(points[3][0]), float(points[3][1]))

            curve = Curve(p_start=start, p_end=end, p_crtl1=ctrl1, p_crtl2=ctrl2)
            path = SketchPath(curves=[curve])
            paths.append(path)

        return Sketch(paths=paths)

    @property
    def model_archive_path(self) -> FsPath:
        return self._checkpoint_dir / MODEL_ARCHIVE_NAME

    @property
    def refine_archive_path(self) -> FsPath:
        return self._checkpoint_dir / REFINE_ARCHIVE_NAME

    @staticmethod
    def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
        """Determine which device SwiftSketch should run on.

        By default, a CUDA-capable GPU is required. Users can override the
        selection via the ``SKETCHKIT_SWIFTSKETCH_DEVICE`` environment
        variable or by explicitly passing a ``device`` argument. When neither
        is supplied and CUDA is unavailable, the model refuses to run unless
        the ``SKETCHKIT_ALLOW_CPU_FALLBACK`` environment variable is set to
        ``"1"``. This makes it possible to exercise the pipeline on CPU for
        testing while keeping the production default focused on GPU runtimes.
        """

        allow_cpu_env = os.getenv("SKETCHKIT_ALLOW_CPU_FALLBACK", "0")
        env_device = os.getenv("SKETCHKIT_SWIFTSKETCH_DEVICE")

        def _validate(requested: torch.device) -> torch.device:
            if requested.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError(
                    "Requested CUDA device but no CUDA runtime is available."
                )
            return requested

        if device is not None:
            return _validate(torch.device(device))

        if env_device:
            return _validate(torch.device(env_device))

        if torch.cuda.is_available():
            return torch.device("cuda")

        if allow_cpu_env == "1":
            return torch.device("cpu")

        raise RuntimeError(
            "SwiftSketchModel requires a CUDA-capable GPU. Set the"
            " environment variable SKETCHKIT_ALLOW_CPU_FALLBACK=1 to"
            " override this requirement for CPU-only environments."
        )

    def _ensure_archive(self, archive_path: FsPath, url: str) -> None:
        if archive_path.exists():
            return
        if not self._auto_download:
            raise FileNotFoundError(
                f"Required checkpoint archive missing: {archive_path}. Auto-download is disabled."
            )
        gdown.download(url, str(archive_path), quiet=False, fuzzy=True)

    def _initialise_pipeline(self) -> None:
        if self._model is not None:
            return

        self.ensure_assets()

        self._model_args = self._load_args(self.model_archive_path, MODEL_ARCHIVE_FOLDER)
        self._num_paths = int(self._model_args.num_paths)
        self._canvas_width = int(self._model_args.canvas_width)
        self._canvas_height = int(self._model_args.canvas_height)
        self._scaling_factor = float(self._model_args.scaling_factor)

        seed = getattr(self._model_args, "seed", 0)
        fixseed(seed)

        dist_device = self._device_index(self.device)
        dist_util.setup_dist(dist_device)

        # augment runtime specific attributes expected by downstream helpers
        self._model_args.device = str(self.device)
        self._model_args.guidance_param = self.guidance_param
        self._model_args.fix_scale = int(self.fix_scale)
        self._model_args.use_refine = int(self.use_refine)
        self._model_args.generate_batch_size = 1

        model, diffusion = self._diffusion_factory(self._model_args)
        state_dict = self._load_state_dict(self.model_archive_path, MODEL_ARCHIVE_FOLDER)
        load_model_wo_clip(model, state_dict)

        if self.guidance_param != 1:
            model = ClassifierFreeSampleModel(model)

        model.to(self.device)
        model.eval()
        self._model = model
        self._diffusion = diffusion

        feature_type = getattr(self._model_args, "image_features_type", "CLIPMiddle_layer4")
        features_model = self._feature_extractor_factory(self.device, feature_type)
        features_model.to(self.device)
        features_model.eval()
        self._features_model = features_model

        mask_model = self._mask_model_factory(self.device)
        mask_model.to(self.device)
        mask_model.eval()
        self._mask_model = mask_model

        if self.use_refine:
            self._refine_args = self._load_args(self.refine_archive_path, REFINE_ARCHIVE_FOLDER)
            self._refine_args.device = str(self.device)
            self._refine_args.guidance_param = self.guidance_param
            self._refine_args.fix_scale = int(self.fix_scale)
            self._refine_args.generate_batch_size = 1
            refine_model = self._refine_model_factory(self._refine_args)
            refine_state_dict = self._load_state_dict(self.refine_archive_path, REFINE_ARCHIVE_FOLDER)
            load_model_wo_clip(refine_model, refine_state_dict)
            refine_model.to(self.device)
            refine_model.eval()
            self._refine_model = refine_model

    @staticmethod
    def _device_index(device: torch.device) -> int:
        if device.type == "cuda":
            return device.index if device.index is not None else 0
        return -1

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
    def _load_args(archive_path: FsPath, folder: str) -> SimpleNamespace:
        with zipfile.ZipFile(archive_path) as zf:
            args_data = json.loads(zf.read(f"{folder}/args.json"))
        return SimpleNamespace(**args_data)

    @staticmethod
    def _find_model_member(zf: zipfile.ZipFile, folder: str) -> str:
        for name in zf.namelist():
            if name.startswith(f"{folder}/model") and name.endswith(".pt"):
                return name
        raise FileNotFoundError(f"Unable to locate model checkpoint inside archive: {folder}")

    def _load_state_dict(self, archive_path: FsPath, folder: str) -> dict:
        with zipfile.ZipFile(archive_path) as zf:
            member = self._find_model_member(zf, folder)
            with zf.open(member) as fh:
                buffer = io.BytesIO(fh.read())
        return torch.load(buffer, map_location="cpu")

    def _default_mask_model_factory(self, device: torch.device) -> torch.nn.Module:
        model = AutoModelForImageSegmentation.from_pretrained(
            "briaai/RMBG-1.4", trust_remote_code=True
        )
        return model.to(device)

    def _default_feature_extractor_factory(
        self, device: torch.device, image_features_type: str
    ) -> torch.nn.Module:
        layer_map = {
            "CLIPMiddle_layer3": 3,
            "CLIPMiddle_layer4": 4,
            "CLIPMiddle_layer5": 5,
        }
        layer = layer_map.get(image_features_type, 4)
        return image_feature_models.CLIPMidlleFeutures(device, layer)

    def _default_diffusion_factory(
        self, args: SimpleNamespace
    ) -> Tuple[torch.nn.Module, object]:
        return create_model_and_diffusion(args)
