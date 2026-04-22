"""
MangaNinja Model for Reference-based Line Art Colorization (SD 1.5 Version)

Paper: "MangaNinja: Line Art Colorization with Precise Reference Following" (CVPR 2025)
Repository: https://github.com/ali-vilab/MangaNinjia

Note: MangaNinja is built on SD 1.5, not SDXL. It uses a reference image
(instead of text prompts) to guide colorization, achieving remarkable
consistency with the reference.
"""

import os
import torch
import numpy as np
from PIL import Image
from typing import Tuple, Union, Any, Optional

from diffusers import (
    ControlNetModel,
    DDIMScheduler,
    AutoencoderKL,
)
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from sketchkit.utils.file import CACHE_DIR
from .manganinja.pipeline import MangaNinjiaPipeline
from .manganinja.models.unet_2d_condition import UNet2DConditionModel
from .manganinja.models.refunet_2d_condition import RefUNet2DConditionModel
from .manganinja.point_network import PointNet
from .manganinja.annotator.lineart import BatchLineartDetector

MANGANINJA_WEIGHTS_DIR = os.path.join(CACHE_DIR, "weights", "colorization", "manga_ninja")

MANGANINJA_HF_REPO = "Johanan0528/MangaNinjia"

MANGANINJA_WEIGHT_FILES = [
    "denoising_unet.pth",
    "reference_unet.pth",
    "controlnet.pth",
    "point_net.pth",
]

MANGANINJA_ANNOTATOR_FILES = {
    "sk_model.pth": "lllyasviel/Annotators",
}


def _get_weights_dir() -> str:
    os.makedirs(MANGANINJA_WEIGHTS_DIR, exist_ok=True)
    return MANGANINJA_WEIGHTS_DIR


def _check_weights_available() -> bool:
    for fname in MANGANINJA_WEIGHT_FILES:
        fpath = os.path.join(MANGANINJA_WEIGHTS_DIR, fname)
        if not os.path.exists(fpath) or os.path.getsize(fpath) < 1024:
            return False
    annotator_dir = os.path.join(MANGANINJA_WEIGHTS_DIR, "annotator")
    for fname in MANGANINJA_ANNOTATOR_FILES:
        fpath = os.path.join(annotator_dir, fname)
        if not os.path.exists(fpath) or os.path.getsize(fpath) < 1024:
            return False
    return True


_PROXY_ENV_VARS = ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"]


def _with_clean_env(func, remove_proxy=False, remove_endpoint=False):
    saved = {}
    keys = list(_PROXY_ENV_VARS) if remove_proxy else []
    if remove_endpoint:
        keys.append("HF_ENDPOINT")
    for k in keys:
        saved[k] = os.environ.pop(k, None)
    if remove_endpoint:
        os.environ["HF_ENDPOINT"] = "https://huggingface.co"
    try:
        return func()
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
            elif k in os.environ:
                del os.environ[k]
        if remove_endpoint and "HF_ENDPOINT" in os.environ and os.environ["HF_ENDPOINT"] == "https://huggingface.co":
            del os.environ["HF_ENDPOINT"]


def _download_via_subprocess(repo_id, filename, local_dir, endpoint=None):
    import subprocess
    import sys
    code = (
        "import os, sys;\n"
        "for k in ['http_proxy','https_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','all_proxy','HF_ENDPOINT']:\n"
        "    os.environ.pop(k, None);\n"
    )
    if endpoint:
        code += f"os.environ['HF_ENDPOINT'] = {endpoint!r};\n"
    code += (
        "from huggingface_hub import hf_hub_download;\n"
        f"hf_hub_download(repo_id={repo_id!r}, filename={filename!r}, local_dir={local_dir!r}"
    )
    if endpoint:
        code += f", endpoint={endpoint!r}"
    code += ");\nprint('OK');"
    env = os.environ.copy()
    for k in _PROXY_ENV_VARS:
        env.pop(k, None)
    env.pop("HF_ENDPOINT", None)
    if endpoint:
        env["HF_ENDPOINT"] = endpoint
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Download failed: {result.stderr[-500:]}")


def _download_file_with_fallback(hf_hub_download, repo_id, filename, local_dir):
    try:
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
        return
    except Exception:
        pass

    try:
        print(f"  Retrying {filename} without proxy (subprocess)...")
        _download_via_subprocess(repo_id, filename, local_dir)
        return
    except Exception:
        pass

    try:
        print(f"  Retrying {filename} directly from huggingface.co (subprocess)...")
        _download_via_subprocess(repo_id, filename, local_dir, endpoint="https://huggingface.co")
        return
    except Exception:
        pass

    try:
        print(f"  Retrying {filename} without proxy (in-process)...")
        _with_clean_env(lambda: hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir), remove_proxy=True)
        return
    except Exception:
        pass

    print(f"  Retrying {filename} directly from huggingface.co (in-process)...")
    _with_clean_env(
        lambda: hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir, endpoint="https://huggingface.co"),
        remove_proxy=True, remove_endpoint=True,
    )


def _download_weights():
    from huggingface_hub import hf_hub_download

    weights_dir = _get_weights_dir()
    print(f"Downloading MangaNinja weights to {weights_dir} ...")
    for fname in MANGANINJA_WEIGHT_FILES:
        fpath = os.path.join(weights_dir, fname)
        if os.path.exists(fpath) and os.path.getsize(fpath) >= 1024:
            print(f"  [skip] {fname} already exists")
            continue
        print(f"  Downloading {fname} ...")
        _download_file_with_fallback(hf_hub_download, MANGANINJA_HF_REPO, fname, weights_dir)

    annotator_dir = os.path.join(weights_dir, "annotator")
    os.makedirs(annotator_dir, exist_ok=True)
    for fname, repo_id in MANGANINJA_ANNOTATOR_FILES.items():
        fpath = os.path.join(annotator_dir, fname)
        if os.path.exists(fpath) and os.path.getsize(fpath) >= 1024:
            print(f"  [skip] annotator/{fname} already exists")
            continue
        print(f"  Downloading annotator/{fname} ...")
        _download_file_with_fallback(hf_hub_download, repo_id, fname, annotator_dir)

    print("✅ MangaNinja weights downloaded successfully")


_NETWORK_ERROR_NAMES = {
    "LocalEntryNotFoundError",
    "ProxyError",
    "ConnectionError",
}

def _is_network_error(exc: BaseException) -> bool:
    return isinstance(exc, (ConnectionError, TimeoutError, OSError)) or \
        type(exc).__name__ in _NETWORK_ERROR_NAMES


_HF_MODELS_TO_CACHE = [
    ("runwayml/stable-diffusion-v1-5", ["scheduler", "vae", "unet", "tokenizer", "text_encoder"]),
    ("openai/clip-vit-large-patch14", []),
    ("lllyasviel/control_v11p_sd15_lineart", []),
]


def _prefetch_hf_models():
    import subprocess
    import sys
    hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
    for repo_id, _ in _HF_MODELS_TO_CACHE:
        repo_dir = repo_id.replace("/", "--")
        no_exist_dir = os.path.join(hf_cache, f"models--{repo_dir}", ".no_exist")
        if os.path.exists(no_exist_dir):
            import shutil
            shutil.rmtree(no_exist_dir, ignore_errors=True)
    code = (
        "import os;\n"
        "for k in ['http_proxy','https_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','all_proxy','HF_ENDPOINT']:\n"
        "    os.environ.pop(k, None);\n"
        "os.environ['HF_ENDPOINT'] = 'https://huggingface.co';\n"
        "from huggingface_hub import snapshot_download;\n"
    )
    for repo_id, _ in _HF_MODELS_TO_CACHE:
        code += f"snapshot_download(repo_id={repo_id!r}, endpoint='https://huggingface.co');\n"
    code += "print('OK');\n"
    env = os.environ.copy()
    for k in _PROXY_ENV_VARS:
        env.pop(k, None)
    env.pop("HF_ENDPOINT", None)
    env["HF_ENDPOINT"] = "https://huggingface.co"
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"HF model prefetch failed: {result.stderr[-500:]}")


class MangaNinjaModel:

    def __init__(
        self,
        device: str = "cuda",
        pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5",
        image_encoder_path: str = "openai/clip-vit-large-patch14",
        controlnet_model_name_or_path: str = "lllyasviel/control_v11p_sd15_lineart",
        torch_dtype: torch.dtype = torch.float16,
        auto_download: bool = True,
        **kwargs: Any,
    ):
        print("Loading MangaNinja model...")

        if not _check_weights_available():
            if auto_download:
                _download_weights()
            else:
                raise FileNotFoundError(
                    f"MangaNinja weights not found in {MANGANINJA_WEIGHTS_DIR}. "
                    f"Set auto_download=True to download them automatically, or "
                    f"manually download from https://huggingface.co/{MANGANINJA_HF_REPO}"
                )

        if auto_download:
            self._ensure_hf_models_cached(pretrained_model_name_or_path, image_encoder_path, controlnet_model_name_or_path)

        weights_dir = _get_weights_dir()
        manga_denoising_unet_path = os.path.join(weights_dir, "denoising_unet.pth")
        manga_reference_unet_path = os.path.join(weights_dir, "reference_unet.pth")
        manga_controlnet_path = os.path.join(weights_dir, "controlnet.pth")
        manga_point_net_path = os.path.join(weights_dir, "point_net.pth")

        annotator_ckpts_path = os.path.join(weights_dir, "annotator")
        os.makedirs(annotator_ckpts_path, exist_ok=True)

        self._load_pipeline(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            image_encoder_path=image_encoder_path,
            controlnet_model_name_or_path=controlnet_model_name_or_path,
            device=device,
            annotator_ckpts_path=annotator_ckpts_path,
            manga_denoising_unet_path=manga_denoising_unet_path,
            manga_reference_unet_path=manga_reference_unet_path,
            manga_controlnet_path=manga_controlnet_path,
            manga_point_net_path=manga_point_net_path,
        )

        self.device = device
        print("✅ MangaNinja model loaded successfully")

    def _ensure_hf_models_cached(self, pretrained_model_name_or_path, image_encoder_path, controlnet_model_name_or_path):
        repos = [
            pretrained_model_name_or_path,
            image_encoder_path,
            controlnet_model_name_or_path,
        ]
        hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
        all_cached = True
        for repo_id in repos:
            repo_dir = repo_id.replace("/", "--")
            ref_path = os.path.join(hf_cache, f"models--{repo_dir}", "refs", "main")
            if not os.path.exists(ref_path):
                all_cached = False
                break
        if all_cached:
            return

        print("  Pre-caching HuggingFace models via subprocess...")
        import subprocess
        import sys
        hf_cache_dir = os.path.expanduser("~/.cache/huggingface")
        for repo_id in repos:
            repo_dir = repo_id.replace("/", "--")
            no_exist_dir = os.path.join(hf_cache, f"models--{repo_dir}", ".no_exist")
            if os.path.exists(no_exist_dir):
                import shutil
                shutil.rmtree(no_exist_dir, ignore_errors=True)
        code = (
            "import os;\n"
            "for k in ['http_proxy','https_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','all_proxy','HF_ENDPOINT']:\n"
            "    os.environ.pop(k, None);\n"
            "os.environ['HF_ENDPOINT'] = 'https://huggingface.co';\n"
            "from huggingface_hub import snapshot_download;\n"
        )
        for repo_id in repos:
            code += f"snapshot_download(repo_id={repo_id!r}, endpoint='https://huggingface.co');\n"
        code += "print('OK');\n"
        env = os.environ.copy()
        for k in _PROXY_ENV_VARS:
            env.pop(k, None)
        env.pop("HF_ENDPOINT", None)
        env["HF_ENDPOINT"] = "https://huggingface.co"
        result = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  Warning: HF model prefetch failed: {result.stderr[-200:]}")
            print("  Will try loading from cache or network directly...")

    def _load_pipeline(self, pretrained_model_name_or_path, image_encoder_path,
                       controlnet_model_name_or_path, device, annotator_ckpts_path,
                       manga_denoising_unet_path, manga_reference_unet_path,
                       manga_controlnet_path, manga_point_net_path):
        preprocessor = BatchLineartDetector(annotator_ckpts_path)
        preprocessor.to(device, dtype=torch.float32)

        noise_scheduler = DDIMScheduler.from_pretrained(
            pretrained_model_name_or_path, subfolder="scheduler",
            
        )
        vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path, subfolder="vae",
            
        )

        denoising_unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="unet",
            in_channels=4,
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True,
            
        )

        reference_unet = RefUNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="unet",
            in_channels=4,
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True,
            
        )

        refnet_tokenizer = CLIPTokenizer.from_pretrained(image_encoder_path)
        refnet_text_encoder = CLIPTextModel.from_pretrained(image_encoder_path)
        refnet_image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path)

        controlnet = ControlNetModel.from_pretrained(
            controlnet_model_name_or_path,
            in_channels=4,
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True,
            
        )
        controlnet_tokenizer = CLIPTokenizer.from_pretrained(image_encoder_path)
        controlnet_text_encoder = CLIPTextModel.from_pretrained(image_encoder_path)
        controlnet_image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path)

        point_net = PointNet()

        controlnet.load_state_dict(
            torch.load(manga_controlnet_path, map_location="cpu"),
            strict=False,
        )
        point_net.load_state_dict(
            torch.load(manga_point_net_path, map_location="cpu"),
            strict=False,
        )
        reference_unet.load_state_dict(
            torch.load(manga_reference_unet_path, map_location="cpu"),
            strict=False,
        )
        denoising_unet.load_state_dict(
            torch.load(manga_denoising_unet_path, map_location="cpu"),
            strict=False,
        )

        self.pipeline = MangaNinjiaPipeline(
            reference_unet=reference_unet,
            controlnet=controlnet,
            denoising_unet=denoising_unet,
            vae=vae,
            refnet_tokenizer=refnet_tokenizer,
            refnet_text_encoder=refnet_text_encoder,
            refnet_image_encoder=refnet_image_encoder,
            controlnet_tokenizer=controlnet_tokenizer,
            controlnet_text_encoder=controlnet_text_encoder,
            controlnet_image_encoder=controlnet_image_encoder,
            scheduler=noise_scheduler,
            point_net=point_net,
        )
        self.pipeline = self.pipeline.to(torch.device(device))
        self.preprocessor = preprocessor

    def generate(
        self,
        control_image: Image.Image,
        reference_image: Image.Image,
        prompt: str = "",
        size: Optional[Union[int, Tuple[int, int]]] = None,
        negative_prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        guidance_scale_ref: float = 9.0,
        guidance_scale_point: float = 15.0,
        is_lineart: bool = False,
        seed: Optional[int] = None,
        point_ref: Optional[torch.Tensor] = None,
        point_main: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Image.Image:
        """Generates a colorized image from line art using a reference image.

        Args:
            control_image (Image.Image): The input line art image to colorize.
            reference_image (Image.Image): The reference image providing color guidance.
            prompt (str): Not used by MangaNinja (kept for interface compatibility).
            size (Optional[Union[int, Tuple[int, int]]]): Target resolution.
                MangaNinja operates at 512x512 internally. Defaults to 512.
            negative_prompt (str): Not used by MangaNinja (kept for interface compatibility).
            num_inference_steps (int): Number of denoising steps. Defaults to 50.
            guidance_scale (float): Not used directly; guidance_scale_ref and
                guidance_scale_point are used instead.
            guidance_scale_ref (float): Guidance scale for reference image influence.
                Higher values make the model follow the reference more closely. Defaults to 9.0.
            guidance_scale_point (float): Guidance scale for point control influence.
                Higher values make the model follow point guidance more closely. Defaults to 15.0.
            is_lineart (bool): If True, the input is already a line art image and
                no additional line art extraction is needed. Defaults to False.
            seed (Optional[int]): Random seed for deterministic generation.
            point_ref (Optional[torch.Tensor]): Point map on the reference image
                for point-guided colorization. Shape: (1, 1, H, W).
            point_main (Optional[torch.Tensor]): Point map on the line art image
                for point-guided colorization. Shape: (1, 1, H, W).
            **kwargs (Any): Additional arguments.

        Returns:
            Image.Image: The colorized image.

        Raises:
            RuntimeError: If image generation fails during pipeline execution.
        """
        processing_res = 512
        if size is not None:
            if isinstance(size, int):
                processing_res = size
            else:
                processing_res = min(size[0], size[1])

        ref_image = reference_image.resize((processing_res, processing_res))
        target_image = control_image.resize((processing_res, processing_res))

        if point_ref is None:
            point_ref = torch.zeros(1, 1, processing_res, processing_res, dtype=torch.uint8)
        if point_main is None:
            point_main = torch.zeros(1, 1, processing_res, processing_res, dtype=torch.uint8)

        generator = None
        if seed is not None:
            device = self.device if self.device != "cuda" or torch.cuda.is_available() else "cpu"
            generator = torch.cuda.manual_seed(seed) if device == "cuda" else torch.Generator(device=device).manual_seed(seed)

        try:
            pipe_out = self.pipeline(
                is_lineart=is_lineart,
                ref1=ref_image,
                raw2=target_image,
                edit2=target_image,
                denosing_steps=num_inference_steps,
                processing_res=processing_res,
                match_input_res=True,
                batch_size=1,
                show_progress_bar=True,
                guidance_scale_ref=guidance_scale_ref,
                guidance_scale_point=guidance_scale_point,
                preprocessor=self.preprocessor,
                generator=generator,
                point_ref=point_ref,
                point_main=point_main,
            )
            return pipe_out.img_pil
        except Exception as e:
            raise RuntimeError(f"MangaNinja generation failed: {e}") from e
