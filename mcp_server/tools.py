"""
SketchKit MCP Server
Wrapper for SketchKit functionalities (Vectorization, Stylization, Cleanup, Image2Sketch, Sketch2Model).
"""

import os
import sys
import logging
from typing import List
from PIL import Image
import numpy as np
from pathlib import Path
import shutil
from typing import List, Optional, Dict, Any, Union

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    # User might need to install 'mcp'
    print("Warning: 'mcp' package not found. Please install it with `pip install mcp`.")

    # Dummy class to prevent import error during development if mcp is missing
    class FastMCP:
        def __init__(self, name):
            self.name = name

        def tool(self):
            return lambda f: f

        def run(self):
            print("MCP Server running (dummy mode)...")


# Ensure sketchkit is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import SketchKit modules
try:
    from sketchkit.core.sketch import Sketch
    from sketchkit.vectorization import Vectorizer
    from sketchkit.stylization import Stylizer
    from sketchkit.cleanup import Cleaner
    from sketchkit.image2sketch import image2sketch
    from sketchkit.sketch2model import Modeler
    from sketchkit.renderer.npr_renderer import NPRRenderer
    from sketchkit.sketch2model import sketch_to_3d
    from sketchkit.animation import Animator
    from sketchkit.ordering import Orderer
    from sketchkit.sketch2image import ImageGenerator
    from sketchkit.colorization import Colorizer
    from sketchkit.renderer.cairo_renderer import CairoRenderer
    from sketchkit.utils.file import save_seq_gif
except ImportError as e:
    print(f"Error importing SketchKit modules: {e}")
    sys.exit(1)

# Initialize MCP Server
mcp = FastMCP("sketchkit")

# Logging setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Helper Functions ---


def _load_image_np(path: str) -> np.ndarray:
    # path = os.path.join("temp_storage", path)
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        return np.array(Image.open(path).convert("L"))
    except Exception as e:
        raise ValueError(f"Failed to load image as numpy array at {path}: {e}")


def _load_image_pil(path: str) -> Image.Image:
    path = os.path.join("temp_storage", path)
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        return Image.open(path)
    except Exception as e:
        raise ValueError(f"Failed to load image at {path}: {e}")


# --- Helper Functions (Core Logic) ---


def _run_vectorize(image_paths: List[str], method: str, output_dir: str) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    results = []
    try:
        vectorizer = Vectorizer(method=method)
    except Exception as e:
        return [f"Failed to initialize Vectorizer: {e}"]

    for path in image_paths:
        try:
            logging.info(f"Vectorizing {path}...")
            img_np = _load_image_np(path)
            sketch = vectorizer.run(img_np)

            base_name = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(output_dir, f"{base_name}.json")
            sketch.to_json(out_path)
            results.append(out_path)
        except Exception as e:
            msg = f"Error processing {path}: {str(e)}"
            logging.error(msg)
            results.append(msg)
    return results


def _run_stylize(
    input_paths: List[str], method: str, output_dir: str, **kwargs
) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    results = []
    try:
        stylizer = Stylizer(method=method, **kwargs)
    except Exception as e:
        return [f"Failed to initialize Stylizer: {e}"]

    for path in input_paths:
        try:
            logging.info(f"Stylizing {path}...")
            if not os.path.exists(path):
                results.append(f"File not found: {path}")
                continue

            base_name = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(output_dir, f"{base_name}_{method}.png")

            if path.endswith(".json"):
                sketch = Sketch.from_json(path)
                stylizer.run(sketch, output_path=out_path, **kwargs)
            else:
                stylizer.run_from_png(path, output_path=out_path, **kwargs)
            results.append(out_path)
        except Exception as e:
            msg = f"Error processing {path}: {str(e)}"
            logging.error(msg)
            results.append(msg)
    return results


def _run_cleanup(image_paths: List[str], method: str, output_dir: str) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    results = []
    try:
        cleaner = Cleaner(method=method)
    except Exception as e:
        return [f"Failed to initialize Cleaner: {e}"]

    for path in image_paths:
        try:
            logging.info(f"Cleaning {path}...")
            img_np = _load_image_np(path)
            output_img = cleaner.run(img_np)

            base_name = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(output_dir, f"{base_name}_clean.png")
            Image.fromarray(output_img).save(out_path)
            results.append(out_path)
        except Exception as e:
            msg = f"Error processing {path}: {str(e)}"
            logging.error(msg)
            results.append(msg)
    return results


def _run_image2sketch(
    image_paths: List[str], method: str, output_dir: str, size: int
) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    results = []
    for path in image_paths:
        try:
            logging.info(f"Converting photo {path}...")
            path_for_loader = os.path.basename(path)
            pil_image = _load_image_pil(path_for_loader).convert("RGB")
            res = image2sketch(pil_image, method=method, size=size)
            base_name = os.path.splitext(os.path.basename(path))[0]

            if isinstance(res, Sketch):
                out_path = os.path.join(output_dir, f"{base_name}_{method}.json")
                res.to_json(out_path)
            elif isinstance(res, Image.Image):
                out_path = os.path.join(output_dir, f"{base_name}_{method}.png")
                res.save(out_path)
            else:
                msg = f"Unknown result type type({type(res)}) for {path}"
                logging.warning(msg)
                results.append(msg)
                continue
            results.append(out_path)
        except Exception as e:
            msg = f"Error processing {path}: {str(e)}"
            logging.error(msg)
            results.append(msg)
    return results


def _run_sketch2model(
    input_paths: List[str], method: str, output_dir: str, **kwargs
) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for path in input_paths:
        try:
            logging.info(f"Generating 3D model for {path}...")
            if not os.path.exists(path):
                results.append(f"File not found: {path}")
                continue

            base_name = os.path.splitext(os.path.basename(path))[0]
            out_ext = ".glb"
            out_path = os.path.join(output_dir, f"{base_name}{out_ext}")

            input_data = path
            if path.endswith(".json"):
                input_data = Sketch.from_json(path)

            final_path = sketch_to_3d(
                input_data,
                method="doubao-seed3d",
                output_path=out_path,
                file_format="glb",
            )

            results.append(str(Path(final_path).with_suffix(".glb")))
        except Exception as e:
            msg = f"Error processing {path}: {str(e)}"
            logging.error(msg)
            results.append(msg)
    return results


def _run_render3d(obj_paths: List[str], output_dir: str, **kwargs) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    results = []
    from sketchkit.core.camera import Camera

    camera = Camera()
    camera.set_look_at([0, 0, 0])
    camera.set_xyz((2.5, 0.0, 0.0))

    for path in obj_paths:
        try:
            logging.info(f"Rendering 3D obj {path}...")
            if not os.path.exists(path):
                results.append(f"File not found: {path}")
                continue

            if not path.endswith(".obj"):
                logging.warning(f"Skipping non-obj file for rendering: {path}")
                continue

            renderer = NPRRenderer(obj_path=path)

            base_name = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(output_dir, f"{base_name}.png")

            imgs = renderer._render(cameras=[camera])
            imgs[0].save(out_path)
            results.append(out_path)
        except Exception as e:
            msg = f"Error processing {path}: {str(e)}"
            logging.error(msg)
            results.append(msg)
    return results


def _run_sketch2image(
    input_paths: List[str], method: str, output_dir: str, prompt: str, **kwargs
) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    results: List[str] = []
    generator = ImageGenerator(method=method)

    for path in input_paths:
        try:

            if path.lower().endswith(".json"):
                input = Sketch.from_json(path)
            else:
                input = Image.open(path).convert("RGB")
                # img_np = _load_image_np(path)
                # sketch_obj = Vectorizer(method="LineDrawer").run(img_np)
                # path = os.path.join(output_dir, os.path.basename(path) + ".json")
                # sketch_obj.to_json(path)

            # logging.info(f"Sketch2Image {path} (method={method})")
            # input_obj = Sketch.from_json(path)

            pil_img = generator.run(input, prompt=prompt, **kwargs)
            if not isinstance(pil_img, Image.Image):
                raise TypeError(
                    f"Unexpected output type from ImageGenerator.run: {type(pil_img)}"
                )

            base_name = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(output_dir, f"{base_name}_s2i_{method}.png")
            pil_img.convert("RGB").save(out_path)
            results.append(out_path)
        except Exception as e:
            msg = f"Error processing {path}: {e}"
            logging.error(msg)
            results.append(msg)
    return results


def _run_sketch2image_sdxl(
    input_paths: List[str], output_dir: str, prompt: str, **kwargs
):
    from diffusers import (
        ControlNetModel,
        StableDiffusionXLControlNetPipeline,
        AutoencoderKL,
    )
    from diffusers import EulerAncestralDiscreteScheduler
    import torch
    import numpy as np

    controlnet_conditioning_scale = 1.0

    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"

    eulera_scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
        "../hf_models/models--stabilityai--stable-diffusion-xl-base-1.0",
        subfolder="scheduler",
    )

    controlnet = ControlNetModel.from_pretrained(
        "xinsir/controlnet-scribble-sdxl-1.0", torch_dtype=torch.float16
    )

    # when test with other base model, you need to change the vae also.
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
    )

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        "../hf_models/models--stabilityai--stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        vae=vae,
        safety_checker=None,
        torch_dtype=torch.float16,
        scheduler=eulera_scheduler,
    ).to("cuda")
    results = []
    for input_path in input_paths:
        controlnet_img = Image.open(input_path)

        width, height = controlnet_img.size
        ratio = np.sqrt(1024.0 * 1024.0 / (width * height))
        new_width, new_height = int(width * ratio), int(height * ratio)
        controlnet_img = controlnet_img.resize((new_width, new_height))

        images = pipe(
            prompt,
            negative_prompt=negative_prompt,
            image=controlnet_img,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            width=new_width,
            height=new_height,
            num_inference_steps=30,
        ).images

        base_name = os.path.splitext(os.path.basename(input_path))[0]
        out_path = os.path.join(output_dir, f"{base_name}_s2i_sdxl.png")

        images[0].save(out_path)
        results.append(out_path)

    return results


def _run_colorization(
    input_paths: List[str], method: str, output_dir: str, prompt: str, **kwargs
) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    results: List[str] = []
    colorizer = Colorizer(method=method)

    for path in input_paths:
        try:
            logging.info(f"Colorization {path} (method={method})")
            input_obj = Sketch.from_json(path)

            pil_img = colorizer.run(input_obj, prompt=prompt, **kwargs)
            if not isinstance(pil_img, Image.Image):
                raise TypeError(
                    f"Unexpected output type from Colorizer.run: {type(pil_img)}"
                )

            base_name = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(output_dir, f"{base_name}_color_{method}.png")
            pil_img.convert("RGB").save(out_path)
            results.append(out_path)
        except Exception as e:
            msg = f"Error processing {path}: {e}"
            logging.error(msg)
            results.append(msg)
    return results


def _run_ordering_gif(
    method: str,
    output_dir: str,
    json_path: Optional[str] = None,
    img_path: Optional[str] = None,
    video_path: Optional[str] = None,
    sample_rate: int = 10,
    fps: int = 12,
) -> str:
    """Create a progressive ordering GIF.

    - method='Fu' requires json_path (Sketch JSON).
    - method='LineDrawer' requires img_path (raster) and video_path (reference video).
    """
    os.makedirs(output_dir, exist_ok=True)

    if method not in {"Fu", "LineDrawer"}:
        raise ValueError("method must be 'Fu' or 'LineDrawer'")
    if sample_rate <= 0 or sample_rate > 200:
        raise ValueError("sample_rate must be in (0, 200]")
    if fps <= 0 or fps > 60:
        raise ValueError("fps must be in (0, 60]")

    if method == "Fu":
        if not json_path:
            raise ValueError("json_path is required when method='Fu'")
        sketch = Sketch.from_json(json_path)
        orderer = Orderer.create(method="Fu", device="cpu")
        ordered_sketch = orderer.run(sketch)
    else:
        if not img_path or not video_path:
            raise ValueError(
                "img_path and video_path are required when method='LineDrawer'"
            )
        img_np = np.array(Image.open(img_path).convert("L"), dtype=np.uint8)
        orderer = Orderer.create(method="LineDrawer")
        ordered_sketch = orderer.run(img_np, video_path)

    if not isinstance(ordered_sketch, Sketch):
        raise TypeError(
            f"Unexpected output type from Orderer.run: {type(ordered_sketch)}"
        )

    frames = ordered_sketch.get_progressive_frames(length_per_frame=sample_rate)
    renderer = CairoRenderer()
    raster_images = [renderer.render(frame) for frame in frames]

    out_path = os.path.join(output_dir, f"ordering_{method.lower()}.gif")
    save_seq_gif(raster_images, out_path, fps=fps)
    return out_path


def _run_animation(
    img0_path: str,
    img1_path: str,
    output_dir: str,
    inner_frames: int = 24,
) -> str:
    """Interpolate between two images (RIFE) and return a GIF path.

    Notes:
    - SketchKit's Animator(RIFE) writes a GIF to ./animation/dynamic.gif in the
      reference Flask app. Here we copy that file (if produced) into output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    if inner_frames < 0 or inner_frames > 120:
        raise ValueError("inner_frames must be in [0, 120]")

    animator = Animator(method="RIFE")

    img0_np = np.array(Image.open(img0_path).convert("RGB"), dtype=np.uint8)
    img1_np = np.array(Image.open(img1_path).convert("RGB"), dtype=np.uint8)

    _ = animator.run(img0_np, img1_np, inner_frames=inner_frames)

    # Try to grab the GIF produced by Animator (same as app(2).py behavior)
    produced = Path("./animation/dynamic.gif")
    out_path = Path(output_dir) / "animation_rife.gif"

    if produced.exists():
        shutil.copyfile(produced, out_path)
        return str(out_path)

    # Fallback: if Animator didn't write a GIF, we create one from returned frames (best-effort)
    try:
        frames = animator.run(img0_np, img1_np, inner_frames=inner_frames)
        pil_frames = [
            Image.fromarray(f.astype(np.uint8)).convert("RGB") for f in frames
        ]
        save_seq_gif(pil_frames, str(out_path), fps=12)
        return str(out_path)
    except Exception as e:
        raise RuntimeError(
            "Animator did not produce ./animation/dynamic.gif and fallback GIF creation failed"
        ) from e


# --- MCP Tools ---
@mcp.tool()
def vectorize_images(image_paths: List[str], method: str = "DeepVecSIG24") -> List[str]:
    """
    Vectorize a list of raster images (PNG/JPG) into vector sketches (JSON).

    Args:
        image_paths: List of file paths to input images.
        method: Vectorization method. Options: "LineDrawer", "DeepVecSIG24".
    """
    return _run_vectorize(
        image_paths, method, os.path.join(os.getcwd(), "output/vectorized")
    )


@mcp.tool()
def stylize(
    input_paths: List[str], method: str = "brushstrokesengine", **kwargs
) -> List[str]:
    """
    Stylize sketches (JSON) or images (PNG).

    Args:
        input_paths: List of file paths (either JSON sketches or PNG images).
        method: Stylization method ("brushstrokesengine").
        **kwargs: detailed parameters (e.g. model_name="style2", style_id="playdoh10").
    """
    return _run_stylize(
        input_paths, method, os.path.join(os.getcwd(), "output/stylized"), **kwargs
    )


@mcp.tool()
def cleanup_sketches(image_paths: List[str], method: str = "MasterSketch") -> List[str]:
    """
    Clean up rough raster sketches into clean vector sketches.

    Args:
        image_paths: List of paths to raster sketch images.
        method: Cleanup method ("MasterSketch").
    """
    return _run_cleanup(
        image_paths, method, os.path.join(os.getcwd(), "output/cleanup")
    )


@mcp.tool()
def photos_to_sketches(
    image_paths: List[str], method: str = "swiftsketch", size: int = 512
) -> List[str]:
    """
    Convert photos to sketches.

    Args:
        image_paths: List of paths to input photos.
        method: Method ("hed", "photosketch", "swiftsketch").
                "swiftsketch" returns a vector Sketch (JSON), others return raster (PNG).
        size: Size to resize input image (default 512).
    """
    return _run_image2sketch(
        image_paths, method, os.path.join(os.getcwd(), "output/image2sketch"), size
    )


@mcp.tool()
def sketches_to_3d(
    input_paths: List[str], method: str = "doubao-seed3d", **kwargs
) -> List[str]:
    """
    Convert sketches (raster or vector) to 3D models.

    Args:
        input_paths: List of paths to sketches (PNG, JPG, or JSON).
        method: Method ("teddy", "sens", "doubao-seed3d").
    """
    return _run_sketch2model(
        input_paths, method, os.path.join(os.getcwd(), "output/3d"), **kwargs
    )


@mcp.tool()
def render_3d_objs(obj_paths: List[str], **kwargs) -> List[str]:
    """
    Render 3D OBJ files to sketch images using NPR (Non-Photorealistic Rendering).

    Args:
        obj_paths: List of paths to .obj files.
        **kwargs: Render options (e.g. sug_thresh=0.005).
    """
    return _run_render3d(
        obj_paths, os.path.join(os.getcwd(), "output/render3d"), **kwargs
    )


@mcp.tool()
def sketch2image(
    input_paths: List[str],
    prompt: str,
    # method: str = "controlnet",
    # kwargs_json: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Generate images from a sketch (JSON) or line-art image (PNG/JPG).

    Args:
        input_paths: List of paths to .json sketches or .png/.jpg images.
        prompt: Text prompt.
    """
    # if not prompt or not prompt.strip():
    #     return ["prompt is required"]
    # method = (method or "controlnet").strip().lower()
    # if method not in {"controlnet", "t2i_adapter", "pix2pix"}:
    #     return [
    #         f"Unsupported method: {method}. Allowed: ['controlnet','t2i_adapter','pix2pix']"
    #     ]

    # kwargs_json = kwargs_json or {}
    # for k in [
    #     "controlnet_conditioning_scale",
    #     "adapter_conditioning_scale",
    #     "image_guidance_scale",
    # ]:
    #     if k in kwargs_json:
    #         try:
    #             kwargs_json[k] = float(kwargs_json[k])
    #         except Exception:
    #             pass

    # return _run_sketch2image(
    #     input_paths,
    #     method,
    #     os.path.join(os.getcwd(), "output/sketch2image"),
    #     prompt.strip(),
    #     **kwargs_json,
    # )

    return _run_sketch2image_sdxl(
        input_paths,
        os.path.join(os.getcwd(), "output/sketch2image"),
        prompt.strip()
    )

    # return _run_sketch2image_dashscope(
    #     input_paths,
    #     os.path.join(os.getcwd(), "output/sketch2image"),
    #     prompt.strip(),
    # )


@mcp.tool()
def colorization(
    input_paths: List[str],
    prompt: str,
    method: str = "controlnet_lineart",
    kwargs_json: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Colorize line-art / sketches.

    Args:
        input_paths: List of paths to .json sketches or .png/.jpg images.
        prompt: Text prompt.
        method: One of: controlnet_lineart, controlnet_scribble.
        kwargs_json: Optional dict forwarded to Colorizer.run.
    """
    if not prompt or not prompt.strip():
        return ["prompt is required"]
    method = (method or "controlnet_lineart").strip().lower()
    if method not in {"controlnet_lineart", "controlnet_scribble"}:
        return [
            f"Unsupported method: {method}. Allowed: ['controlnet_lineart','controlnet_scribble']"
        ]

    kwargs_json = kwargs_json or {}
    for k in [
        "guidance_scale",
        "strength",
        "controlnet_conditioning_scale",
        "adapter_conditioning_scale",
        "image_guidance_scale",
    ]:
        if k in kwargs_json:
            try:
                kwargs_json[k] = float(kwargs_json[k])
            except Exception:
                pass
    if "num_inference_steps" in kwargs_json:
        try:
            kwargs_json["num_inference_steps"] = int(kwargs_json["num_inference_steps"])
        except Exception:
            pass

    return _run_colorization(
        input_paths,
        method,
        os.path.join(os.getcwd(), "output/colorization"),
        prompt.strip(),
        **kwargs_json,
    )


@mcp.tool()
def ordering(
    method: str = "Fu",
    json_path: Optional[str] = None,
    img_path: Optional[str] = None,
    video_path: Optional[str] = None,
    sample_rate: int = 10,
    fps: int = 12,
) -> List[str]:
    """Generate a progressive drawing GIF by ordering strokes.

    Args:
        method: 'Fu' or 'LineDrawer'
        json_path: required if method='Fu'
        img_path: required if method='LineDrawer'
        video_path: required if method='LineDrawer'
        sample_rate: progressive frame granularity
        fps: gif fps
    """
    try:
        out = _run_ordering_gif(
            method=(method or "").strip(),
            output_dir=os.path.join(os.getcwd(), "output/ordering"),
            json_path=json_path,
            img_path=img_path,
            video_path=video_path,
            sample_rate=sample_rate,
            fps=fps,
        )
        return [out]
    except Exception as e:
        return [f"ordering failed: {e}"]


@mcp.tool()
def animation(
    img0_path: str,
    img1_path: str,
    inner_frames: int = 24,
) -> List[str]:
    """
    Interpolate between two images using RIFE and generate an animation GIF.

    Args:
        img0_path: Path to the first image (start frame).
        img1_path: Path to the second image (end frame).
        inner_frames: Number of interpolated frames between the two images.

    Returns:
        A list containing the output GIF path if successful,
        or a list with a single error message string if failed.
    """
    try:
        # basic validation (important for MCP robustness)
        if not os.path.exists(img0_path):
            return [f"animation failed: img0_path not found ({img0_path})"]
        if not os.path.exists(img1_path):
            return [f"animation failed: img1_path not found ({img1_path})"]
        if inner_frames < 0:
            return ["animation failed: inner_frames must be >= 0"]

        output_dir = os.path.join(os.getcwd(), "output", "animation")

        out_path = _run_animation(
            img0_path=img0_path,
            img1_path=img1_path,
            inner_frames=inner_frames,
            output_dir=output_dir,
        )

        return [out_path]

    except Exception as e:
        return [f"animation failed: {str(e)}"]


# @mcp.tool()
# def sketch2image_dashscope(
#     img_path: str, prompt: str, style: str = "<auto>"
# ) -> List[str]:
#     """
#     Generate an image from a sketch using Aliyun DashScope (Wanx) API.

#     Args:
#         img_path: Path to the local sketch file or URL.
#         prompt: The text prompt describing the image.
#         style: Output style, default '<auto>'. Options: <3d cartoon>, <anime>, <oil painting>, <watercolor>, <sketch>, <chinese painting>, <flat illustration>.
#         api_key: Optional API key. If not provided, uses DASHSCOPE_API_KEY env var.
#     """
#     return _run_sketch2image_dashscope(img_path, prompt, style)


if __name__ == "__main__":
    # Start the MCP server
    mcp.run()
