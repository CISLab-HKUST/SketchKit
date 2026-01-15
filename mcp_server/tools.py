"""
SketchKit MCP Server
Wrapper for SketchKit functionalities (Vectorization, Stylization, Cleanup, Image2Sketch, Sketch2Model).
"""

import os
import sys
import logging
from typing import List, Optional, Union
from PIL import Image
import numpy as np
from pathlib import Path
import trimesh

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
            sketch = cleaner.run(img_np)

            base_name = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(output_dir, f"{base_name}_clean.json")
            _save_sketch_json(sketch, out_path)
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
            pil_image = _load_image_pil(path).convert("RGB")
            res = image2sketch(pil_image, method=method, size=size)

            base_name = os.path.splitext(os.path.basename(path))[0]
            if isinstance(res, Sketch):
                out_path = os.path.join(output_dir, f"{base_name}_{method}.json")
                _save_sketch_json(res, out_path)
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
    try:
        modeler = Modeler(method=method)
    except Exception as e:
        return [f"Failed to initialize Modeler: {e}"]

    def _obj_to_glb(obj_path: Path, glb_path: Path) -> Path:
        """
        Convert OBJ -> GLB for web visualization.
        """
        mesh = trimesh.load(obj_path, force="mesh")
        if mesh is None:
            raise RuntimeError("Failed to load OBJ for conversion.")
        glb_bytes = trimesh.exchange.gltf.export_glb(mesh)
        glb_path.write_bytes(glb_bytes)
        return glb_path

    for path in input_paths:
        try:
            logging.info(f"Generating 3D model for {path}...")
            if not os.path.exists(path):
                results.append(f"File not found: {path}")
                continue

            base_name = os.path.splitext(os.path.basename(path))[0]
            out_ext = ".obj"
            out_path = os.path.join(output_dir, f"{base_name}{out_ext}")

            input_data = path
            if path.endswith(".json"):
                input_data = Sketch.from_json(path)

            final_path = modeler.run(input_data, output_path=out_path, **kwargs)

            _obj_to_glb(Path(final_path), Path(final_path).with_suffix(".glb"))
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
def sketches_to_3d(input_paths: List[str], method: str = "doubao-seed3d", **kwargs) -> List[str]:
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


if __name__ == "__main__":
    # Start the MCP server
    mcp.run()
