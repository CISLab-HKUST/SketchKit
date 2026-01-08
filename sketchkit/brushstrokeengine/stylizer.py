"""
Brushstroke Stylizer for sketch-to-stylized image conversion.
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Union, Optional, Dict, Any
from PIL import Image
from datetime import datetime

from sketchkit.core.sketch import Sketch
from sketchkit.renderer.cairo_renderer import CairoRenderer, CairoRenderOptions
from .models import (
    download_models,
    check_models_available,
    get_model_path,
    get_models_dir,
)


class BrushstrokeStylizer:
    """
    A class for stylizing sketches using the Neural Brushstroke Engine (NeuBE).
    
    This class provides functionality to convert vector sketches into stylized images
    using various brushstroke styles from the NeuBE model.
    
    Attributes:
        model_name (str): The model name to use ('style1' or 'style2').
        style_id (str): The style ID to use for stylization.
        models_dir (Path): Directory containing the models.
        device (str): Device to use (for future use, currently not used).
    """
    
    def __init__(
        self,
        model_name: str = "style2",
        style_id: str = "playdoh10",
        models_dir: Optional[Union[str, Path]] = None,
        device: str = "cuda",
        auto_download: bool = True,
        **kwargs: Any,
    ):
        """
        Initialize the BrushstrokeStylizer.
        
        Args:
            model_name: The model name to use. Options: 'style1', 'style2'. 
                Defaults to 'style2'.
            style_id: The style ID to use for stylization. Defaults to 'playdoh10'.
            models_dir: Directory containing the models. If None, uses default location.
            device: Device to use (currently not used, kept for API consistency).
            auto_download: If True, automatically download models if not found.
            **kwargs: Additional parameters (currently unused).
        """
        self.model_name = model_name
        self.style_id = style_id
        self.device = device
        # Per-instance default options that users can override via kwargs in run/run_from_png
        self.default_options = {
            "feature_blending_level": kwargs.get("feature_blending_level", 2),
            "color_mode": kwargs.get("color_mode", "1"),
            "crop_margin": kwargs.get("crop_margin", 10),
            "on_white": kwargs.get("on_white", True),
            "render_mode": kwargs.get("render_mode", "clear"),
            "stitching_mode": kwargs.get("stitching_mode", "all"),
            "style_id2": kwargs.get("style_id2", None),
            "style_blend_alpha": kwargs.get("style_blend_alpha", 0.5),
            "no_uvs_mapping": kwargs.get("no_uvs_mapping", False),
        }
        
        # Set models directory
        if models_dir is None:
            self.models_dir = get_models_dir()
        else:
            self.models_dir = Path(models_dir)
        
        # Check and download models if needed
        if not check_models_available(model_name, self.models_dir):
            if auto_download:
                print(f"📥 Models not found. Downloading models for {model_name}...")
                download_models(models_dir=self.models_dir)
            else:
                raise FileNotFoundError(
                    f"Models not found for {model_name} at {self.models_dir}. "
                    f"Set auto_download=True or call download_models() first."
                )
        
        # Verify model is available
        self.model_path = get_model_path(model_name, self.models_dir)
        self.checkpoint_path = self.model_path / "snapshot.pkl"
        
        # Default library path
        self.library_path = self.model_path / "brush_libs" / "projected_training_styles.pkl"
        
        if not self.library_path.exists():
            # Try alternative library paths
            alt_libs = [
                self.model_path / "brush_libs" / "projected_novel_styles.pkl",
                self.model_path / "brush_libs" / "clip_brushes.pkl",
            ]
            for alt_lib in alt_libs:
                if alt_lib.exists():
                    self.library_path = alt_lib
                    break
            else:
                print(f"⚠️  Warning: No brush library found at {self.library_path}")

        # Default geometry encoder checkpoint (used by PaintEngineFactory)
        self.encoder_checkpoint = None
        geo_enc_dir = self.models_dir / "geo_encoders"
        weak_ckpt = geo_enc_dir / "weak.pt"
        strong_ckpt = geo_enc_dir / "strong.pt"
        if weak_ckpt.exists():
            self.encoder_checkpoint = weak_ckpt
        elif strong_ckpt.exists():
            self.encoder_checkpoint = strong_ckpt
    
    def run(
        self,
        sketch: Sketch,
        output_path: Optional[str] = None,
        canvas_size: Optional[int] = None,
        style_id: Optional[str] = None,
        library_path: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> Image.Image:
        """
        Stylize a sketch using the brushstroke engine.
        
        Args:
            sketch: Input vector sketch.
            output_path: Output file path. If None, uses a temporary path.
            canvas_size: Canvas size for rendering. If None, uses sketch's actual width and height.
            style_id: Style ID to use. If None, uses the instance default.
            library_path: Path to brush library. If None, uses default.
            **kwargs: Additional parameters for stylization:
                - feature_blending_level (int): Feature blending level. Defaults to 2.
                - color_mode (str): Color mode. Defaults to '1'.
                - crop_margin (int): Crop margin. Defaults to 10.
                - on_white (bool): Render on white background. Defaults to True.
        
        Returns:
            Stylized image as PIL Image.
        """
        # Use provided style_id or instance default
        style_id = style_id or self.style_id
        
        # Create temporary directory for intermediate files
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path.cwd() / "brushstroke_output" / timestamp
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / f"stylized_{style_id}.png")
        else:
            output_path = Path(output_path)
            output_dir = output_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_path)
        
        # Step 1: Render sketch to PNG (use sketch's actual dimensions if canvas_size is None)
        png_path = self._render_sketch_to_png(
            sketch, 
            output_dir=Path(output_path).parent,
            canvas_size=canvas_size,
        )
        
        # Step 2: Stylize using forger
        stylized_path = self._run_forger_stylization(
            png_path=png_path,
            style_id=style_id,
            library_path=library_path,
            output_dir=Path(output_path).parent,
            output_file=output_path,  # Use the specified output_path
            **kwargs,
        )
        
        # Step 3: Load and return stylized image
        stylized_image = Image.open(stylized_path)
        return stylized_image

    def run_from_png(
        self,
        png_path: Union[str, Path],
        output_path: Optional[str] = None,
        style_id: Optional[str] = None,
        library_path: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> Image.Image:
        """
        Stylize an existing PNG geometry image using the brushstroke engine.

        This method assumes the input PNG is a line drawing (black on white),
        similar to the --geom_image expected by sketchkit.brushstrokeengine.forger.viz.paint_image_main.

        Args:
            png_path: Path to the input PNG geometry image.
            output_path: Output file path. If None, uses a temporary path.
            style_id: Style ID to use. If None, uses the instance default.
            library_path: Path to brush library. If None, uses default.
            **kwargs: Additional parameters for stylization (same as run()).

        Returns:
            Stylized image as PIL Image.
        """
        # Use provided style_id or instance default
        style_id = style_id or self.style_id

        png_path = Path(png_path)
        if not png_path.exists():
            raise FileNotFoundError(f"Input PNG not found: {png_path}")

        # Determine output path and directory
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path.cwd() / "brushstroke_output" / timestamp
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_dir / f"stylized_{style_id}.png")
        else:
            output_path = Path(output_path)
            output_dir = output_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = str(output_path)

        # Run stylization directly on the input PNG
        stylized_path = self._run_forger_stylization(
            png_path=str(png_path),
            style_id=style_id,
            library_path=library_path,
            output_dir=output_dir,
            output_file=output_path,  # Use the specified output_path
            **kwargs,
        )

        stylized_image = Image.open(stylized_path)
        return stylized_image
    
    def _get_option(self, name: str, kwargs: Dict[str, Any]):
        """Get option value from kwargs or fall back to instance default."""
        if name in kwargs:
            return kwargs[name]
        return self.default_options.get(name)
    
    def _render_sketch_to_png(
        self,
        sketch: Sketch,
        output_dir: Path,
        canvas_size: Optional[int] = None,
    ) -> str:
        """Render sketch to PNG using CairoRenderer.
        
        Args:
            sketch: Input vector sketch.
            output_dir: Directory to save the rendered PNG.
            canvas_size: Canvas size for rendering. If None, uses sketch's actual width and height.
        
        Returns:
            Path to the rendered PNG file.
        """
        # Use sketch's actual dimensions if canvas_size is not specified
        if canvas_size is None:
            # Use sketch's width and height, but fallback to 512x512 if invalid
            canvas_width = int(sketch.width) if sketch.width > 0 else 512
            canvas_height = int(sketch.height) if sketch.height > 0 else 512
        else:
            # If canvas_size is specified, use it for both width and height (square canvas)
            canvas_width = canvas_height = canvas_size
        
        render_opts = CairoRenderOptions(
            canvas_size=(canvas_width, canvas_height),
            background_color=(1, 1, 1),
        )
        renderer = CairoRenderer(render_opts)
        raster_image = renderer.render(sketch, render_opts)

        # Save as PNG
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        png_filename = f"sketch_{timestamp}.png"
        png_path = output_dir / png_filename
        png_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(raster_image, Image.Image):
            raster_image.save(png_path, "PNG")
        else:
            raster_image_pil = Image.fromarray(raster_image.astype(np.uint8), "RGB")
            raster_image_pil.save(png_path, "PNG")

        return str(png_path)
    
    def _import_forger(self):
        """
        Import forger module using relative imports.
        
        Since stylizer.py and forger/thirdparty are in the same directory,
        we can use relative imports for cleaner code.
        
        The forger module internally uses absolute imports (import forger.xxx),
        so we need to register 'forger' and 'thirdparty' in sys.modules
        as top-level modules before importing.
        
        Returns:
            The forger module.
        
        Raises:
            ImportError: If forger module is not found.
        """
        try:
            # First, register 'forger' and 'thirdparty' as top-level modules in sys.modules
            # This allows forger's internal code to use absolute imports like 'import forger.xxx'
            from . import forger
            from . import thirdparty
            
            # Register as top-level modules so internal absolute imports work
            if 'forger' not in sys.modules:
                sys.modules['forger'] = forger
            if 'thirdparty' not in sys.modules:
                sys.modules['thirdparty'] = thirdparty
            
            # Now import the required submodules using absolute imports
            # This matches the original project's import style and automatically
            # adds modules to package namespaces (e.g., forger.ui.library)
            from .thirdparty.stylegan2_ada_pytorch import dnnlib
            from .forger import ui
            
            # Use absolute imports to automatically add modules to forger.ui namespace
            # This is the same way the original project imports modules
            import forger.ui.library
            import forger.ui.brush
            import forger.util.img_proc as img_proc
            import forger.viz.style_transfer as style_transfer
            import forger.viz.visualize as visualize
            
            if hasattr(forger, "ui") and hasattr(forger.ui, "brush") and hasattr(forger.ui, "library"):
                return forger
        except ImportError as e:
            raise ImportError(
                "forger module not found. Please ensure brushstroke_engine is in sketchkit/brushstrokeengine:\n"
                "  1. Place brushstroke_engine code in sketchkit/brushstrokeengine/\n"
                "  2. Run: pip install -e .\n\n"
                "The forger module should be automatically included as a subpackage.\n"
                f"Original error: {e}"
            ) from e
    
    def _run_forger_stylization(
        self,
        png_path: str,
        style_id: str,
        library_path: Optional[Union[str, Path]] = None,
        output_dir: Optional[Path] = None,
        output_file: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ) -> str:
        """Run forger stylization by directly calling forger functions.
        
        This method directly imports and uses forger modules without requiring
        pip install -e . Only needs the brushstroke_engine directory in the filesystem.
        """
        # Import forger module (will add to sys.path if needed)
        forger = self._import_forger()
        
        # Import required modules
        import torch
        from skimage.io import imsave, imread
        try:
            import tqdm
        except ImportError:
            raise ImportError(
                "tqdm is required. Install with: pip install tqdm>=4.56.0"
            )
        
        # Use provided library_path or default
        if library_path is None:
            library_path = self.library_path
        else:
            library_path = Path(library_path)
        
        library_path = str(library_path)
        
        # Set output directory
        if output_dir is None:
            output_dir = Path(png_path).parent
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # If output_file is provided, use it directly; otherwise generate from PNG name
        use_custom_output_file = output_file is not None
        output_file_path = None
        if use_custom_output_file:
            output_file_path = Path(output_file)
            output_dir = output_file_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            # Extract prefix from output_file (without extension)
            output_file_prefix = str(output_file_path.with_suffix(''))
        else:
            # Generate output file prefix from PNG name
            png_name = Path(png_path).stem
            output_file_prefix = str(output_dir / png_name)
        
        # Get parameters
        feature_blending_level = self._get_option('feature_blending_level', kwargs)
        color_mode = self._get_option('color_mode', kwargs)
        crop_margin = self._get_option('crop_margin', kwargs)
        on_white = self._get_option('on_white', kwargs)
        render_mode = self._get_option('render_mode', kwargs)
        stitching_mode = self._get_option('stitching_mode', kwargs)
        style_id2 = self._get_option('style_id2', kwargs)
        style_blend_alpha = self._get_option('style_blend_alpha', kwargs)
        no_uvs_mapping = self._get_option('no_uvs_mapping', kwargs)
        
        print(f"🎨 Running brushstroke stylization...")
        print(f"   Style: {style_id}")
        print(f"   Model: {self.model_name}")
        
        try:
            # Setup device
            if isinstance(self.device, str):
                device = torch.device(self.device if self.device != "cuda" else (0 if torch.cuda.is_available() else "cpu"))
            else:
                device = torch.device(self.device)
            
            # forger is already imported via self._import_forger() above
            # img_proc is already imported and available as forger.util.img_proc

            # Create engine
            encoder_checkpoint = kwargs.get("encoder_checkpoint", self.encoder_checkpoint)
            engine = forger.ui.brush.PaintEngineFactory.create(
                encoder_checkpoint=str(encoder_checkpoint) if encoder_checkpoint else None,
                gan_checkpoint=str(self.checkpoint_path),
                device=device,
            )
            
            # Load brush library
            library = forger.ui.library.BrushLibrary.from_arg(
                library_path,
                z_dim=engine.G.z_dim,
            )
            
            # Setup brush options
            brush_options = forger.ui.brush.GanBrushOptions()
            brush_options.debug = False
            brush_options.enable_uvs_mapping = not no_uvs_mapping
            
            # Set colors if specified
            if color_mode is not None:
                if color_mode in ['1', '2']:
                    # forger is already imported via self._import_forger() above
                    opts = forger.ui.brush.GanBrushOptions()
                    if color_mode == '1':
                        library.set_style(style_id, opts)
                    elif color_mode == '2':
                        library.set_style(style_id2 or style_id, opts)
                    colors = engine.uvs_mapper.get_colors_raw(opts)
                    brush_options.set_color(0, colors[0, :, 0] / 2 + 0.5)
                    brush_options.set_color(1, colors[0, :, 1] / 2 + 0.5)
            
            # Set style
            if style_id2 is None:
                library.set_style(style_id, brush_options)
            else:
                library.set_interpolated_style(
                    style_id, style_id2, style_blend_alpha, brush_options
                )
            
            # Read geometry image
            def _read_any_geo(fname):
                """Read geometry image (from paint_image_main.py)."""
                img = torch.from_numpy(imread(fname)).to(torch.float32)
                if len(img.shape) == 2:
                    img = img.unsqueeze(-1)
                if img.shape[2] == 3:
                    img = img[..., :3].mean(dim=2).unsqueeze(-1)
                elif img.shape[2] == 4:
                    mean = img[..., :3].mean(dim=2)
                    alpha = img[..., 3] / 255
                    img = (mean * alpha + 255 * (1 - alpha)).unsqueeze(-1)
                mn = torch.min(img)
                if mn > 0:
                    img = img - mn
                mx = torch.max(img)
                if 0 < mx < 255:
                    img = img * (255.0 / mx.item())
                img = img.to(torch.uint8).numpy()
                img = (forger.util.img_proc.threshold_img(img, to_float=False).astype(np.float32) * 255).astype(np.uint8)
                return img
            
            def pad_geo(geo, crop_margin):
                """Pad geometry image."""
                geo_padded = np.ones(
                    (geo.shape[0] + crop_margin, geo.shape[1] + crop_margin, geo.shape[2]),
                    dtype=np.uint8
                ) * 255
                geo_padded[crop_margin:, crop_margin:, :] = geo
                return geo_padded
            
            # Read and pad geometry
            patch_width = engine.G.img_resolution
            geom = _read_any_geo(png_path)
            orig_geo_shape = geom.shape
            geom = pad_geo(geom, crop_margin)
            
            # Generate stitching crops
            # forger is already imported via self._import_forger() above
            stitching_crops, geom = forger.viz.style_transfer.generate_stitching_crops(
                geom, patch_width, mode=stitching_mode, overlap_margin=crop_margin * 2
            )
            
            # Initialize result canvas
            result = np.zeros((geom.shape[0], geom.shape[1], 4), dtype=np.uint8)
            
            # Create painting helper
            helper = forger.ui.brush.PaintingHelper(engine)
            helper.make_new_canvas(
                result.shape[0], result.shape[1], feature_blending=feature_blending_level
            )
            helper.set_render_mode(render_mode)
            
            # Render each patch
            with torch.no_grad():
                for i in tqdm.tqdm(range(len(stitching_crops)), total=len(stitching_crops)):
                    y = stitching_crops[i][0]
                    x = stitching_crops[i][1]
                    brush_options.set_position(x, y)
                    geom_patch = 255 - geom[y:y+patch_width, x:x+patch_width, :]
                    
                    res, _, meta = helper.render_stroke(
                        geom_patch, None, brush_options,
                        meta={'x': x, 'y': y, 'crop_margin': crop_margin}
                    )
                    
                    res_y = meta['y']
                    res_x = meta['x']
                    res_height = res.shape[0]
                    res_width = res.shape[1]
                    result[res_y:res_y+res_height, res_x:res_x+res_width, :] = res
            
            # Apply white background if requested
            if on_white:
                alpha = result[..., 3:].astype(np.float32) / 255
                result = result[..., :3].astype(np.float32) * alpha + 255 * (1 - alpha)
                result[..., 3:] = 255
                result = result.clip(0, 255).astype(np.uint8)
            
            # Crop to original size
            result = result[
                crop_margin:crop_margin + orig_geo_shape[0],
                crop_margin:crop_margin + orig_geo_shape[1],
                :
            ]
            
            # Generate output filename
            if use_custom_output_file:
                # Use the provided output_file directly
                final_output_file = str(output_file_path)
            else:
                # Generate filename from prefix
                style_name = style_id
                if style_id2 is not None:
                    style_name += f'_{style_blend_alpha:.1f}{style_id2}'
                final_output_file = f"{output_file_prefix}_{render_mode}_{style_name}.png"
            
            # Save result
            imsave(final_output_file, result)
            print(f"✅ Stylization complete: {final_output_file}")
            return final_output_file
            
        except Exception as e:
            raise RuntimeError(f"Stylization failed: {e}") from e

