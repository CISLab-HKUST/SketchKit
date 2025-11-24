"""
Sketch2Model - Minimal Implementation Module

Provides minimal implementations of three sketch-to-3D model conversion methods:
1. Teddy/MonsterMash - Traditional skeleton inflation method
2. SENS - Learning-based neural network method
3. Doubao-Seed3D - Large model-based API method

All methods provide a unified API interface that accepts Sketch objects or image arrays as input.
"""

from typing import Union, Optional

# Import core types
try:
    from sketchkit.core.sketch import Sketch
except ImportError:
    import sys
    from pathlib import Path
    current_dir = Path(__file__).parent
    if str(current_dir.parent / "core") not in sys.path:
        sys.path.insert(0, str(current_dir.parent))
    from sketchkit.core.sketch import Sketch

# Import implementations from methods submodule
from .methods.teddy_monstermash import teddy_from_raster
from .methods.sens_minimal import sketch_to_3d_sens as _sketch_to_3d_sens_impl
from .methods.doubao import sketch_to_3d_doubao_impl


def sketch_to_3d_teddy(
    sketch_input: Union[Sketch, str, 'np.ndarray'],
    output_path: Optional[str] = None,
    **kwargs
) -> str:
    """
    Convert sketch to 3D model using Teddy/MonsterMash method.
    
    Args:
        sketch_input: Input sketch, can be Sketch object, file path, or numpy array
        output_path: Output OBJ file path (if None, will be auto-generated)
        **kwargs: Additional parameters
            - triangle_opts: Triangle library options (default: "pqa25QYY")
            - subs_factor: Subsampling factor (default: 2)
            - smooth_factor: Smoothing factor (default: 10)
            - default_inflation: Default inflation amount (default: 2.0)
            - region_inflations: List of inflation amounts per region (optional)
            - check_single_contour: Whether to check single-connected contour (default: True)
    
    Returns:
        str: Output OBJ file path
    
    Example:
        >>> from sketchkit.core.sketch import Sketch, Path, Curve, Vertex, Point
        >>> # Create sketch...
        >>> output = sketch_to_3d_teddy(sketch, "output.obj")
    """
    import numpy as np
    from sketchkit.renderer.cairo_renderer import CairoRenderer
    import os
    import tempfile
    import time
    
    # Handle different input types
    if isinstance(sketch_input, Sketch):
        renderer = CairoRenderer(canvas_size=512, canvas_color=(1, 1, 1))
        raster_image = renderer.render(sketch_input, fit_canvas=True)
        if raster_image.ndim == 3:
            raster_image = np.dot(raster_image[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
    elif isinstance(sketch_input, str):
        from PIL import Image
        img = Image.open(sketch_input).convert('L')
        raster_image = np.array(img, dtype=np.uint8)
    else:
        raster_image = np.asarray(sketch_input, dtype=np.uint8)
        if raster_image.ndim == 3:
            raster_image = np.dot(raster_image[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
    
    # Generate output path
    if output_path is None:
        temp_dir = tempfile.gettempdir()
        timestamp = int(time.time() * 1000)
        output_path = os.path.join(temp_dir, f"teddy_output_{timestamp}.obj")
    
    return teddy_from_raster(raster_image, output_path=output_path, **kwargs)


def sketch_to_3d_doubao(
    sketch_input: Union[Sketch, str, 'np.ndarray'],
    output_path: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    **kwargs
) -> str:
    """
    Convert sketch to 3D model using Doubao-Seed3D API.
    
    Args:
        sketch_input: Input sketch, can be Sketch object, file path, or numpy array
        output_path: Output file path (if None, will be auto-generated)
        api_key: Doubao API key (if None, will read from environment variable DOUBAO_API_KEY)
        api_base: API base URL (if None, uses default)
        **kwargs: Additional parameters
            - model_name: Model name (default: "doubao-seed3d-1-0-250928")
            - file_format: Output format (default: "glb")
            - subdivision_level: Subdivision level (default: "medium")
            - max_wait_time: Maximum wait time in seconds (default: 600)
    
    Returns:
        str: Output file path
    
    Example:
        >>> output = sketch_to_3d_doubao(sketch, "output.glb", api_key="your_key")
    """
    import numpy as np
    from sketchkit.renderer.cairo_renderer import CairoRenderer
    import os
    import tempfile
    import time
    
    # Handle different input types
    if isinstance(sketch_input, Sketch):
        renderer = CairoRenderer(canvas_size=512, canvas_color=(1, 1, 1))
        raster_image = renderer.render(sketch_input, fit_canvas=True)
        if raster_image.ndim == 2 or (raster_image.ndim == 3 and raster_image.shape[2] == 1):
            raster_image = np.stack([raster_image] * 3, axis=-1)
        elif raster_image.ndim == 3 and raster_image.shape[2] == 4:
            raster_image = raster_image[:, :, :3]
    elif isinstance(sketch_input, str):
        from PIL import Image
        img = Image.open(sketch_input).convert('RGB')
        raster_image = np.array(img, dtype=np.uint8)
    else:
        raster_image = np.asarray(sketch_input, dtype=np.uint8)
        if raster_image.ndim == 2:
            raster_image = np.stack([raster_image] * 3, axis=-1)
        elif raster_image.ndim == 3 and raster_image.shape[2] == 4:
            raster_image = raster_image[:, :, :3]
    
    # Generate output path
    if output_path is None:
        temp_dir = tempfile.gettempdir()
        timestamp = int(time.time() * 1000)
        file_format = kwargs.get('file_format', 'glb')
        output_path = os.path.join(temp_dir, f"doubao_output_{timestamp}.{file_format}")
    
    return sketch_to_3d_doubao_impl(
        raster_image,
        output_path=output_path,
        api_key=api_key,
        api_base=api_base,
        **kwargs
    )


def sketch_to_3d(
    sketch_input: Union[Sketch, str, 'np.ndarray'],
    method: str = 'sens',
    output_path: Optional[str] = None,
    **kwargs
) -> str:
    """
    Unified sketch-to-3D model conversion interface.
    
    Args:
        sketch_input: Input sketch, can be Sketch object, file path, or numpy array
        method: Method to use, options: 'teddy', 'sens', 'doubao-seed3d'
        output_path: Output file path (if None, will be auto-generated)
        **kwargs: Method-specific parameters
    
    Returns:
        str: Output file path
    
    Example:
        >>> # Use Teddy method
        >>> output = sketch_to_3d(sketch, method='teddy', output_path='output.obj')
        >>> 
        >>> # Use SENS method
        >>> output = sketch_to_3d(sketch, method='sens', output_path='output.obj')
        >>> 
        >>> # Use Doubao-Seed3D method
        >>> output = sketch_to_3d(sketch, method='doubao-seed3d', output_path='output.glb', api_key='your_key')
    """
    if method == 'teddy':
        return sketch_to_3d_teddy(sketch_input, output_path, **kwargs)
    elif method == 'sens':
        return _sketch_to_3d_sens_impl(sketch_input, output_path, **kwargs)
    elif method in ['doubao-seed3d', 'doubao']:
        return sketch_to_3d_doubao(sketch_input, output_path, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Available methods: 'teddy', 'sens', 'doubao-seed3d'")


def sketch_to_3d_sens(
    sketch_input: Union[Sketch, str, 'np.ndarray'],
    output_path: Optional[str] = None,
    **kwargs
) -> str:
    """
    Convert sketch to 3D model using SENS method.
    
    Args:
        sketch_input: Input sketch, can be Sketch object, file path, or numpy array
        output_path: Output OBJ file path (if None, will be auto-generated)
        **kwargs: Additional parameters
            - model_path: Model path (default: None, uses auto-downloaded model)
            - device: Computing device (default: 'cpu')
    
    Returns:
        str: Output OBJ file path
    """
    return _sketch_to_3d_sens_impl(sketch_input, output_path, **kwargs)


__all__ = [
    'sketch_to_3d',
    'sketch_to_3d_teddy',
    'sketch_to_3d_sens',
    'sketch_to_3d_doubao',
]
