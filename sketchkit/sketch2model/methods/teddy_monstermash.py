"""
Teddy/MonsterMash sketch-to-3D conversion module.

This module provides functions to convert sketches to 3D models using
the full MonsterMash/Teddy skeleton-based inflation algorithm.
"""

import numpy as np
import cv2
import os
from typing import Tuple, Optional, Union
from PIL import Image

try:
    from sketchkit.core.sketch import Sketch
except ImportError:
    Sketch = None  # Will be handled in the function

# Try to import the C++ module
# First, try to find the module in common locations
import sys
from pathlib import Path

TEDDY_CPP_AVAILABLE = False
teddy_sketch2model = None

# Try to import directly first
try:
    import teddy_sketch2model
    TEDDY_CPP_AVAILABLE = True
except ImportError:
    # Try to find the .so file in build_teddy directory
    # Look for build_teddy relative to this file
    current_file = Path(__file__).resolve()
    
    # Calculate possible paths based on file location
    # File is in: sketchkit/sketch2model/methods/teddy_monstermash.py
    # We need: third_party/teddy/build (new structure)
    # Or: external/monstermash/build_teddy (legacy structure)
    possible_paths = [
        # New structure: third_party/teddy/build
        current_file.parent.parent.parent.parent / "third_party" / "teddy" / "build",
        current_file.parent.parent.parent / "third_party" / "teddy" / "build",
        # Legacy structure: external/monstermash/build_teddy
        current_file.parent.parent.parent.parent / "external" / "monstermash" / "build_teddy",
        current_file.parent.parent.parent / "external" / "monstermash" / "build_teddy",
        # Also try the directories themselves (where .so might be)
        current_file.parent.parent.parent.parent / "third_party" / "teddy",
        current_file.parent.parent.parent / "external" / "monstermash",
        # Try common installation locations
        Path.home() / ".local" / "lib" / "python3.8" / "site-packages" / "teddy_sketch2model",
        Path.home() / ".local" / "lib" / "python3.9" / "site-packages" / "teddy_sketch2model",
        Path.home() / ".local" / "lib" / "python3.10" / "site-packages" / "teddy_sketch2model",
    ]
    
    for build_dir in possible_paths:
        if build_dir.exists() and build_dir.is_dir():
            # Add to sys.path
            if str(build_dir) not in sys.path:
                sys.path.insert(0, str(build_dir))
            # Try to import again
            try:
                import teddy_sketch2model
                TEDDY_CPP_AVAILABLE = True
                break
            except ImportError as e:
                # Debug: print the error if it's not just "No module named"
                if "No module named" not in str(e):
                    print(f"  Debug: Import error from {build_dir}: {e}")
                continue

if not TEDDY_CPP_AVAILABLE:
    print("Warning: teddy_sketch2model C++ module not available. "
          "Please compile the module first.")
    print(f"  Build instructions: See third_party/teddy/README.md")
    print(f"  Searched paths:")
    for p in possible_paths:
        exists = "✓" if p.exists() else "✗"
        print(f"    {exists} {p}")


def preprocess_raster_for_teddy(
    raster_sketch: np.ndarray,
    target_size: int = 512
) -> np.ndarray:
    """
    Preprocess raster sketch for Teddy/MonsterMash processing.
    
    Converts RGB image to grayscale, resizes to target size, and ensures
    correct polarity (white background, black lines).
    
    Args:
        raster_sketch: Input raster image (H, W, 3) or (H, W) as uint8
        target_size: Target size for resizing (default: 512)
        
    Returns:
        Preprocessed grayscale image (target_size, target_size) as uint8
        with white background (255) and black lines (0)
    """
    # Convert to grayscale if needed
    if len(raster_sketch.shape) == 3:
        gray = cv2.cvtColor(raster_sketch, cv2.COLOR_RGB2GRAY)
    else:
        gray = raster_sketch.copy()
    
    # Ensure uint8
    if gray.dtype != np.uint8:
        gray = (gray * 255).astype(np.uint8) if gray.max() <= 1.0 else gray.astype(np.uint8)
    
    # Resize to target size
    if gray.shape[0] != target_size or gray.shape[1] != target_size:
        gray = cv2.resize(gray, (target_size, target_size), interpolation=cv2.INTER_AREA)
    
    # Check polarity: if mean is < 127, likely black background with white lines
    # We want white background (255) with black lines (0)
    mean_val = gray.mean()
    if mean_val < 127:
        gray = 255 - gray
    
    return gray


def generate_teddy_input_images(
    gray_sketch: np.ndarray,
    check_single_contour: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate outline and region images for Teddy/MonsterMash sketch2model.
    
    According to SKETCH2MODEL_USAGE.md:
    - outlineImg: White background (255) with black outline (0)
    - regionImg: Black background (0) with white filled region (255)
    
    Args:
        gray_sketch: Preprocessed grayscale image (H, W) as uint8
                    with white background and black lines
        check_single_contour: If True, validate that only one main contour exists
        
    Returns:
        Tuple of (outline_img, region_img) both as uint8 arrays
        
    Raises:
        ValueError: If no contour found or multiple large contours detected
    """
    # Step 1: Extract lines/edges from the sketch
    # Use Canny edge detection for better edge extraction
    edges = cv2.Canny(gray_sketch, 50, 150)
    
    # Also try simple thresholding as fallback
    _, binary = cv2.threshold(gray_sketch, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Combine both methods: use edges where available, otherwise use binary
    # This ensures we capture the sketch lines
    combined_lines = np.maximum(edges, binary)
    
    # Step 2: Create outline image (white background, black outline)
    outline_img = np.full_like(gray_sketch, 255, dtype=np.uint8)
    outline_img[combined_lines > 0] = 0
    
    # Step 3: Find contours and create region image
    # Use the binary image for contour detection
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if len(contours) == 0:
        raise ValueError("No contour found in sketch. Please ensure the sketch has a closed outline.")
    
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    largest_area = cv2.contourArea(largest_contour)
    
    # Check for single contour constraint
    if check_single_contour:
        # Check if there are other significant contours
        total_area = gray_sketch.shape[0] * gray_sketch.shape[1]
        for contour in contours:
            area = cv2.contourArea(contour)
            if contour is not largest_contour and area > largest_area * 0.25:
                raise ValueError(
                    f"Multiple significant contours detected. "
                    f"Teddy method currently only supports single connected closed contours. "
                    f"Found {len(contours)} contours, largest area: {largest_area:.0f}, "
                    f"second largest: {area:.0f}"
                )
    
    # Step 4: Create region image (black background, white filled region)
    region_img = np.zeros_like(gray_sketch, dtype=np.uint8)
    cv2.drawContours(region_img, [largest_contour], -1, color=255, thickness=-1)
    
    # Optional: Validate connectivity
    if check_single_contour:
        num_labels, labels = cv2.connectedComponents(region_img)
        # Should have 2 components: background (0) and filled region (1)
        if num_labels > 2:
            # Multiple disconnected regions - this might be okay for some cases
            # but we'll warn about it
            print(f"Warning: Region image has {num_labels - 1} disconnected components. "
                  f"Expected 1 filled region.")
    
    return outline_img, region_img


def validate_teddy_inputs(
    outline_img: np.ndarray,
    region_img: np.ndarray
) -> bool:
    """
    Validate that outline and region images meet Teddy requirements.
    
    Args:
        outline_img: Outline image (should be white bg, black outline)
        region_img: Region image (should be black bg, white region)
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    # Check dimensions match
    if outline_img.shape != region_img.shape:
        raise ValueError(
            f"Outline and region images must have same dimensions. "
            f"Got {outline_img.shape} vs {region_img.shape}"
        )
    
    # Check outline image: should be mostly white (background)
    outline_mean = outline_img.mean()
    if outline_mean < 200:  # Should be mostly white (255)
        raise ValueError(
            f"Outline image should have white background. "
            f"Mean value is {outline_mean:.1f}, expected > 200"
        )
    
    # Check region image: should have some white pixels (filled region)
    region_max = region_img.max()
    if region_max < 128:
        raise ValueError(
            f"Region image should have white filled region. "
            f"Max value is {region_max}, expected >= 128"
        )
    
    # Check that outline and region are related (outline should be on region boundary)
    # This is a soft check - outline pixels should be near region boundaries
    outline_pixels = (outline_img < 128).sum()
    region_pixels = (region_img > 128).sum()
    
    if outline_pixels == 0:
        raise ValueError("Outline image has no black pixels (no outline detected)")
    
    if region_pixels == 0:
        raise ValueError("Region image has no white pixels (no filled region)")
    
    return True


def teddy_from_raster(
    raster_sketch: np.ndarray,
    output_path: Optional[str] = None,
    temp_dir: Optional[str] = None,
    triangle_opts: str = "pqa25QYY",
    subs_factor: int = 2,
    smooth_factor: int = 10,
    default_inflation: float = 2.0,
    region_inflations: Optional[list] = None,
    check_single_contour: bool = True
) -> str:
    """
    Convert raster sketch to 3D mesh using full Teddy/MonsterMash algorithm.
    
    This function performs the complete pipeline:
    1. Preprocess raster image (grayscale, resize, polarity check)
    2. Generate outline and region images
    3. Call C++ sketch2model function
    4. Save result to OBJ file
    
    Args:
        raster_sketch: Input raster image (H, W, 3) or (H, W) as uint8
        output_path: Output OBJ file path. If None, generates a temp path
        temp_dir: Temporary directory for output. If None, uses system temp
        triangle_opts: Triangle library options (default: "pqa25QYY")
        subs_factor: Subsampling factor (default: 2)
        smooth_factor: Smoothing factor (default: 10)
        default_inflation: Default inflation amount (default: 2.0)
        region_inflations: Optional list of inflation amounts per region
        check_single_contour: If True, validate single connected contour
        
    Returns:
        Path to generated OBJ file
        
    Raises:
        RuntimeError: If C++ module is not available
        ValueError: If preprocessing or validation fails
    """
    if not TEDDY_CPP_AVAILABLE:
        raise RuntimeError(
            "teddy_sketch2model C++ module is not available. "
            "Please compile the module first using CMakeLists_teddy.txt"
        )
    
    # Step 1: Preprocess raster image
    gray_sketch = preprocess_raster_for_teddy(raster_sketch, target_size=512)
    
    # Step 2: Generate outline and region images
    outline_img, region_img = generate_teddy_input_images(
        gray_sketch,
        check_single_contour=check_single_contour
    )
    
    # Step 3: Validate inputs
    validate_teddy_inputs(outline_img, region_img)
    
    # Step 4: Prepare output path
    if output_path is None:
        import tempfile
        if temp_dir is None:
            temp_dir = os.path.join(os.path.expanduser("~"), ".cache", "sketchkit", "teddy")
        os.makedirs(temp_dir, exist_ok=True)
        
        import time
        timestamp = int(time.time() * 1000)
        output_path = os.path.join(temp_dir, f"teddy_{timestamp}.obj")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Step 5: Prepare region inflations
    region_inflations_list = region_inflations if region_inflations else []
    
    # Step 6: Call C++ sketch2model function
    try:
        result_path = teddy_sketch2model.sketch2model(
            outline_img,
            region_img,
            output_path,
            triangle_opts=triangle_opts,
            subs_factor=subs_factor,
            smooth_factor=smooth_factor,
            default_inflation=default_inflation,
            region_inflations=region_inflations_list
        )
        
        return result_path
        
    except Exception as e:
        raise RuntimeError(f"Teddy sketch2model failed: {e}") from e


def sketch_to_3d_teddy_impl(
    sketch_input: Union['Sketch', str, np.ndarray],
    output_path: Optional[str] = None,
    **kwargs
) -> str:
    """
    Convert sketch to 3D model using Teddy/MonsterMash method.
    
    This function provides a unified interface consistent with other sketch-to-3D methods.
    
    Args:
        sketch_input: Input sketch, can be Sketch object, file path (str), or numpy array (H, W) or (H, W, 3) uint8
        output_path: Output OBJ file path (if None, will be auto-generated)
        **kwargs: Additional parameters
            - triangle_opts: Triangle library options (default: "pqa25QYY")
            - subs_factor: Subsampling factor (default: 2)
            - smooth_factor: Smoothing factor (default: 10)
            - default_inflation: Default inflation amount (default: 2.0)
            - region_inflations: List of inflation amounts per region (optional)
            - check_single_contour: Whether to check single-connected contour (default: True)
    
    Returns:
        str: Path to generated OBJ file
        
    Example:
        >>> from sketchkit.sketch2model import sketch_to_3d
        >>> output_path = sketch_to_3d('sketch.png', method='teddy', output_path='output.obj')
        >>> print(f"Model saved to: {output_path}")
    """
    # Handle different input types
    if Sketch is not None and isinstance(sketch_input, Sketch):
        # Sketch object - render to image
        from sketchkit.renderer.cairo_renderer import CairoRenderer
        renderer = CairoRenderer(canvas_size=512, canvas_color=(1, 1, 1))
        raster_image = renderer.render(sketch_input, fit_canvas=True)
        if raster_image.ndim == 3:
            raster_image = np.dot(raster_image[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
    elif isinstance(sketch_input, str):
        # File path - load image
        img = Image.open(sketch_input).convert('L')
        raster_image = np.array(img, dtype=np.uint8)
    else:
        # Assume numpy array
        raster_image = np.asarray(sketch_input, dtype=np.uint8)
        if raster_image.ndim == 3:
            raster_image = np.dot(raster_image[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
    
    # Generate output path if not provided
    if output_path is None:
        import tempfile
        import time
        temp_dir = os.path.join(os.path.expanduser("~"), ".cache", "sketchkit", "teddy")
        os.makedirs(temp_dir, exist_ok=True)
        timestamp = int(time.time() * 1000)
        output_path = os.path.join(temp_dir, f"teddy_{timestamp}.obj")
    
    # Call the existing teddy_from_raster function
    return teddy_from_raster(
        raster_image,
        output_path=output_path,
        triangle_opts=kwargs.get('triangle_opts', 'pqa25QYY'),
        subs_factor=kwargs.get('subs_factor', 2),
        smooth_factor=kwargs.get('smooth_factor', 10),
        default_inflation=kwargs.get('default_inflation', 2.0),
        region_inflations=kwargs.get('region_inflations', None),
        check_single_contour=kwargs.get('check_single_contour', True)
    )

