"""
Teddy/MonsterMash sketch-to-3D conversion module.

This module provides functions to convert sketches to 3D models using
the full MonsterMash/Teddy skeleton-based inflation algorithm.
"""

import numpy as np
import os
from typing import Tuple, Optional, Union
from PIL import Image, ImageDraw

# Try to import scipy for advanced image operations
try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    ndimage = None
    import warnings
    warnings.warn(
        "scipy not available. Some image processing operations may be limited. "
        "Install with: pip install scipy",
        UserWarning
    )

# Try to import skimage for contour finding (optional but helpful)
try:
    from skimage import measure
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    measure = None

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
    # Try to find the .so file in build directory
    current_file = Path(__file__).resolve()
    
    # Only search new structure: third_party/teddy/build
    possible_paths = [
        # New structure: third_party/teddy/build (relative to project root)
        current_file.parent.parent.parent.parent / "third_party" / "teddy" / "build",
        current_file.parent.parent.parent / "third_party" / "teddy" / "build",
        # Try common installation locations
        Path.home() / ".local" / "lib" / "python3.8" / "site-packages" / "teddy_sketch2model",
        Path.home() / ".local" / "lib" / "python3.9" / "site-packages" / "teddy_sketch2model",
        Path.home() / ".local" / "lib" / "python3.10" / "site-packages" / "teddy_sketch2model",
    ]
    
    for build_dir in possible_paths:
        if build_dir.exists() and build_dir.is_dir():
            # Check if .so file exists in this directory
            so_files = list(build_dir.glob("teddy_sketch2model*.so"))
            if not so_files:
                continue  # Directory exists but no .so file
            
            # Add to sys.path
            if str(build_dir) not in sys.path:
                sys.path.insert(0, str(build_dir))
            # Try to import again
            try:
                import teddy_sketch2model
                TEDDY_CPP_AVAILABLE = True
                break
            except ImportError:
                continue

if not TEDDY_CPP_AVAILABLE:
    print("Warning: teddy_sketch2model C++ module not available. "
          "Please compile the module first.")
    print(f"  Build instructions: See third_party/teddy/README.md")
    print(f"  Searched paths:")
    for p in possible_paths:
        exists = "✓" if p.exists() else "✗"
        if p.exists():
            # Check if .so file exists
            so_files = list(p.glob("teddy_sketch2model*.so"))
            so_status = f" (has .so: {so_files[0].name})" if so_files else " (no .so file)"
            print(f"    {exists} {p}{so_status}")
        else:
            print(f"    {exists} {p}")


def preprocess_raster_for_teddy(
    raster_sketch: Union[Image.Image, np.ndarray],
    target_size: int = 512
) -> np.ndarray:
    """
    Preprocess raster sketch for Teddy/MonsterMash processing.
    
    Converts RGB image to grayscale, resizes to target size, and ensures
    correct polarity (white background, black lines).
    
    Args:
        raster_sketch: Input raster image as PIL Image or numpy array (H, W, 3) or (H, W) as uint8
        target_size: Target size for resizing (default: 512)
        
    Returns:
        Preprocessed grayscale image (target_size, target_size) as uint8
        with white background (255) and black lines (0)
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(raster_sketch, Image.Image):
        raster_sketch = np.array(raster_sketch.convert('RGB'), dtype=np.uint8)
    
    # Ensure it's a numpy array
    raster_sketch = np.asarray(raster_sketch, dtype=np.uint8)
    
    # Convert to grayscale if needed
    if len(raster_sketch.shape) == 3:
        img = Image.fromarray(raster_sketch)
        gray = np.array(img.convert('L'), dtype=np.uint8)
    else:
        gray = raster_sketch.copy()
    
    # Ensure uint8
    if gray.dtype != np.uint8:
        gray = (gray * 255).astype(np.uint8) if gray.max() <= 1.0 else gray.astype(np.uint8)
    
    # Resize to target size using PIL
    if gray.shape[0] != target_size or gray.shape[1] != target_size:
        img = Image.fromarray(gray)
        gray = np.array(img.resize((target_size, target_size), Image.Resampling.LANCZOS), dtype=np.uint8)
    
    # Check polarity: if mean is < 127, likely black background with white lines
    # We want white background (255) with black lines (0)
    mean_val = gray.mean()
    if mean_val < 127:
        gray = 255 - gray
    
    return gray


def _calculate_contour_area(contour_points: np.ndarray) -> float:
    """Calculate area of a contour using the shoelace formula.
    
    Args:
        contour_points: Array of (x, y) coordinates, shape (N, 2)
    
    Returns:
        Area of the contour
    """
    if len(contour_points) < 3:
        return 0.0
    # Ensure 2D array
    if contour_points.ndim != 2 or contour_points.shape[1] != 2:
        return 0.0
    x = contour_points[:, 0]
    y = contour_points[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _fill_contour(contour_points: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Fill a contour using PIL ImageDraw.
    
    Args:
        contour_points: Array of (x, y) coordinates, shape (N, 2)
        shape: Output image shape (height, width)
    
    Returns:
        Filled binary image as uint8 array
    """
    img = Image.new('L', (shape[1], shape[0]), 0)
    draw = ImageDraw.Draw(img)
    # Convert contour points to tuple list for PIL
    # contour_points should be (x, y) format
    points = [(int(float(p[0])), int(float(p[1]))) for p in contour_points]
    if len(points) > 2:
        draw.polygon(points, fill=255)
    return np.array(img, dtype=np.uint8)


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
    # Step 1: Create binary image (threshold at 127)
    # Black lines (< 127) become 0, white background (>= 127) becomes 255
    binary = np.where(gray_sketch < 127, 255, 0).astype(np.uint8)
    
    # Step 2: Create outline image (white background, black outline)
    outline_img = np.full_like(gray_sketch, 255, dtype=np.uint8)
    outline_img[binary > 0] = 0
    
    # Step 3: Find the largest connected component and create region image
    if SCIPY_AVAILABLE:
        # Use scipy to find connected components (most reliable method)
        labeled, num_features = ndimage.label(binary)
        if num_features == 0:
            raise ValueError("No contour found in sketch. Please ensure the sketch has a closed outline.")
        
        # Get sizes of all components
        sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
        largest_label = np.argmax(sizes) + 1
        largest_area = sizes[largest_label - 1]
        
        # Check for single contour constraint
        if check_single_contour and num_features > 1:
            second_largest_idx = np.argsort(sizes)[-2] + 1
            second_largest_area = sizes[second_largest_idx - 1]
            if second_largest_area > largest_area * 0.25:
                raise ValueError(
                    f"Multiple significant contours detected. "
                    f"Teddy method currently only supports single connected closed contours. "
                    f"Found {num_features} components, largest area: {largest_area:.0f}, "
                    f"second largest: {second_largest_area:.0f}"
                )
        
        # Create region image from largest component
        region_img = np.where(labeled == largest_label, 255, 0).astype(np.uint8)
        
    elif SKIMAGE_AVAILABLE:
        # Use skimage for contour finding (fallback)
        # Note: skimage.measure.find_contours returns (row, col) coordinates as floats
        contours = measure.find_contours(binary, 127, fully_connected='high')
        
        if len(contours) == 0:
            raise ValueError("No contour found in sketch. Please ensure the sketch has a closed outline.")
        
        # Find the largest contour by area
        # skimage returns contours as (row, col) = (y, x), we need (x, y)
        contour_areas = []
        for contour in contours:
            # contour is (N, 2) with (row, col) = (y, x)
            # Convert to (x, y) for area calculation
            xy_contour = np.flip(contour, axis=1)  # Swap to (col, row) = (x, y)
            contour_areas.append(_calculate_contour_area(xy_contour))
        
        largest_idx = np.argmax(contour_areas)
        largest_contour = contours[largest_idx]
        largest_area = contour_areas[largest_idx]
        
        # Check for single contour constraint
        if check_single_contour:
            for i, (contour, area) in enumerate(zip(contours, contour_areas)):
                if i != largest_idx and area > largest_area * 0.25:
                    raise ValueError(
                        f"Multiple significant contours detected. "
                        f"Teddy method currently only supports single connected closed contours. "
                        f"Found {len(contours)} contours, largest area: {largest_area:.0f}, "
                        f"second largest: {area:.0f}"
                    )
        
        # Step 4: Create region image (black background, white filled region)
        # Convert (row, col) to (x, y) for PIL
        xy_contour = np.flip(largest_contour, axis=1)
        region_img = _fill_contour(xy_contour, gray_sketch.shape)
    else:
        # Fallback: simple threshold-based approach (no scipy or skimage)
        if binary.sum() == 0:
            raise ValueError("No contour found in sketch. Please ensure the sketch has a closed outline.")
        # Just use the binary image directly (may not work well for complex shapes)
        region_img = binary
    
    # Optional: Validate connectivity using scipy
    if check_single_contour and SCIPY_AVAILABLE:
        labeled, num_labels = ndimage.label(region_img)
        # Should have 2 components: background (0) and filled region (1)
        if num_labels > 2:
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
    raster_sketch: Union[Image.Image, np.ndarray],
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
        raster_sketch: Input raster image as PIL Image or numpy array (H, W, 3) or (H, W) as uint8
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

