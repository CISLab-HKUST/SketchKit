"""
Sketch2Model - Unified Interface

Provides a unified API for sketch-to-3D model conversion using different methods.
"""

__version__ = "0.0.1"

from typing import Union, Optional

# Import Modeler class
from .modeler import Modeler

# Import method implementations
from .methods.teddy_monstermash import sketch_to_3d_teddy_impl
from .methods.sens_standalone import sketch_to_3d_sens_standalone
from .methods.doubao import sketch_to_3d_doubao_impl

try:
    from sketchkit.core.sketch import Sketch
except ImportError:
    import sys
    from pathlib import Path
    current_dir = Path(__file__).parent
    if str(current_dir.parent / "core") not in sys.path:
        sys.path.insert(0, str(current_dir.parent))
    from sketchkit.core.sketch import Sketch


def sketch_to_3d(
    sketch_input: Union[Sketch, str, 'np.ndarray'],
    method: str,
    output_path: Optional[str] = None,
    **kwargs
) -> str:
    """
    Unified function to convert a sketch to a 3D model using various methods.
    
    Args:
        sketch_input: Input sketch, can be Sketch object, file path, or numpy array
        method: Conversion method ('teddy', 'sens', or 'doubao-seed3d')
        output_path: Output file path (if None, will be auto-generated)
        **kwargs: Method-specific parameters
    
    Returns:
        str: Path to generated 3D model file
    
    Example:
        >>> from sketchkit.sketch2model import sketch_to_3d
        >>> output = sketch_to_3d('sketch.png', method='teddy', output_path='output.obj')
    """
    if method == 'teddy':
        return sketch_to_3d_teddy_impl(sketch_input, output_path, **kwargs)
    elif method == 'sens':
        return sketch_to_3d_sens_standalone(sketch_input, output_path, **kwargs)
    elif method in ['doubao-seed3d', 'doubao']:
        return sketch_to_3d_doubao_impl(sketch_input, output_path, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Available methods: 'teddy', 'sens', 'doubao-seed3d'")


__all__ = ['sketch_to_3d', 'Modeler']
