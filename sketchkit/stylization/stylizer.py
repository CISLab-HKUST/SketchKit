from typing import Union, Optional, Dict, Any
from PIL import Image

from sketchkit.core.sketch import Sketch
from .NeuralBrushstroke import NeuralBrushstroke


class Stylizer:
    """
    A unified wrapper class for stylizing sketches using various methods.
    
    This class provides a consistent API across different stylization methods,
    allowing users to switch between methods easily.
    
    Attributes:
        method (str): The stylization method being used.
        _stylizer: The underlying method-specific stylizer instance.
    """
    
    def __init__(
        self,
        method: str = "NeuralBrushstroke",
        **kwargs: Any,
    ):
        """
        Initialize the Stylizer with the specified method.
        
        Args:
            method: The stylization method to use.
                Currently supported: 'NeuralBrushstroke' (default).
                Future methods: 'method2', 'method3', etc.
            **kwargs: Method-specific parameters.
                For 'NeuralBrushstroke', see NeuralBrushstroke.__init__ for available parameters.
        """
        self.method = method
        
        # Initialize method-specific stylizer
        if method == "NeuralBrushstroke":
            self._stylizer = NeuralBrushstroke(**kwargs)
        else:
            raise ValueError(
                f"Unknown method: {method}. "
                f"Available methods: 'NeuralBrushstroke'"
            )
    
    def run(
        self,
        sketch: Sketch,
        output_path: Optional[str] = None,
        **kwargs: Any,
    ) -> Image.Image:
        """
        Stylize a sketch using the selected method.
        
        Args:
            sketch: Input vector sketch.
            output_path: Output file path. If None, uses a temporary path.
            **kwargs: Method-specific parameters (passed to the underlying method).
        
        Returns:
            Stylized image as PIL Image.
        """
        return self._stylizer.run(sketch, output_path=output_path, **kwargs)
    
    def run_from_png(
        self,
        png_path: Union[str, "Path"],
        output_path: Optional[str] = None,
        **kwargs: Any,
    ) -> Image.Image:
        """
        Stylize an existing PNG image using the selected method.
        
        Args:
            png_path: Path to the input PNG geometry image.
            output_path: Output file path. If None, uses a temporary path.
            **kwargs: Method-specific parameters (passed to the underlying method).
        
        Returns:
            Stylized image as PIL Image.
        """
        return self._stylizer.run_from_png(png_path, output_path=output_path, **kwargs)
    
    @property
    def model_name(self) -> Optional[str]:
        """Get the model name (if supported by the method)."""
        if hasattr(self._stylizer, 'model_name'):
            return self._stylizer.model_name
        return None
    
    @property
    def style_id(self) -> Optional[str]:
        """Get the style ID (if supported by the method)."""
        if hasattr(self._stylizer, 'style_id'):
            return self._stylizer.style_id
        return None

