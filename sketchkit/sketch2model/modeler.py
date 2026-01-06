"""
Modeler class for sketch-to-3D model conversion.
"""

from typing import Union, Optional
import os

try:
    from sketchkit.core.sketch import Sketch
except ImportError:
    import sys
    from pathlib import Path
    current_dir = Path(__file__).parent
    if str(current_dir.parent / "core") not in sys.path:
        sys.path.insert(0, str(current_dir.parent))
    from sketchkit.core.sketch import Sketch

from .methods.teddy_monstermash import teddy_from_raster
from .methods.sens_standalone import sketch_to_3d_sens_standalone as _sketch_to_3d_sens_impl
from .methods.doubao import sketch_to_3d_doubao_impl


class Modeler:
    """
    A class for sketch-to-3D model conversion.

    This class provides functionality to convert sketches to 3D model files
    using various methods (Teddy, SENS, Doubao-Seed3D).

    Attributes:
        method (str): The conversion method that is used.
        device (str): The device that is used (for methods that support it).
    """

    def __init__(self, method: str = "sens", device: str = "cpu"):
        """
        Initializes the Modeler with the specified method.

        Args:
            method (str, optional): The conversion method to use.
                Options: 'teddy', 'sens', 'doubao-seed3d'. Defaults to 'sens'.
            device (str, optional): The computing device to use.
                Options: 'cpu', 'cuda'. Defaults to 'cpu'.
        """
        self.method = method
        self.device = device

        # Initialize method-specific components if needed
        if method == "sens":
            # SENS will be initialized on first use
            self._sens_inference = None
        elif method == "teddy":
            # Teddy uses C++ module, will be imported on first use
            pass
        elif method in ["doubao-seed3d", "doubao"]:
            # Doubao is API-based, no initialization needed
            pass
        else:
            raise ValueError(
                f"Unknown method: {method}. "
                f"Available methods: 'teddy', 'sens', 'doubao-seed3d'"
            )

    def run(
        self,
        input: Union[Sketch, str, 'np.ndarray'],
        output_path: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Convert sketch to 3D model.

        Args:
            input: Input sketch, can be Sketch object, file path, or numpy array.
            output_path: Output file path (if None, will be auto-generated).
            **kwargs: Method-specific parameters.

        Returns:
            str: Output file path of the generated 3D model.

        Example:
            >>> from sketchkit.sketch2model import Modeler
            >>> modeler = Modeler(method='teddy')
            >>> output = modeler.run(sketch, output_path='output.obj')
        """
        if self.method == "teddy":
            return self._run_teddy(input, output_path, **kwargs)
        elif self.method == "sens":
            return self._run_sens(input, output_path, **kwargs)
        elif self.method in ["doubao-seed3d", "doubao"]:
            return self._run_doubao(input, output_path, **kwargs)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _run_teddy(
        self,
        sketch_input: Union[Sketch, str, 'np.ndarray'],
        output_path: Optional[str] = None,
        **kwargs
    ) -> str:
        """Run Teddy method."""
        import numpy as np
        from sketchkit.renderer.cairo_renderer import CairoRenderer
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

    def _run_sens(
        self,
        sketch_input: Union[Sketch, str, 'np.ndarray'],
        output_path: Optional[str] = None,
        **kwargs
    ) -> str:
        """Run SENS method."""
        # Pass device from instance
        if 'device' not in kwargs:
            kwargs['device'] = self.device
        return _sketch_to_3d_sens_impl(sketch_input, output_path, **kwargs)

    def _run_doubao(
        self,
        sketch_input: Union[Sketch, str, 'np.ndarray'],
        output_path: Optional[str] = None,
        **kwargs
    ) -> str:
        """Run Doubao-Seed3D method."""
        import numpy as np
        from sketchkit.renderer.cairo_renderer import CairoRenderer
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
            **kwargs
        )

