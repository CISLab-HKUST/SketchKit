import os
import numpy as np
from sketchkit.cleanup.MasterSketch import MasterSketch

class Cleaner:
    """
    A class for rough sketch cleanup.

    This class provides functionality to convert a raster rough sketch to a clean sketch.

    Attributes:
        method (str): The vectorization method that is used.
        device (str): The device that is used.
    """
    def __init__(self, method: str="MasterSketch", device: str="cuda"):
        """
        Initializes the Cleaner with ....

        Args:
            method (str, optional): The vectorization method that is used.
            device (str, optional): The device that is used.
        """
        self.method = method
        self.device = device

        if method == "MasterSketch":
            self.cleanup_method = MasterSketch(device)
        else:
            raise Exception("Please add implementations.")

    def run(
        self,
        input: np.ndarray,
        output_folder: str = "cleanup",
    ) -> np.ndarray:
        """
        Simplify the input raster image.

        Args:
            input (numpy.ndarray): input raster image in shape (H, W), with values in [0, 255].

        Returns:
            sketch (numpy.ndarray): an image in H*W in [0, 255].
        """
        sketch = self.cleanup_method.run(input, out_folder=output_folder)
        if type(sketch) is not np.ndarray:
            raise Exception("Should return a numpy array. But with %s now." % str(type(sketch)))
        return sketch