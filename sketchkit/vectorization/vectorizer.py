import os
import numpy as np
from sketchkit.core.sketch import Sketch
from sketchkit.vectorization.DeepVecSIG24 import DeepVecSIG24
from sketchkit.vectorization.LineDrawer import LineDrawer

class Vectorizer:
    """
    A class for image vectorization.

    This class provides functionality to convert a raster image to a vector image.

    Attributes:
        method (str): The vectorization method that is used.
        device (str): The device that is used.
    """
    def __init__(self, method: str="DeepVecSIG24", device: str="cuda"):
        """
        Initializes the Vectorizer with ....

        Args:
            method (str, optional): The vectorization method that is used.
            device (str, optional): The device that is used.
        """
        self.method = method
        self.device = device

        if method == "DeepVecSIG24":
            self.vectorize_method = DeepVecSIG24(device)
        elif method in {"LineDrawer", "CNNVE"}:
            self.vectorize_method = LineDrawer(device=device, method=method)
        else:
            raise Exception("Please add implementations.")

    def run(
        self,
        input: np.ndarray
    ) -> Sketch:
        """
        Vectorize the input raster image.

        Args:
            input (numpy.ndarray): input raster image in shape (H, W), with values in [0, 255].

        Returns:
            sketch (Sketch): a Sketch instance.
        """
        sketch = self.vectorize_method.run(input)
        if type(sketch) is not Sketch:
            raise Exception("Should return a Sketch instance. But with %s now." % str(type(sketch)))
        return sketch