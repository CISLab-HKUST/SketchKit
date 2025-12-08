import os
import numpy as np
import cv2
from PIL import Image

from sketchkit.ordering.LineDrawer import LineDrawer

class Orderer:
    """
    A class for sketch ordering.

    Attributes:
        method (str): The ordering method that is used.
        device (str): The device that is used.
    """

    def __init__(self, method: str = "LineDrawer", device: str = "cuda"):
        """
        Initializes the Orderer with the name of the method.

        Args:
            method (str, optional): The ordering method that is used.
            device (str, optional): The device that is used.
        """
        self.method = method
        self.device = device

        if method == "LineDrawer":
            self.order_method = LineDrawer(device=device)
        else:
            raise Exception("Please add implementations.")

    def run(
        self,
        input_image_list: list,
        inner_frames: int = 8,
        **kwargs
    ) -> list[np.ndarray]:
        """
        Generate ordered sketch.

        Args:
            input_image_list (list): list of inputs (e.g. [sketch_path, video_path]).
            inner_frames (int): number of frames (might be unused by some methods).

        Returns:
            list[numpy.ndarray]: a list of image frames or ordered points.
        """
        return self.order_method.run(input_image_list, inner_frames)
