import os
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

    def run(self, input: str):
        """
        Vectorize the input raster image or images in a folder.

        Args:
            input (str): a path to an input raster image or a folder with multiple images.

        Returns:
            sketch_list (list[Sketch] | Sketch): a list of Sketch or a Sketch when there is only one item.
        """
        image_paths = []
        sketch_list = []
        if os.path.isdir(input):
            for img_name in os.listdir(input):
                lower = img_name.lower()
                if lower.endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    image_paths.append(os.path.join(input, img_name))
        else:
            image_paths.append(input)

        for image_path in image_paths:
            sketch = self.vectorize_method.run(image_path)
            sketch_list.append(sketch)

        if len(sketch_list) == 1:
            return sketch_list[0]
        else:
            return sketch_list
