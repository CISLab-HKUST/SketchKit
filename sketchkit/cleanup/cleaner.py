import os
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

    def run(self, input: str):
        """
        Vectorize the input raster image or images in a folder.

        Args:
            input (str): a path to an input raster image or a folder with multiple images.

        Returns:
            sketch_list (list[numpy.ndimage] | numpy.ndimage): a list of sketch image in H*W in [0, 255] or an image when there is only one item.
        """
        image_paths = []
        sketch_list = []
        if os.path.isdir(input):
            for img_name in os.listdir(input):
                if "png" in img_name or "jpg" in img_name or "bmp" in img_name:
                    image_paths.append(os.path.join(input, img_name))
        else:
            image_paths.append(input)

        for image_path in image_paths:
            sketch = self.cleanup_method.run(image_path)
            sketch_list.append(sketch)

        if len(sketch_list) == 1:
            return sketch_list[0]
        else:
            return sketch_list