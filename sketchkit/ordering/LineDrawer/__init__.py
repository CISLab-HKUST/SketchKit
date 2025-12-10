import numpy as np
from .CNNVE import CNNVE
from sketchkit.ordering import Orderer
from sketchkit.core.sketch import Sketch


class LineDrawer(Orderer):
    """
    Wrapper class for LineDrawer animation, matching vectorization structure.
    """

    def __init__(self, method="CNNVE", device="cpu"):
        if method == "CNNVE":
            self.model = CNNVE(device=device)
        else:
            raise ValueError(f"Unsupported method: {method}")

    def run(self, img: np.ndarray, ref_video_path: str) -> Sketch:
        """
        Run LineDrawer animation.

        Args:
            img (numpy.ndarray): PIL image array
            ref_video (str): Path to reference video
            inner_frames (int): Ignored by LineDrawer

        Returns:
            list: List of frames
        """

        sketch = self.model.run(img, ref_video_path)

        # Returns a single Sketch object
        return sketch
