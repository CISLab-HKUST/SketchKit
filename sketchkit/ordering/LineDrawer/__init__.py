import numpy as np
from .CNNVE import CNNVE
from sketchkit.ordering.orderer import Orderer
from sketchkit.core.sketch import Sketch


class LineDrawer(Orderer):
    """
    Implementation of LineDrawer: Stroke-Level Process Reconstruction of Complex Line Art Based on Human Perception.
    
    This method uses a reference video to recover the drawing order of a vector sketch by matching static vector strokes
    to the video frames where they first appear.
    """

    def __init__(self, method="CNNVE", device="cpu"):
        """
        Initialize the LineDrawer with the specified method and device.
        
        Args:
            method: Backend method. Currently only "CNNVE" is supported.
            device: Device to use for computation. "cuda" is recommended for better performance.
        
        Raises:
            ValueError: If an unsupported method is specified.
        """
        if method == "CNNVE":
            self.model = CNNVE(device=device)
        else:
            raise ValueError(f"Unsupported method: {method}")

    def run(self, img: np.ndarray, ref_video_path: str) -> Sketch:
        """
        Run LineDrawer to determine the drawing order using the reference video.

        Args:
            img: numpy array of the sketch image (grayscale).
            ref_video_path: Path to the reference video file.

        Returns:
            Sketch: A Sketch object with ordered paths, where the order of paths corresponds to the drawing sequence.
        """

        sketch = self.model.run(img, ref_video_path)

        # Returns a single Sketch object
        return sketch

__all__ = ["LineDrawer"]
