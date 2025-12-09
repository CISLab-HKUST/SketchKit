"""
LineDrawer wrapper for animation module.
"""

import cv2
from .CNNVE import CNNVE

class LineDrawer:
    """
    Wrapper class for LineDrawer animation, matching vectorization structure.
    """
    def __init__(self, method="CNNVE", device="cpu"):
        if method == "CNNVE":
            self.model = CNNVE(device=device)
        else:
            raise ValueError(f"Unsupported method: {method}")

    def run(self, input_image_list, inner_frames):
        """
        Run LineDrawer animation.
        
        Args:
            input_image_list (list): [sketch_path, video_path]
            inner_frames (int): Ignored by LineDrawer
            
        Returns:
            list: List of frames
        """
        input_0 = input_image_list[0]
        input_1 = input_image_list[1]
        
        img = cv2.imread(input_0, cv2.IMREAD_GRAYSCALE)
        
        frames = self.model.run(img, input_1)
        
        # frames is now a list of Sketch objects
        return frames
