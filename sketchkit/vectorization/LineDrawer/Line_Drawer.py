import os
import cv2
import numpy as np
import tempfile
import shutil
import gzip
import pickle
from PIL import Image
from .CNNVE import CNNVE
from sketchkit.core.sketch import Sketch, Path as SketchPath, Curve, Point


class LineDrawer:
    """
    Native LineDrawer integration that mirrors the structure of other vectorizers.
    """

    def __init__(self, method: str = "CNNVE", device: str = "cuda"):
        self.method = method
        self.device = device

        if method in {"LineDrawer", "CNNVE"}:
            self.vectorize_method = CNNVE(device=device)
        else:
            raise ValueError(f"Unsupported LineDrawer method: {method}")

    def run(self, img: np.ndarray) -> Sketch:
        """
        Vectorize the provided numpy image.

        Args:
            img (np.ndarray): Input image as a 2D numpy array (grayscale).

        Returns:
            Sketch: The vectorized sketch.
        """
        # Create a temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_img_path = os.path.join(temp_dir, "input.png")
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            
            Image.fromarray(img).save(temp_img_path)

            self.vectorize_method.run(temp_img_path)

            graph_merge_path = os.path.join(temp_dir, "input.graph_merge")

            if not os.path.exists(graph_merge_path):
                 raise RuntimeError(f"CNNVE failed to generate graph file at {graph_merge_path}")

            with gzip.open(graph_merge_path, 'rb') as f:
                merge_dic = pickle.load(f)

            h, w = img.shape[:2]
            sketch = self._convert_to_sketch(merge_dic, h, w)
            return sketch

    def _convert_to_sketch(self, merge_dic: dict, height: int, width: int) -> Sketch:
        """
        Convert the CNNVE output dictionary to a Sketch object.
        """
        simpleGraphs = merge_dic.get('simpleGraphs', [])
        paths = []

        for graph_dict in simpleGraphs:
            g = graph_dict['graph']
            for u, v, data in g.edges(data=True):
                edge_obj = data.get('object')
                if edge_obj and hasattr(edge_obj, 'pixels'):
                    pixels = edge_obj.pixels
                    full_pixels = [u.point] + pixels + [v.point]
                    
                    points_xy = np.array([[p[1], p[0]] for p in full_pixels], dtype=np.float32)
                    
                    if len(points_xy) < 2:
                        continue
                        
                    stroke_list = []
                    for i in range(1, len(points_xy)):
                        p_start = points_xy[i-1]
                        p_end = points_xy[i]
                        line = np.stack([p_start, p_end], axis=0)
                        cubic = Curve.from_line(line)
                        stroke_list.append(cubic)
                    
                    if stroke_list:
                        paths.append(SketchPath(stroke_list))

        return Sketch(height, width, paths)
