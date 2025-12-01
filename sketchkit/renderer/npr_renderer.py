from sketchkit.core.sketch import Sketch, Curve
import numpy as np
import cairocffi as cairo
import math
from PIL import Image
import random
from typing import List
from tqdm import tqdm
import os

from sketchkit.utils.file import download_with_wget, CISLAB_CDN

class NPRRenderer:
    def __init__(self, canvas_size: int = 512, obj_path: str = None, root: str = None):
        self.canvas_size = canvas_size
        self.obj_path = obj_path
        if root is None:
            root = os.path.join(
                os.environ.get("HOME", "/"),
                ".cache/sketchkit/bin",
                self.__class__.__name__,
            )
            # root = "."
        if isinstance(root, str):
            root = os.path.expanduser(root)
        os.makedirs(root, exist_ok=True)
        self.root = root

        print(f"Renderer root: {self.root}")
        self.renderer_path = os.path.join(self.root, "nogui_contour_renderer")

        try:
            url = f"{CISLAB_CDN}/bin/SuggestiveContour/nogui_contour_renderer"
            print(f"Downloading renderer from CISLAB CDN: {url}")
            download_with_wget(url, file_path=self.renderer_path, pg_bar=True)
            os.chmod(self.renderer_path, 0o755)
        except Exception:
            print("CISLAB mirror unavailable.")

    def render_suggestive_countour(self, out_path: str, transform: np.ndarray, sug_thresh: float = 0.005):
        """
        Render a sketch composed of Bézier curves using a given texture.

        args:
            out_path: str
                Path to save the output image.
            transform: np.ndarray
                Transform matrix.
            canvas_size: int, optional
                Canvas size. Default is 512.
            sug_thresh: float, optional
                Suggestive threshold. Default is 0.005.

        """
        # 将transform转化为字符串
        transform_str = " ".join(map(str, transform.flatten()))
        # 调用命令行sketchkit/renderer/nogui_contour_renderer
        os.system(f"{self.renderer_path} {self.obj_path} -o {out_path} -w {self.canvas_size} -h {self.canvas_size} -t {transform_str} -q --sugcontour --contour --sug-thresh {sug_thresh}")
    
    def render_ridges_and_valleys(self, out_path: str, transform: np.ndarray, rv_thresh: float = 0.1):
        # 将transform转化为字符串
        transform_str = " ".join(map(str, transform.flatten()))
        # 调用命令行sketchkit/renderer/nogui_contour_renderer
        os.system(f"{self.renderer_path} {self.obj_path} -o {out_path} -w {self.canvas_size} -h {self.canvas_size} -t {transform_str} -q --ridges --valleys --contour --rv-thresh {rv_thresh}")

    def render_appearant_ridges(self, out_path: str, transform: np.ndarray, ar_thresh: float = 0.1):
        # 将transform转化为字符串
        transform_str = " ".join(map(str, transform.flatten()))
        # 调用命令行sketchkit/renderer/nogui_contour_renderer
        os.system(f"{self.renderer_path} {self.obj_path} -o {out_path} -w {self.canvas_size} -h {self.canvas_size} -t {transform_str} -q --apparent --contour --ar-thresh {ar_thresh}")


if __name__ == "__main__":
    renderer = NPRRenderer(obj_path="media/rapid.obj")
    renderer.render_suggestive_countour("test22.ppm", np.array([0, -1, 0, 70, 0, 0, 1, -70, -1, 0, 0, -461, 0, 0, 0, 1]))
    renderer.render_ridges_and_valleys("test23.ppm", np.array([0, -1, 0, 70, 0, 0, 1, -70, -1, 0, 0, -461, 0, 0, 0, 1]))
    renderer.render_appearant_ridges("test24.ppm", np.array([0, -1, 0, 70, 0, 0, 1, -70, -1, 0, 0, -461, 0, 0, 0, 1]))