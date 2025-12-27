from sketchkit.core.sketch import Sketch, Curve
from sketchkit.core.camera import Camera
from sketchkit.renderer.renderer import Renderer3D, Render3DOptions
import numpy as np
import cairocffi as cairo
import math
from PIL import Image
import random
from typing import List
from tqdm import tqdm
import os
from copy import deepcopy

from sketchkit.utils.file import download_with_wget, CISLAB_CDN


class NPRRenderer(Renderer3D):
    def __init__(self, canvas_size: int = 512, obj_path: str = None, root: str = None):
        super().__init__()
        self.obj_path = obj_path
        
        # 设置渲染选项
        self.render_options = Render3DOptions(
            canvas_size=(canvas_size, canvas_size),
            background_color=(255, 255, 255),
            stroke_color=(0, 0, 0),
            stroke_width=1.0
        )
        
        # 设置缓存目录
        if root is None:
            root = os.path.join(
                os.environ.get("HOME", "/"),
                ".cache/sketchkit/bin",
                self.__class__.__name__,
            )
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

    def _render(self, camera: Camera, output_path: str, method: str = "SuggestiveContour", threshold: float = 0.005) -> np.ndarray:
        """
        实现抽象方法，将3D草图渲染为2D图像
        
        Args:
            sketch: 3D草图对象
            camera: 相机参数
            render_options: 渲染选项
            
        Returns:
            np.ndarray: 渲染后的图像数组
        """
        # 这里应该实现具体的渲染逻辑
        # 由于 NPRRenderer 原本是基于外部可执行文件的渲染器，
        # 我们需要将相机参数转换为变换矩阵
        
        # 从相机参数计算变换矩阵（这里需要根据具体需求实现）
        transform = camera.extrinsics
        
        if method == "SuggestiveContour":
            self.render_suggestive_countour(output_path, transform, threshold)
        elif method == "RidgesAndValleys":
            self.render_ridges_and_valleys(output_path, transform, threshold)
        elif method == "AppearantRidges":
            self.render_appearant_ridges(output_path, transform, threshold)
        else:
            raise ValueError(f"Unknown method: {method}")

    def render_suggestive_countour(self, out_path: str, transform: np.ndarray, sug_thresh: float = 0.005):
        """
        Render a sketch composed of Bézier curves using a given texture.

        args:
            out_path: str
                Path to save the output image.
            transform: np.ndarray
                Transform matrix.
            sug_thresh: float, optional
                Suggestive threshold. Default is 0.005.

        """
        # 将transform转化为字符串
        transform_str = " ".join(map(str, transform.flatten()))
        # 调用命令行sketchkit/renderer/nogui_contour_renderer
        os.system(f"{self.renderer_path} {self.obj_path} -o {out_path} -w {self.render_options.canvas_size[0]} -h {self.render_options.canvas_size[1]} -t {transform_str} -q --sugcontour --contour --sug-thresh {sug_thresh}")
    
    def render_ridges_and_valleys(self, out_path: str, transform: np.ndarray, rv_thresh: float = 0.1):
        # 将transform转化为字符串
        transform_str = " ".join(map(str, transform.flatten()))
        # 调用命令行sketchkit/renderer/nogui_contour_renderer
        os.system(f"{self.renderer_path} {self.obj_path} -o {out_path} -w {self.render_options.canvas_size[0]} -h {self.render_options.canvas_size[1]} -t {transform_str} -q --ridges --valleys --contour --rv-thresh {rv_thresh}")

    def render_appearant_ridges(self, out_path: str, transform: np.ndarray, ar_thresh: float = 0.1):
        # 将transform转化为字符串
        transform_str = " ".join(map(str, transform.flatten()))
        # 调用命令行sketchkit/renderer/nogui_contour_renderer
        os.system(f"{self.renderer_path} {self.obj_path} -o {out_path} -w {self.render_options.canvas_size[0]} -h {self.render_options.canvas_size[1]} -t {transform_str} -q --apparent --contour --ar-thresh {ar_thresh}")


if __name__ == "__main__":
    from sketchkit.core.camera import Camera
    
    renderer = NPRRenderer(obj_path="media/rapid.obj")
    camera = Camera()
    #np.array([0, -1, 0, 70, 0, 0, 1, -70, -1, 0, 0, -461, 0, 0, 0, 1])
    # 创建测试草图
    sketch = Sketch()
    
    # 使用新的render方法
    img_array = renderer.render(sketch, camera)
    print(f"Rendered image shape: {img_array.shape}")