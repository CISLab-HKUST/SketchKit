from sketchkit.core.sketch import Sketch, Curve
from sketchkit.core.camera import Camera
from sketchkit.renderer3d.renderer3d import Renderer3D, Render3DOptions
import numpy as np
import cairocffi as cairo
import math
from PIL import Image
import random
from typing import List
from tqdm import tqdm
import os
from copy import deepcopy
from dataclasses import dataclass

from sketchkit.utils.file import download_with_wget, CISLAB_CDN

@dataclass
class NPRRenderOptions:
    canvas_size: tuple[int, int] = (512, 512)
    method: str = "SuggestiveContour"
    threshold: float = 0.005
    # obj_path: str = None
    


class NPRRenderer:
    def __init__(self, obj_path: str, render_options: NPRRenderOptions | None = None, root: str | None = None):
        self.render_options = render_options if render_options is not None else NPRRenderOptions()
        self.obj_path = obj_path
        
        # Check if renderer exists in the package directory
        package_renderer = os.path.join(os.path.dirname(__file__), "nogui_contour_renderer")
        if os.path.exists(package_renderer):
            self.renderer_path = package_renderer
            # Ensure executable permission
            st = os.stat(self.renderer_path)
            os.chmod(self.renderer_path, st.st_mode | 0o111)
            print(f"Using packaged renderer: {self.renderer_path}")
            return
        

    def _render(self, cameras: List[Camera], output_folder: str, render_options: NPRRenderOptions | dict | None = None,):
        """
        实现抽象方法，将3D草图渲染为2D图像
        
        args:
            sketch: 3D草图对象
            camera: 相机参数
            output_folder: 输出文件夹
            render_options: 渲染选项
            
        Returns:
            np.ndarray: 渲染后的图像数组
        """
        if render_options is None:
            opts = self.render_options
        elif isinstance(render_options, dict):
            opts = deepcopy(self.render_options)
            for k, v in render_options.items():
                if hasattr(opts, k):
                    setattr(opts, k, v)
        else:
            opts = render_options
        # 这里应该实现具体的渲染逻辑
        # 由于 NPRRenderer 原本是基于外部可执行文件的渲染器，
        # 我们需要将相机参数转换为变换矩阵
        
        # 从相机参数计算变换矩阵（这里需要根据具体需求实现）
        for i, camera in enumerate(cameras):
            transform = camera.extrinsics_for_opengl.reshape(-1)
            
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            output_path = os.path.join(output_folder, f"{i}.png")
        
            if opts.method == "SuggestiveContour":
                self.render_suggestive_countour(output_path, transform, opts.canvas_size, opts.threshold)
            elif opts.method == "RidgesAndValleys":
                self.render_ridges_and_valleys(output_path, transform, opts.canvas_size, opts.threshold)
            elif opts.method == "AppearantRidges":
                self.render_appearant_ridges(output_path, transform, opts.canvas_size, opts.threshold)
            else:
                raise ValueError(f"Unknown method: {opts.method}")

    def render_suggestive_countour(self, out_path: str, transform: np.ndarray, size: tuple[int, int] = (512, 512), sug_thresh: float = 0.005):
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
        os.system(f"{self.renderer_path} {self.obj_path} -o {out_path} -w {size[0]} -h {size[1]} -t {transform_str} -q --sugcontour --contour --sug-thresh {sug_thresh}")
    
    def render_ridges_and_valleys(self, out_path: str, transform: np.ndarray, size: tuple[int, int] = (512, 512), rv_thresh: float = 0.1):
        # 将transform转化为字符串
        transform_str = " ".join(map(str, transform.flatten()))
        # 调用命令行sketchkit/renderer/nogui_contour_renderer
        os.system(f"{self.renderer_path} {self.obj_path} -o {out_path} -w {size[0]} -h {size[1]} -t {transform_str} -q --ridges --valleys --contour --rv-thresh {rv_thresh}")

    def render_appearant_ridges(self, out_path: str, transform: np.ndarray, size: tuple[int, int] = (512, 512), ar_thresh: float = 0.1):
        # 将transform转化为字符串
        transform_str = " ".join(map(str, transform.flatten()))
        # 调用命令行sketchkit/renderer/nogui_contour_renderer
        os.system(f"{self.renderer_path} {self.obj_path} -o {out_path} -w {size[0]} -h {size[1]} -t {transform_str} -q --apparent --contour --ar-thresh {ar_thresh}")


if __name__ == "__main__":
    from sketchkit.core.camera import Camera
    sug_opt = NPRRenderOptions(method="SuggestiveContour", threshold=0.005)
    rv_opt = NPRRenderOptions(method="RidgesAndValleys", threshold=0.1)
    ar_opt = NPRRenderOptions(method="AppearantRidges", threshold=0.1)

    renderer = NPRRenderer(obj_path="media/rapid.obj")
    camera = Camera()
    camera.set_look_at([-462, -71, 70])
    camera.set_xyz(np.array([-461, -71, 70]))
    

    # 创建测试草图
    
    
    # 使用新的render方法
    renderer._render([camera], "media/output/suggestive", sug_opt)
    renderer._render([camera], "media/output/ridges_and_valleys", rv_opt)

    renderer._render([camera], "media/output/appearant_ridges", ar_opt)
