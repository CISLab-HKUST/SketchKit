from sketchkit.core.sketch import Sketch, Curve
from sketchkit.core.camera import Camera
import numpy as np
import math
from PIL import Image
from typing import List
import os
from copy import deepcopy
from dataclasses import dataclass
import tempfile
import subprocess
import platform


@dataclass
class NPRRenderOptions:
    canvas_size: tuple[int, int] = (512, 512)
    method: str = "SuggestiveContour"
    threshold: float = 0.005


class NPRRenderer:
    def __init__(
        self,
        obj_path: str,
        render_options: NPRRenderOptions | None = None,
    ):
        self.render_options = (
            render_options if render_options is not None else NPRRenderOptions()
        )
        # 将obj_path中的名加上normalized前缀
        obj_dir, obj_name = os.path.split(obj_path)
        normalized_obj_name = "normalized_" + obj_name
        self.obj_path = os.path.join(obj_dir, normalized_obj_name)
        self.normalize_obj(obj_path, self.obj_path)

        binary_name = "nogui_contour_renderer" + (".exe" if os.name == "nt" else "")
        package_dir = os.path.dirname(__file__)
        candidates = [
            os.path.join(package_dir, "_native", binary_name),
            os.path.join(package_dir, binary_name),
        ]
        for candidate in candidates:
            if not os.path.isfile(candidate):
                continue
            with open(candidate, "rb") as binary:
                magic = binary.read(4)
            # The repository contains a Linux executable for legacy installs.
            if platform.system() != "Linux" and magic == b"\x7fELF":
                continue
            self.renderer_path = candidate
            break
        else:
            raise RuntimeError(
                "No compatible RTSC renderer is installed. On macOS run "
                "`brew install pkgconf glfw glew`, then "
                "`uv sync --reinstall-package sketchkit` to build it."
            )

    def _run_renderer(self, out_path, transform, size, flags):
        command = [
            self.renderer_path, self.obj_path, "-o", str(out_path),
            "-w", str(size[0]), "-h", str(size[1]), "-t",
            *map(str, np.asarray(transform).flatten()), "-q", *flags,
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()
            raise RuntimeError(
                f"RTSC rendering failed (exit {exc.returncode}): {detail}"
            ) from exc
        except OSError as exc:
            raise RuntimeError(f"Cannot execute RTSC renderer: {exc}") from exc
        if not os.path.isfile(out_path):
            raise RuntimeError("RTSC exited successfully but produced no image.")

    def normalize_obj(self, input_path: str, output_path: str):
        # 归一化OBJ模型到单位立方体内
        # trimesh读取obj
        import trimesh
        mesh = trimesh.load(input_path)
        vertices = mesh.vertices
        vertices -= np.mean(vertices, axis=0)
        # 归一化到单位立方体内
        max_dim = np.max(np.linalg.norm(vertices, axis=1))
        vertices /= max_dim
        # 中心到原点
        # 写入新的obj文件
        mesh.vertices = vertices
        mesh.export(output_path)

    def _render(
        self,
        cameras: List[Camera],
        render_options: NPRRenderOptions | dict | None = None,
    ):
        """
        实现抽象方法，将3D草图渲染为2D图像

        args:
            sketch: 3D草图对象
            camera: 相机参数
            output_folder: 输出文件夹
            render_options: 渲染选项

        Returns:
            List[Image.Image]: 渲染后的图像列表
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

        images = []

        # 使用临时目录存储渲染结果
        with tempfile.TemporaryDirectory() as temp_dir:
            # 从相机参数计算变换矩阵（这里需要根据具体需求实现）
            for i, camera in enumerate(cameras):

                extrinsics = camera.extrinsics
                # print("Camera extrinsics:\n", extrinsics)

                T = np.array(
                    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
                )  # 转换坐标系从RFU到RDF

                transform = T @ extrinsics

                # print("Transform matrix:\n", transform)
                transform = transform.reshape(-1)

                output_path = os.path.join(temp_dir, f"{i}.ppm")

                if opts.method == "SuggestiveContour":
                    self.render_suggestive_countour(
                        output_path, transform, opts.canvas_size, opts.threshold
                    )
                elif opts.method == "RidgesAndValleys":
                    self.render_ridges_and_valleys(
                        output_path, transform, opts.canvas_size, opts.threshold
                    )
                elif opts.method == "AppearantRidges":
                    self.render_appearant_ridges(
                        output_path, transform, opts.canvas_size, opts.threshold
                    )
                else:
                    raise ValueError(f"Unknown method: {opts.method}")

                # 读取渲染结果
                if os.path.exists(output_path):
                    img = Image.open(output_path)
                    img.load()
                    images.append(img)

        return images

    def render_suggestive_countour(
        self,
        out_path: str,
        transform: np.ndarray,
        size: tuple[int, int] = (512, 512),
        sug_thresh: float = 0.005,
    ):
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
        self._run_renderer(out_path, transform, size,
                           ["--sugcontour", "--contour", "--sug-thresh", str(sug_thresh)])

    def render_ridges_and_valleys(
        self,
        out_path: str,
        transform: np.ndarray,
        size: tuple[int, int] = (512, 512),
        rv_thresh: float = 0.1,
    ):
        self._run_renderer(out_path, transform, size,
                           ["--ridges", "--valleys", "--contour", "--rv-thresh", str(rv_thresh)])

    def render_appearant_ridges(
        self,
        out_path: str,
        transform: np.ndarray,
        size: tuple[int, int] = (512, 512),
        ar_thresh: float = 0.1,
    ):
        self._run_renderer(out_path, transform, size,
                           ["--apparent", "--contour", "--ar-thresh", str(ar_thresh)])


if __name__ == "__main__":
    from sketchkit.core.camera import Camera
    sug_opt = NPRRenderOptions(method="SuggestiveContour", threshold=0.005)
    rv_opt = NPRRenderOptions(method="RidgesAndValleys", threshold=0.1)
    ar_opt = NPRRenderOptions(method="AppearantRidges", threshold=0.1)

    renderer = NPRRenderer(obj_path="media/rapid.obj")
    # camera = Camera()
    # camera.set_look_at([0, 0, 0])
    # camera.set_xyz([0, 2.55, 0])
    # 创建测试草图
    cameras = []
    r = 2.5
    for angle in range(0, 360, 30):
        rad = math.radians(angle)
        camera = Camera()
        camera.set_look_at([0, 0, 0])
        camera.set_xyz([2.5 * math.cos(rad), 2.5 * math.sin(rad), 0])
        cameras.append(camera)

    from sketchkit.utils.file import save_seq_gif

    # 使用新的render方法
    images = renderer._render(cameras, sug_opt)
    save_seq_gif(images, "npr_rendering_suggestive_contour.gif")
    images = renderer._render(cameras, rv_opt)
    save_seq_gif(images, "npr_rendering_ridges_and_valleys.gif")
    images = renderer._render(cameras, ar_opt)
    save_seq_gif(images, "npr_rendering_appearant_ridges.gif")
