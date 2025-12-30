from copy import deepcopy
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from PIL import Image


from sketchkit.core.sketch import Sketch
from sketchkit.core.camera import Camera


@dataclass
class RenderOptions:
    pass


class Renderer(ABC):
    render_options = RenderOptions()

    def render(
        self, sketch: Sketch, render_options: RenderOptions | dict | None = None
    ) -> Image.Image:
        if render_options is None:
            _render_options = self.render_options
        elif type(render_options) is dict:
            _render_options = deepcopy(self.render_options)
            for k, v in render_options.items():
                if hasattr(_render_options, k):
                    setattr(_render_options, k, v)
        elif type(render_options) is render_options:
            _render_options = render_options
        else:
            raise ValueError("Invalid render options.")

        return self._render(sketch, _render_options)

    @abstractmethod
    def _render(self, *args, **kwargs) -> Image.Image:
        pass


@dataclass
class Render3DOptions:
    canvas_size: tuple[int, int] = (512, 512)
    background_color: tuple[int, int, int] = (255, 255, 255)
    stroke_color: tuple[int, int, int] = (0, 0, 0)
    stroke_width: float = 1.0


class Renderer3D(ABC):
    render_options = Render3DOptions()

    def render(
        self,
        sketch: Sketch,
        camera: Camera,
        render_options: RenderOptions | dict | None = None,
    ) -> np.ndarray:

        if render_options is dict:
            _render_options = deepcopy(self.render_options)
            for k, v in source.items():
                if hasattr(_render_options, k):
                    setattr(_render_options, k, v)
        elif render_options is not None:
            _render_options = render_options
        else:
            _render_options = self.render_options

        return self._render(sketch, camera, _render_options)

    @abstractmethod
    def _render(self, *args, **kwargs) -> np.ndarray:
        pass
