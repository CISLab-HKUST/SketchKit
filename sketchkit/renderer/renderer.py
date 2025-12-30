import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from sketchkit.core.sketch import Sketch

@dataclass
class RenderOptions:
    canvas_size: tuple[int, int] = (512, 512)
    background_color: tuple[int, int, int] = (255, 255, 255)
    stroke_color: tuple[int, int, int] = (0, 0, 0)
    stroke_width: float = None  # default should be None, which might be the best for the implementations in sketch3d_renderer.py


class Renderer(ABC):
    render_options = RenderOptions()

    def render(
        self, sketch: Sketch, render_options: RenderOptions | None = None
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
        return self._render(sketch, _render_options)

    @abstractmethod
    def _render(self, *args, **kwargs) -> np.ndarray:
        pass
