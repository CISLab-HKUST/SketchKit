from copy import deepcopy
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from PIL import Image


from sketchkit.core.sketch import Sketch


@dataclass
class RenderOptions:
    canvas_size: tuple[int, int] | None = None


class Renderer(ABC):
    render_options = RenderOptions()

    def render(
        self, sketch: Sketch, render_options: RenderOptions | dict | None = None
    ) -> Image.Image:
        if render_options is None:
            _render_options = deepcopy(self.render_options)
        elif isinstance(render_options, dict):
            _render_options = deepcopy(self.render_options)
            for k, v in render_options.items():
                if hasattr(_render_options, k):
                    setattr(_render_options, k, v)
        elif isinstance(render_options, RenderOptions):
            _render_options = render_options
        else:
            raise ValueError("Invalid render options.")

        if _render_options.canvas_size is None:
            _render_options.canvas_size = (int(sketch.width), int(sketch.height))

        return self._render(sketch, _render_options)

    @abstractmethod
    def _render(self, *args, **kwargs) -> Image.Image:
        pass
