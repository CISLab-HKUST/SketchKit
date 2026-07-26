"""Unified interface for image-to-sketch methods."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from sketchkit.core import Sketch

ImageInput = Union[Image.Image, np.ndarray, str]
Image2SketchResult = Union[Sketch, Image.Image, list[Sketch], list[Image.Image]]
ImageSize = Optional[Union[int, Tuple[int, int]]]


class SketchGenerator:
    """Generate sketches with one of the supported image-to-sketch methods."""

    def __init__(self, method: str = "SwiftSketch", **kwargs: Any) -> None:
        self.method = method
        self.generator_method = self.create(method, **kwargs)

    @staticmethod
    def create(method: str = "SwiftSketch", **kwargs: Any) -> Any:
        """Create the selected backend."""
        method_key = method.replace("-", "").replace("_", "").casefold()

        if method_key == "hed":
            from sketchkit.image2sketch.hed import HEDModel

            return HEDModel(**kwargs)
        if method_key == "photosketch":
            from sketchkit.image2sketch.photosketch import PhotoSketchModel

            return PhotoSketchModel(**kwargs)
        if method_key == "swiftsketch":
            from sketchkit.image2sketch.swiftsketch import SwiftSketchModel

            return SwiftSketchModel(**kwargs)
        if method_key == "anilines":
            from sketchkit.image2sketch.anilines import AniLinesModel

            return AniLinesModel(**kwargs)

        raise ValueError(
            f"Unknown image2sketch method: {method!r}. "
            "Available methods: HED, PhotoSketch, SwiftSketch, AniLines."
        )

    def run(
        self,
        input: Union[ImageInput, Sequence[ImageInput]],
        size: ImageSize = None,
    ) -> Image2SketchResult:
        """Generate a sketch from the supplied image input."""
        return self.generator_method.generate(input, size=size)
