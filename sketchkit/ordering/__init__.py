from __future__ import annotations  # 如果你用的是 Python 3.11 以下，建议加上这行

from abc import ABC, abstractmethod
from typing import overload, Literal

from sketchkit.core.sketch import Sketch


class Orderer(ABC):
    """
    A class for sketch ordering.

    Attributes:
        method (str): The ordering method that is used.
        device (str): The device that is used.
    """

    @staticmethod
    @overload
    def create(
        method: Literal["LineDrawer"],
        device: str = "cuda",
    ) -> LineDrawer: ...

    @staticmethod
    @overload
    def create(
        method: Literal["Fu"],
        device: str = "cuda",
    ) -> "Orderer": ...

    @staticmethod
    def create(
        method: str = "LineDrawer",
        device: str = "cuda",
    ) -> "Orderer":
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if method == "LineDrawer":
            from sketchkit.ordering.LineDrawer import LineDrawer

            return LineDrawer(device=device)
        else:
            raise ValueError(f"Unknown method: {method!r}. Please add implementations.")

    @abstractmethod
    def run(self, *args, **kwargs) -> Sketch: ...
