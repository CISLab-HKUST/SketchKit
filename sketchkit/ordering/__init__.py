
        
          


    
from __future__ import annotations  # 如果你用的是 Python 3.11 以下，建议加上这行

from abc import ABC, abstractmethod
from typing import overload, Literal

from sketchkit.core.sketch import Sketch


class Orderer():
    def __init__(self, method: str = "LineDrawer", device: str = "cuda"):
        super().__init__()
        self.method = OrdererMethod.create(method, device)
    
    def run(self, *args, **kwargs) -> Sketch:
        return self.method.run(*args, **kwargs)

class OrdererMethod(ABC):
    """
    A class for sketch ordering.

    Attributes: 
        method (str): The ordering method that is used.
        device (str): The device that is used.
    """
    def __init__(self, method: str = "LineDrawer", device: str = "cuda"):
        super().__init__()
        self.method = self.create(method, device)
    
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
        device: str = "cpu",
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
        elif method == "Fu":
            from sketchkit.ordering.Fu import AnimatedDrawer

            return AnimatedDrawer()
        else:
            raise ValueError(f"Unknown method: {method!r}. Please add implementations.")

    @abstractmethod
    def run(self, *args, **kwargs) -> Sketch: ...
