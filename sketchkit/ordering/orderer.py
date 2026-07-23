from typing import overload, Literal, TYPE_CHECKING

from sketchkit.core.sketch import Sketch

if TYPE_CHECKING:
    from sketchkit.ordering.LineDrawer import LineDrawer

class Orderer:
    """
    Main interface for sketch ordering. Determines the stroke drawing sequence for a static sketch.
    
    Attributes:
        method: The ordering method instance (e.g., LineDrawer or Fu).
    """
    
    def __init__(self, method: str = "LineDrawer", device: str = "cuda"):
        """
        Initialize the Orderer with the specified method and device.
        
        Args:
            method: Backend name. Supported values are "Fu", "LineDrawer" (case-sensitive).
            device: Device to use for computation. For Fu, "cpu" is sufficient. For LineDrawer, "cuda" is recommended.
        """
        super().__init__()
        self.method = self.create(method, device)
    
    def run(self, *args, **kwargs) -> Sketch:
        """
        Run the ordering process using the selected method.
        
        Args:
            *args: Arguments passed to the selected method's run method.
            **kwargs: Keyword arguments passed to the selected method's run method.
            
            For Fu method:
                sketch: Sketch object containing the static sketch to animate.
            
            For LineDrawer method:
                img: numpy array of the sketch image (grayscale).
                ref_video_path: path to the reference video file.
        
        Returns:
            Sketch: A Sketch object with ordered paths, where the order of paths corresponds to the drawing sequence.
        """
        return self.method.run(*args, **kwargs)
    
    @staticmethod
    @overload
    def create(
        method: Literal["LineDrawer"],
        device: str = "cuda",
    ) -> "LineDrawer": ...

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
        Create an instance of the specified ordering method.
        
        Args:
            method: Backend name. Supported values are "Fu", "LineDrawer" (case-sensitive).
            device: Device to use for computation. For Fu, "cpu" is sufficient. For LineDrawer, "cuda" is recommended.
        
        Returns:
            An instance of the selected ordering method.
        
        Raises:
            ValueError: If an unknown method is specified.
        """
        if method == "LineDrawer":
            from sketchkit.ordering.LineDrawer import LineDrawer

            return LineDrawer(device=device)
        elif method == "Fu":
            from sketchkit.ordering.Fu import AnimatedDrawer

            return AnimatedDrawer()
        else:
            raise ValueError(f"Unknown method: {method!r}. Please add implementations.")