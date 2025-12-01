from .CNNVE import CNNVE


class LineDrawer:
    """
    Native LineDrawer integration that mirrors the structure of other vectorizers.
    """

    def __init__(self, method: str = "CNNVE", device: str = "cuda"):
        self.method = method
        self.device = device

        if method in {"LineDrawer", "CNNVE"}:
            self.vectorize_method = CNNVE(device=device)
        else:
            raise ValueError(f"Unsupported LineDrawer method: {method}")

    def run(self, input_path: str):
        """
        Vectorize the provided image or directory using the configured method.
        """
        return self.vectorize_method.run(input_path)

