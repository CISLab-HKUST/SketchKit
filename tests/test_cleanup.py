import numpy as np
from PIL import Image

from sketchkit.cleanup import Cleaner


if __name__ == "__main__":
    cleaner = Cleaner(method="MasterSketch")

    img_path = "01.png"
    img_np = np.array(Image.open(img_path).convert("L"))
    sketch = cleaner.run(img_np)
