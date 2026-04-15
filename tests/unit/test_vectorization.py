import os
from sketchkit.vectorization import Vectorizer
import numpy as np
from PIL import Image

if __name__ == "__main__":
    vectorizer = Vectorizer(method="LineDrawer")

    img_path = "test_images/a1-2.png"
    img_np = np.array(Image.open(img_path).convert("L"))
    sketch = vectorizer.run(img_np)

    os.makedirs("outputs/vectorization", exist_ok=True)
    output_path = "outputs/vectorization"
    sketch.to_svg(filename=os.path.join(output_path, "pikachu.svg"))
