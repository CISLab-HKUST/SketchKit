# NPR->vec->Ciallo->image:
#  image+video external APIs



from pathlib import Path

from sketchkit.vectorization import Vectorizer


if __name__ == "__main__":
    vectorizer = Vectorizer(method="LineDrawer")

    img_path = Path(__file__).resolve().parent.parent / "data" / "ordering" / "input.png"
    import numpy as np
    from PIL import Image
    
    img_np = np.array(Image.open(img_path).convert("L"))
    
    sketch = vectorizer.run(img_np)
