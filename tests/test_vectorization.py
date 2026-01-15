import sys
import os
sys.path.append(os.getcwd())
from sketchkit.vectorization import Vectorizer


if __name__ == '__main__':
    vectorizer = Vectorizer(method="LineDrawer")

    img_path = "0-0.png"
    import os
    import numpy as np
    from PIL import Image
    
    img_np = np.array(Image.open(img_path).convert("L"))
    
    sketch = vectorizer.run(img_np)
