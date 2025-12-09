import sys
import os
sys.path.append(os.getcwd())

import numpy as np
from PIL import Image
from sketchkit.ordering import Orderer
from sketchkit.core.sketch import Sketch


if __name__ == '__main__':
    orderer = Orderer(method="LineDrawer")

    # Load image
    img_path = "a1-2.png"
    img_np = np.array(Image.open(img_path).convert("L"))
    
    # Video path
    video_path = "a1-2.png.mp4"
    
    if not os.path.exists(video_path):
        print(f"⚠ Warning: Video file {video_path} not found")
        print("Test will fail - please provide a video file")
    else:
        print(f"Running ordering with image path: {img_path}")
        print(f"Video path: {video_path}")
        
        # Orderer expects paths
        sketch = orderer.run([img_path, video_path])
        
        print(f"\n✓ Ordered sketch generated.")
        
        # Verify it is a Sketch object
        if isinstance(sketch, Sketch):
            print(f"  ✓ Output confirmed as Sketch object")
            print(f"  path_num: {sketch.path_num}")
            print(f"  curve_num: {sketch.curve_num}")
        else:
            print(f"  ⚠ Error: Output is {type(sketch)}, expected Sketch")

        print("Done.")