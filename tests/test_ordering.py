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
        sketches = orderer.run([img_path, video_path])
        
        print(f"\n✓ Ordered sketches generated: {len(sketches)}")
        if len(sketches) > 0:
            last_sketch = sketches[-1]
            print(f"  Last sketch has {last_sketch.path_num} paths and {last_sketch.curve_num} curves.")
            
            # Verify it is a Sketch object
            if isinstance(last_sketch, Sketch):
                print("  ✓ Output confirmed as Sketch objects")
            else:
                print(f"  ⚠ Error: Output is {type(last_sketch)}, expected Sketch")

        print("Done.")