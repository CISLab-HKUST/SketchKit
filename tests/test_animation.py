# from sketchkit.animation import Animator

# if __name__ == "__main__":
#     animator = Animator()

#     img0 = "0-0.png"
#     img1 = "0-1.png"
#     img_list = animator.run(
#         img0, img1, inner_frames=24, output_folder="output/animation"
#     )

import sys
import os
sys.path.append(os.getcwd())

import numpy as np
from PIL import Image
from sketchkit.animation import Animator


if __name__ == '__main__':
    animator = Animator(method="LineDrawer")

    # Load image
    img_path = "a1-2.png"
    img_np = np.array(Image.open(img_path).convert("L"))
    
    # Video path
    video_path = "a1-2.png.mp4"
    
    if not os.path.exists(video_path):
        print(f"⚠ Warning: Video file {video_path} not found")
        print("Test will fail - please provide a video file")
    else:
        print(f"Running animation with image path: {img_path}")
        print(f"Video path: {video_path}")
        
        # Animator expects paths and output_folder to save GIF
        frames = animator.run(img_path, video_path, output_folder="test_output")
        
        print(f"\n✓ Animation frames generated: {len(frames)}")
        print("Check test_output folder for results and GIF.")
