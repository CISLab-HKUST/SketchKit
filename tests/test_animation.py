import numpy as np
from PIL import Image

from sketchkit.animation import Animator


if __name__ == "__main__":
    animator = Animator()

    img0_path = "0-0.png"
    img1_path = "0-1.png"
    img0_np = np.array(Image.open(img0_path).convert("RGB"))
    img1_np = np.array(Image.open(img1_path).convert("RGB"))

    img_list = animator.run(
        img0_np, img1_np, inner_frames=24
    )