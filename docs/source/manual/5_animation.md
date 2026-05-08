# Animation

The sketch animation/inbetweening module provides functionality to generate the inbetween frames given a starting and an ending keyframe.

**Source**: `tests/unit/test_animation.py`


## Available Methods

The module currently supports the following methods:

```{eval-rst}
.. autosummary::
   :toctree: generated

   sketchkit.animation.RIFE
```

### RIFE

The method implements [Real-Time Intermediate Flow Estimation for Video Frame Interpolation](https://github.com/hzwer/ECCV2022-RIFE) from ECCV 2022. It is originally designed for video frame interpolation for arbitrary videos.

**Output:** After using the code above, the inbetween frames and a `dynamic.gif` can be found in the `animation` folder in the same directory as the script.



## Code

The main interface is the `Animator` class.

```python
from sketchkit.animation import Animator
import numpy as np
from PIL import Image

# Load the input keyframes
img0 = np.array(Image.open("input0.png").convert("RGB"))
img1 = np.array(Image.open("input1.png").convert("RGB"))

# Initialize the animator with RIFE method
animator = Animator(method="RIFE")

# Run animation.
img_list = animator.run(img0, img1, inner_frames=24)
```

**Arguments**

- `inner_frames`: the number of intermediate frames.
