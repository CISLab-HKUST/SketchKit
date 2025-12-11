# Animation

The sketch animation/inbetweening module provides functionality to generate the inbetween frames given a starting and an ending keyframe.

## Usage

The main interface is the `Animator` class.

```python
from sketchkit.animation import Animator

# Initialize the animator
# Supported methods: "RIFE"
animator = Animator(method="RIFE")

# Run on the given starting and ending keyframes in numpy image arrays.
# One can define the number of intermediate frames.
img_list = animator.run(img0, img1, inner_frames=24)
```


## Available Methods

The module currently supports the following methods:

```{eval-rst}
.. autosummary::
   :toctree: generated

   sketchkit.animation.RIFE
```

### RIFE

The method implements [Real-Time Intermediate Flow Estimation for Video Frame Interpolation](https://github.com/hzwer/ECCV2022-RIFE) from ECCV 2022. After using the code above, the inbetween frames and a `dynamic.gif` can be found in the `animation` folder in the same directory as the script.
