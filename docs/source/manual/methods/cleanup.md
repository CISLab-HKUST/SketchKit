# Sketch Cleanup

The sketch cleanup (a.k.a. simplification) module provides functionality to convert a raster rough sketch into a clean sketch. 

**Source**: `tests/unit/test_cleanup.py`

## Available Methods

The module currently supports the following methods:

```{eval-rst}
.. autosummary::
   :toctree: generated

   sketchkit.cleanup.MasterSketch
```

### Mastering Sketching

The method implements [Mastering Sketching: Adversarial Augmentation for Structured Prediction](https://esslab.jp/~ess/en/research/sketch_master/) from SIGGRAPH 2018. It is a method based on generative adversarial network (GAN) with adversarial training.

**Output:** After using the code below, a `result.png` denoting the simplified image can be found in the `cleanup` folder in the same directory as the script.



## Code

The main interface is the `Cleaner` class.

```python
from sketchkit.cleanup import Cleaner
import numpy as np
from PIL import Image

# Load an input image
img = np.array(Image.open("input.png").convert("L"))

# Initialize the cleaner with MasterSketch method
cleaner = Cleaner(method="MasterSketch")

# Run cleanup
sketch = cleaner.run(img)
```

