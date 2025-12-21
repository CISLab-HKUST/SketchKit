# Cleanup

The sketch cleanup (a.k.a. simplification) module provides functionality to convert a raster rough sketch into a clean sketch. 

## Usage

The main interface is the `Cleaner` class.

```python
from sketchkit.cleanup import Cleaner

# Initialize the cleaner
# Supported methods: "MasterSketch"
cleaner = Cleaner(method="MasterSketch")

# Run on a given numpy image array
sketch = cleaner.run(img)
```


## Available Methods

The module currently supports the following methods:

```{eval-rst}
.. autosummary::
   :toctree: generated

   sketchkit.cleanup.MasterSketch
```

### Mastering Sketching

The method implements [Mastering Sketching: Adversarial Augmentation for Structured Prediction](https://esslab.jp/~ess/en/research/sketch_master/) from SIGGRAPH 2018. After using the code above, a `result.png` denoting the simplified image can be found in the `cleanup` folder in the same directory as the script.
