# Vectorization

The vectorization module provides functionality to convert raster images into vector sketches. This is useful for tasks that require stroke-level information from pixel-based inputs.

## Usage

The main interface is the `Vectorizer` class.

```python
from sketchkit.vectorization import Vectorizer

# Initialize the vectorizer
# Supported methods: "DeepVecSIG24", "LineDrawer"
vectorizer = Vectorizer(method="DeepVecSIG24", device="cuda")

# Run on a given numpy image array
sketch = vectorizer.run(img)
```


## Available Methods

The module currently supports the following methods:

```{eval-rst}
.. autosummary::
   :toctree: generated

   sketchkit.vectorization.DeepVecSIG24
   sketchkit.vectorization.LineDrawer
```

### Deep-Sketch-Vectorization

The method implements [Deep Sketch Vectorization via Implicit Surface Extraction](https://cragl.cs.gmu.edu/sketchvector/). *Additional description*

*Any other usage beyond basic usage*

*Give some example code of unique features of this method*

```python
xxx
```

### LineDrawer

*Method description*

*Any other usage beyond basic usage*