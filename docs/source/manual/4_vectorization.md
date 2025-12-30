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

The method implements [Deep Sketch Vectorization via Implicit Surface Extraction](https://cragl.cs.gmu.edu/sketchvector/) from *SIGGRAPH 2024*, which is a fast and accurate vectorization method that can turn clean raster line drawings with complex topology into high quality vector graphics. The method is capable of handling high valence star-junctions.

#### Additional Outputs

After using the code above, a `final.svg` denoting the vector image and its corresponding stroke visualization `vis.svg` can be found in the `svg` folder in the same directory as the script.

### LineDrawer

LineDrawer is a stroke-level vectorization method based on human perception, originally presented in *CAD/Graphics 2025*. The method implements the CNNVE (CNN-based Vectorization Engine) approach to reconstruct complex line art at the stroke level.

#### Overview

The LineDrawer vectorization pipeline consists of the following steps:

1. **Skeletonization**: Converts the input image into a skeleton representation
2. **Graph Building**: Constructs a pixel-level graph from the skeleton
3. **Graph Simplification**: Merges edges to create simplified graph structures
4. **Stroke Extraction**: Uses deep learning to extract stroke masks from the graph
5. **Stroke Merging**: Combines stroke masks to produce the final vectorized output

#### Usage Example

```python
from sketchkit.vectorization import Vectorizer
import numpy as np
from PIL import Image

# Load an input image
img = np.array(Image.open("input.png").convert("L"))

# Initialize the vectorizer with LineDrawer method
vectorizer = Vectorizer(method="LineDrawer", device="cuda")

# Run vectorization
sketch = vectorizer.run(img)

# Access the vectorized paths
print(f"Number of paths: {sketch.path_num}")
print(f"Total curves: {sketch.curve_num}")
```

#### Output Format

The LineDrawer method returns a `Sketch` object containing:
- **Paths**: Each path represents a stroke extracted from the image
- **Curves**: Each path contains multiple cubic Bézier curves
- **Vertices**: Start and end points of each curve with coordinate information

The output can be further processed, rendered, or exported using other SketchKit modules.
