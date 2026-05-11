# Sketch-Based Modeling

The sketch-based modeling module provides functionality to generate 3D models from sketch inputs, which include both vector sketches and raster images.

**Source**: `tests/unit/test_sketch2model.py`

## Available Methods

The module currently supports the following methods:

```{eval-rst}
.. autosummary::
   :toctree: generated

   sketchkit.sketch2model.Teddy
   sketchkit.sketch2model.SENS
   sketchkit.sketch2model.Doubao
```

### Teddy

The method implements [Teddy: A Sketching Interface for 3-D Freeform Design](https://doi.org/10.1145/311535.311602) from *SIGGRAPH 1999*. It inflates user-drawn 2D silhouette sketches to generate smooth 3D freeform models through an interactive sketch-based modeling pipeline.

**Output:** Returns a 3D mesh model in triangle mesh format. In the example below, the saved file is `teddy_model.obj`.

### SENS

The method implements [SENS: Part-Aware Sketch-based Implicit Neural Shape Modeling](https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.15015) from *Computer Graphics Forum 2024*. It uses part-aware implicit neural representations to infer complete 3D shapes from single or multi-view sketches, learning the sketch-to-3D mapping through neural networks.

**Output:** Returns a 3D mesh reconstruction from the implicit field. In the example below, the saved file is `sens_model.obj`.

### Seed3D

The method implements [Seed3D 1.0: From Images to High-Fidelity Simulation-Ready 3D Assets](https://bytedance.github.io/seed3d/) from *ByteDance Seed, 2025*. It leverages large-scale generative models to generate high-fidelity, simulation-ready 3D assets from sketches or images, including complete modeling of geometry, materials, and physical properties.

**Output:** Returns a high-quality 3D mesh model with optional PBR materials. In the example below, the saved file is `seed3d_model.glb`.

## Code

The main interface is the `Modeler` class.

```python
from sketchkit.datasets.differsketching import DifferSketching
from sketchkit.sketch2model import Modeler

# Load an input sketch
dataset = DifferSketching()
sketch = dataset[0]


# 1) Teddy
modeler = Modeler(method="teddy")
out_teddy = modeler.run(sketch, output_path="teddy_model.obj")

# 2) SENS
modeler = Modeler(method="SENS")
out_teddy = modeler.run(sketch, output_path="sens_model.obj")

# 3) Doubao
modeler = Modeler(method="doubao")
out_teddy = modeler.run(sketch, output_path="seed3d_model.glb")

# You can view the models by tools like Meshlab.
```

## Usage Notes

**Constructor**

- `Modeler(method=...)`
  - `method`: backend name. Supported values are `"teddy"`, `"sens"`, `"doubao"`, `"seed3d"`, `"seed-3d"`, `"doubao-seed3d"` (uppercase aliases are also accepted).

**Run Method**

- `run(input, output_path)`
  - `input`: accepts a `PIL.Image`, a `Sketch`, or an image path.
  - `output_path`: output path of the model.
