# Get Started

## Installation

Generally, get into `SketchKit` dir, activate your virtual env and run `pip install .`.

### Conda

```bash
conda create -n sketchkit python==3.12
conda activate sketchkit
cd SketchKit
pip install .
```

### uv

```bash
uv sync
uv pip install .
source .venv/bin/activate
```

## Load dataset

```python
from sketchkit.datasets import QuickDraw

# default load
dataset = QuickDraw()

# load all data at once
# dataset = QuickDraw(load_all=True)

# download dataset from CISLab
# dataset = QuickDraw(cislab_source=True)

# search data with "category = cat" and "split = train"
cats = dataset.items_metadata[
    (dataset.items_metadata["category"] == "cat")
    & (dataset.items_metadata["split"] == "train")
]

# get sketches from the dataset
cats_sketch = [dataset[row.id] for _, row in cats[:100].iterrows()]
```

## Render the sketch

```python
from PIL import Image
from sketchkit.renderer.cairo_renderer import CairoRenderer

...
sketch = dataset[100]

renderer = CairoRenderer(800, (1, 1, 1))
raster_image = renderer.render(sketch)
outpath = "opensketch-test.png"
raster_image_png = Image.fromarray(raster_image, "RGB")
raster_image_png.save(outpath, "PNG")
```
