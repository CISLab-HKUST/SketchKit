# Get Started

## How to install Sketchkit

### Install anaconda

Anaconda is a free, open-source distribution of Python and R designed for data science, machine learning, and scientific computing. It includes many popular libraries and tools, and comes with the Conda package manager to easily manage environments and deployments.

Download the Anaconda installer that matches your operating system from the Tsinghua mirror.

For example, my operating system is Windows, so I click **Anaconda3-2025.06-1-Windows-x86_64.exe** to download the installer.

**Important:** During installation, check the box **"Add Anaconda to my PATH environment variable"** so you can use the `conda` command directly in the terminal.

After installation, open terminal, type `conda --version`, if you see the version information of Anaconda, then it is installed successfully.

### Install Sketchkit

Extract `SketchKit.zip` to `SketchKit` folder

Open Powershell/Terminal in `SketchKit` folder

Type follow command:

```bash
conda create -n sketchkit python==3.12
conda activate sketchkit
pip install .
```
After installation, you can import `sketchkit` in your python program.

### Example
#### Load dataset

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

#### Render the sketch

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

