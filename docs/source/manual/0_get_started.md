# Get Started

## How to install Sketchkit

### Install Anaconda

Anaconda is a free, open-source distribution of Python and R designed for data science, machine learning, and scientific computing. It includes many popular libraries and tools, and comes with the Conda package manager to easily manage environments and deployments.

#### Download Anaconda

Download the Anaconda installer that matches your operating system from one of these sources:
- **Official**: [Anaconda Downloads](https://www.anaconda.com/download) (Global)
- **Mirror**: [Tsinghua Mirror](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/) (Faster in China)



#### Installation Steps

1. **Run the installer** and follow the setup wizard
2. **Important**: During installation, check the box **"Add Anaconda to my PATH environment variable"** so you can use the `conda` command directly in the terminal
3. **Restart your terminal** after installation completes

#### Verify Installation

Open terminal/PowerShell and type:
```bash
conda --version
```

If you see the version information (e.g., `conda 24.7.1`), then Anaconda is installed successfully.

**Troubleshooting:**
- If `conda` command is not found, restart your terminal or manually add Anaconda to your PATH
- On Windows, try using "Anaconda Prompt" from the Start menu

### Install SketchKit

#### Configure API keys
The API keys are stored in `~/sketchkit/config`. The keys should be stored like the following:

```sh
OPENAI_API_KEY="sk-proj-xxxxxxxxxxxxx"
```

#### Step 1: Prepare SketchKit Files
Extract `SketchKit.zip` to a folder named `SketchKit`

#### Step 2: Open Terminal
Navigate to the `SketchKit` folder in your terminal:

**Method 1 (File Manager):**
- Right-click in the `SketchKit` folder and select "Open in Terminal" or "Open PowerShell here"

**Method 2 (Command Line):**
```bash
cd /path/to/SketchKit
```
Replace `/path/to/SketchKit` with the actual path to your SketchKit folder.

#### Step 3: Install SketchKit
Run the following commands:

```bash
# Create a new conda environment with Python 3.12
conda create -n sketchkit python==3.12

# Activate the environment  
conda activate sketchkit

# Install SketchKit in development mode
pip install -e .
```
After installation, you can import `sketchkit` in your python program.

#### Step 4: Verify Installation
Test your installation:

```python
import sketchkit
from sketchkit.datasets import QuickDraw
print("SketchKit installed successfully!")
```

**Note**: Use `pip install -e .` (with `-e`) for development installation, which allows you to modify the code and see changes immediately.

### Example
#### Load dataset

SketchKit provides several options for loading datasets:

##### Basic Loading (Recommended for beginners)
```python
# Load metadata only (fast, low memory usage)
dataset = QuickDraw()
```
**Load All Data to Memory**
```python
# Load all sketches to memory (faster access, high memory usage)
dataset = QuickDraw(load_all=True)
```

**Use Alternative Download Source**
```python
# Download from CISLab mirror (may be faster in some regions)
dataset = QuickDraw(cislab_source=True)
```
##### Working with Dataset Metadata

All datasets provide a `items_metadata` attribute that helps you search and filter sketches:

```python
from sketchkit.datasets import QuickDraw

# default load
dataset = QuickDraw()

# search data with "category = cat" and "split = train"
cats = dataset.items_metadata[
    (dataset.items_metadata["category"] == "cat")
    & (dataset.items_metadata["split"] == "train")
]

# get sketches from the dataset
cats_sketch = [dataset[row.id] for _, row in cats[:100].iterrows()]
```

#### Render the sketch

Once you have loaded sketches, you can render them as raster images using SketchKit's CairoRenderer. This converts vector sketch data into images (e.g., PNG).

```python
from PIL import Image
from sketchkit.renderer import CairoRenderer

# Load a sketch from the dataset
sketch = dataset[100]

# Create a renderer with canvas size and background color (RGB)
renderer = CairoRenderer(800, (1, 1, 1))  # 800x800 canvas, white background
raster_image = renderer.render(sketch)

# Save the rendered image
outpath = "my_sketch.png"
raster_image_png = Image.fromarray(raster_image, "RGB")
raster_image_png.save(outpath, "PNG")
print(f"Sketch saved to {outpath}")
```

