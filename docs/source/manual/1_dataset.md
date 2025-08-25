# Sketch Datasets

## Conversion

<!-- We should write a class (use `torch.utils.data.Dataset` as a base class) for each dataset. An example is [TUBerlin](../../data/datasets/tu_berlin.py#L142) in folder `data/datasets`. The main procedure is as follows: -->

1. Automatically download and extract the dataset if it is not existed.
2. Load the dataset to obtain data like class names and all the file names.
3. Define the `__getitem__` function:

   - Parse the original data into the desired input format of SketchVG.
   - Define the SketchVG format and return.

## Supported Datasets

   | Dataset | Status | Description | Domain |
   |---------|---------|-------------|---------|
   | OpenSketch | | Open source sketch dataset | General sketches |
   | DifferSketching | | Differentiable sketching dataset | Sketch generation |
   | Tracing-vs-freehand | | Comparison dataset for drawing styles | Drawing analysis |
   | TU-Berlin| ✅ |||
   |Sketchy Database |✅| Large-scale sketch recognition datasets | Sketch classification |
   | Sketch2Pose | | Sketch to pose estimation dataset | Human pose |
   | Where do people draw lines | | Line drawing behavior analysis | Drawing patterns |
   | Benchmark for sketch cleanup (GMU) | | Sketch cleaning and simplification | Sketch processing |
   | PhotoSketching (CMU) | | Photo to sketch conversion dataset | Image-to-sketch |
   | Haoran's awesome datasets | | Collection of sketch datasets | Various |
   | QuickDraw |✅ | Google's quick drawing dataset | Doodle recognition |
   | VR sketching | | Virtual reality sketching dataset | 3D/VR sketching |

## Examples

After the dataset conversion, we can load a dataset with a single-line code:

```python
from sketchkit.data.datasets.tu_berlin import TUBerlin
my_data = TUBerlin()
```

See testing scripts in folders `scripts` for more details.
