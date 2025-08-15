# Data

## Data Format

We use **SketchVG** class in folder `data/dataformat/sketch` as our sketch data format. The class has a nested structure as follows:

```
SketchVG
  - Path
    - BaseStroke (Line/QuadraticBezierCurve/CubicBezierCurve)
      - Vertex
      - Vertex
      - Attributes: thickness, pressure, color, opacity, texture
    - BaseStroke
    ...
  - Path
  ...
```

## Dataset Conversion

We should write a class (use `torch.utils.data.Dataset` as a base class) for each dataset. An example is [TUBerlin](./sketchkit/data/datasets/tu_berlin.py) in folder `data/datasets`. The main procedure is as follows:

1. Automatically download and extract the dataset if it is not existed.
2. Load the dataset to obtain data like class names and all the file names.
3. Define the `__getitem__` function: 

   - Parse the original data into the desired input format of SketchVG.
   - Define the SketchVG format and return.



## Examples

After the dataset conversion, we can load a dataset with a single-line code:

```python
from sketchkit.data.datasets.tu_berlin import TUBerlin
my_data = TUBerlin()
```

See testing scripts in folders `scripts` for more details.
