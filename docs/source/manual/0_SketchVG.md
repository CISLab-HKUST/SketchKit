# SketchVG

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

## Examples

```
from sketchkit.core import Vertex, SketchVG
```