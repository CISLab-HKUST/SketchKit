# Representation

## Overview
SketchKit uses a unified representation for sketches, making it easy to work with various datasets.  <br>
All sketches are converted into the **Sketch standard format**, which provides a structured way to access paths, curves and points.

## Architecture Overview
A `Sketch` is composed of several levels of hierarchy, which contains a list of `Path`s. <br>
A `Path` contains a list of `Curve`s. <br>
And each `Curve` is a cubic Bézier curve with start and end points and two control points. <br>
A `Point` contains coordinate values (x/y) and other drawing attributes.

The structure can be visualized as follows:

```python
Sketch
  - Path
    - Curve
      - Start Point
      - End Point
      - Control Point 1
      - Control Point 2
    - Curve
    ...
  - Path
  ...
```

## Core Components
### Sketch
- The top-level object representing a full sketch.  
- Contains multiple paths. 

### Path
- A sequence of one or more curves that form a continuous stroke.  
- e.g. A single hand-drawn line

### Curve
- A cubic Bézier curve connecting two points. 
- e.g. cutting a path into multiple parts, each part represents a curve
- Defined by:
  - Start Point
  - End Point
  - Control Point 1
  - Control Point 2

### Point
- The most basic element in a sketch.
- Stores the 2D position `(x, y)` of a point.
- Can also carry drawing/style attributes, such as:
  - `pressure`: pen pressure at this point
  - `thickness`: stroke width at this point
  - `color`: RGB color
  - `opacity`: opacity value
  - `attributes`: custom metadata
- Used for all curve control points and endpoints.

## Example

Here's a example to demonstrate how to explore SketchKit's representation:

```python
from sketchkit.datasets import OpenSketch
from sketchkit.renderer import CairoRenderer

# Load a dataset and get a sketch
print("Loading OpenSketch dataset...")
dataset = OpenSketch()
sketch = dataset[0]

print("=== Sketch Overview ===")
print(f"Dataset: {type(dataset).__name__}")
print(f"Sketch has {sketch.path_num} paths")
print(f"Total curves: {sketch.curve_num}")

# Analyze each path
print("\n=== Path Analysis ===")
for i, path in enumerate(sketch.paths):
    print(f"Path {i}:")
    print(f"  - Curves: {len(path.curves)}")

    # Look at first curve in detail
    if path.curves:
        first_curve = path.curves[0]
        print(
            f"  - First curve start: ({first_curve.p_start.x:.1f}, {first_curve.p_start.y:.1f})"
        )
        print(
            f"  - First curve end: ({first_curve.p_end.x:.1f}, {first_curve.p_end.y:.1f})"
        )

        # Check for pressure information
        if first_curve.p_start.pressure is not None:
            print(f"  - Start pressure: {first_curve.p_start.pressure:.2f}")
        else:
            print("  - No pressure information available")

# Calculate some statistics
print("\n=== Statistics ===")
total_points = 0
all_x_coords = []
all_y_coords = []

for path in sketch.paths:
    for curve in path.curves:
        total_points += 2  # start and end point
        all_x_coords.extend([curve.p_start.x, curve.p_end.x])
        all_y_coords.extend([curve.p_start.y, curve.p_end.y])

if all_x_coords:
    print(f"Total points: {total_points}")
    print(f"X range: {min(all_x_coords):.1f} to {max(all_x_coords):.1f}")
    print(f"Y range: {min(all_y_coords):.1f} to {max(all_y_coords):.1f}")

# Render the sketch
print("\n=== Rendering ===")
renderer = CairoRenderer()
image = renderer.render(sketch)
output_path = "example_sketch.png"
image.save(output_path)
print(f"Sketch rendered and saved to {output_path}")
```