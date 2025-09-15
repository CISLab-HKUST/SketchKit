# Representation

## Overview
SketchKit uses a unified representation for sketches, making it easy to work with various datasets.  <br>
All sketches are converted into the **Sketch standard format**, which provides a structured way to access paths, curves and points.

## Core Concepts

A `Sketch` is composed of several levels of hierarchy, which contains a list of `Path`s. <br>
A `Path` contains a list of `Curve`s. <br>
And each `Curve` is a cubic bezier curve with start and end vertices and two control points. <br>
A `Vertex` is a point with many other drawing attributes than naive `Point` (which only has x,y position.)

### Sketch
- The top-level object representing a full sketch.  
- Contains multiple paths. 

### Path
- A sequence of one or more curves that form a continuous stroke.  
- e.g. A single hand-drawn line

### Curve
- A cubic Bézier curve connecting two vertices. 
- e.g. cutting a path into multiple parts, each part represents a curve
- Defined by:
  - Start Vertex
  - End Vertex
  - Control Point 1
  - Control Point 2

### Vertex
- A specialized point with additional drawing attributes.  
- Attributes include:
  - `pressure`: pen pressure at this point
  - `thickness`: stroke width at this point
  - Other style information
- e.g. every curve’s start and end points are vertices.

### Point
- The most basic element in a sketch.  
- Stores the 2D position `(x, y)` of a point.  
- e.g. mostly serve as **control points** for cubic Bézier curves, defining the curve's shape.

## Visualization

The structure can be visualized as follows:

```python
Sketch
  - Path
    - Curve
      - Start Vertex
      - End Vertex
      - Control Point 1
      - Control Point 2
    - Curve
    ...
  - Path
  ...
```
