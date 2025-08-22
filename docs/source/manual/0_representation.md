# Representation

## Data Format

The representation of sketches is defined in `sketchkit/core/sketch`. 

We use **Sketch** class in folder `data/dataformat/sketch` as our sketch data format. 

A `Sketch` contains a list of `Path`s. A `Path` contains a list of `Stroke`s. And each `Stroke` is a cubic bezier curve with start and end vertices and two control points. A `Vertex` is a point with many other drawing attributes than naive `Point` (which only has x,y position.)

The structure can be visuialized as follows:

```
Sketch
  - Path
    - Stroke
      - Start Vertex
      - End Vertex
      - Control Point 1
      - Control Point 2
    - Stroke
    ...
  - Path
  ...
```