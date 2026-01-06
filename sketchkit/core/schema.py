from pydantic import BaseModel
from typing import List, Optional, Tuple

# Compact representations using Tuples (serialized as JSON arrays)

# [x, y]
PointArray = Tuple[float, float]

# [x, y, pressure, thickness, color, opacity]
VertexArray = Tuple[
    float,                          # x
    float,                          # y
    Optional[float],                # pressure
    Optional[float],                # thickness
    Optional[Tuple[float, float, float]], # color (r,g,b)
    Optional[float]                 # opacity
]

# [start, c1, c2, end]
CurveArray = Tuple[VertexArray, PointArray, PointArray, VertexArray]

class PathSchema(BaseModel):
    curves: List[CurveArray]
    label: Optional[str] = None

class SketchSchema(BaseModel):
    height: float
    width: float
    paths: List[PathSchema]
