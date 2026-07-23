from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Tuple


JsonDict = Dict[str, Any]
ColorTuple = Tuple[float, float, float]


class PointSchema(BaseModel):
    x: float
    y: float
    pressure: Optional[float] = None
    thickness: Optional[float] = None
    color: Optional[ColorTuple] = None
    opacity: Optional[float] = None
    attributes: JsonDict = Field(default_factory=dict)


class CurveSchema(BaseModel):
    p_start: PointSchema
    p_crtl1: PointSchema
    p_crtl2: PointSchema
    p_end: PointSchema
    attributes: JsonDict = Field(default_factory=dict)


class PathSchema(BaseModel):
    curves: List[CurveSchema]
    label: Optional[str] = None
    attributes: JsonDict = Field(default_factory=dict)


class SketchSchema(BaseModel):
    height: float
    width: float
    paths: List[PathSchema]
    attributes: JsonDict = Field(default_factory=dict)


class Point3DSchema(BaseModel):
    x: float
    y: float
    z: float
    thickness: Optional[float] = None
    color: Optional[ColorTuple] = None
    opacity: Optional[float] = None
    attributes: JsonDict = Field(default_factory=dict)


class Curve3DSchema(BaseModel):
    p_start: Point3DSchema
    p_ctrl1: Point3DSchema
    p_ctrl2: Point3DSchema
    p_end: Point3DSchema
    widths: Optional[List[float]] = None
    attributes: JsonDict = Field(default_factory=dict)


class Path3DSchema(BaseModel):
    curves: List[Curve3DSchema]
    label: Optional[str] = None
    attributes: JsonDict = Field(default_factory=dict)


class Sketch3DSchema(BaseModel):
    paths: List[Path3DSchema]
    attributes: JsonDict = Field(default_factory=dict)
