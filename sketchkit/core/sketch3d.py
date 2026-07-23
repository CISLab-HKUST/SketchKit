import numpy as np
import torch
import json


from .schema import (
    Curve3DSchema,
    Path3DSchema,
    Point3DSchema,
    Sketch3DSchema,
)

from .sketch import Point, Curve, Path, Sketch
from .camera import Camera


def _copy_attributes(attributes: dict | None) -> dict:
    return dict(attributes) if attributes is not None else {}


def _serialize_point3d(point: "Point3D") -> Point3DSchema:
    return Point3DSchema(
        x=float(point.x),
        y=float(point.y),
        z=float(point.z),
        thickness=float(point.thickness) if point.thickness is not None else None,
        color=tuple(point.color.tolist()) if point.color is not None else None,
        opacity=float(point.opacity) if point.opacity is not None else None,
        attributes=_copy_attributes(point.attributes),
    )


def _deserialize_point3d(point_data: Point3DSchema) -> "Point3D":
    return Point3D(
        x=point_data.x,
        y=point_data.y,
        z=point_data.z,
        thickness=point_data.thickness,
        color=point_data.color,
        opacity=point_data.opacity,
        attributes=point_data.attributes,
    )


# ================= 3D Sketch Data Structures =====================
class Point3D:
    """3D point with optional drawing attributes."""

    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        thickness: float | None = None,
        color: tuple[float, float, float] | np.ndarray | None = None,
        opacity: float | None = None,
        attributes: dict | None = None,
    ):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.thickness = None if thickness is None else float(thickness)
        self.color = None if color is None else np.array(color, dtype=np.float32)
        self.opacity = None if opacity is None else float(opacity)
        self.attributes = _copy_attributes(attributes)

    def as_array(self):
        return np.array([self.x, self.y, self.z], dtype=np.float32)

    def to_2D(self, camera: Camera) -> Point:
        """Project the 3D point to 2D using the given camera."""
        import numpy as np

        # Convert to homogeneous coordinates


class Curve3D:
    """Cubic Bézier curve in 3D defined by two endpoints and two control points."""

    def __init__(
        self,
        p_start: Point3D,
        p_end: Point3D,
        p_ctrl1: Point3D,
        p_ctrl2: Point3D,
        attributes: dict | None = None,
    ):
        self.p_start = p_start
        self.p_end = p_end
        self.p_ctrl1 = p_ctrl1
        self.p_ctrl2 = p_ctrl2
        self.attributes = _copy_attributes(attributes)

    @property
    def widths(self) -> np.ndarray | None:
        # Automatically collect widths from the 4 control points
        # [width_at_p_start, width_at_p_ctrl1, width_at_p_ctrl2, width_at_p_end]
        thicknesses = [
            self.p_start.thickness,
            self.p_ctrl1.thickness,
            self.p_ctrl2.thickness,
            self.p_end.thickness,
        ]

        # Check if any thickness is specified
        if any(t is not None for t in thicknesses):
            # Build width array, using default for None values
            widths_list = []
            for t in thicknesses:
                if t is not None:
                    widths_list.append(float(t))
                else:
                    # Use a sentinel value or fallback
                    widths_list.append(None)

            # If we have mixed None and values, we need to decide:
            # Option 1: Keep None entries and handle in renderer
            # Option 2: Fill None with a default value here
            # Let's fill with first non-None value or 1.0 as fallback
            if all(w is None for w in widths_list):
                return None
            else:
                # Find first non-None value as default
                default_width = next((w for w in widths_list if w is not None), 1.0)
                final_widths = [
                    w if w is not None else default_width for w in widths_list
                ]
                return np.array(final_widths, dtype=np.float32)
        else:
            return None

    def sample(self, samples: int = 32) -> np.ndarray:
        t = np.linspace(0, 1, samples, dtype=np.float32)[:, None]
        P0 = self.p_start.as_array()
        P1 = self.p_ctrl1.as_array()
        P2 = self.p_ctrl2.as_array()
        P3 = self.p_end.as_array()
        pts = (
            (1 - t) ** 3 * P0
            + 3 * (1 - t) ** 2 * t * P1
            + 3 * (1 - t) * t**2 * P2
            + t**3 * P3
        )
        return pts

    def as_tensor(self) -> torch.Tensor:
        ctrl_points = np.stack(
            [
                self.p_start.as_array(),
                self.p_ctrl1.as_array(),
                self.p_ctrl2.as_array(),
                self.p_end.as_array(),
            ],
            axis=0,
        )
        return torch.from_numpy(ctrl_points)


class Path3D:
    def __init__(self, curves: list[Curve3D] | None = None, label: str | None = None, attributes: dict | None = None):
        self.curves = curves if curves else []
        self.label = label
        self.attributes = _copy_attributes(attributes)

    @property
    def curve_num(self):
        return len(self.curves)

    def add(self, curve: Curve3D):
        self.curves.append(curve)

    def stack_tensor(self) -> torch.Tensor:
        return torch.stack([c.as_tensor() for c in self.curves], dim=0)


class Sketch3D:
    def __init__(self, paths: list[Path3D] | None = None, attributes: dict | None = None):
        self.paths = paths if paths else []
        self.attributes = _copy_attributes(attributes)

    @property
    def path_num(self):
        return len(self.paths)

    @property
    def curve_num(self):
        return sum(p.curve_num for p in self.paths)

    def add(self, path: Path3D):
        self.paths.append(path)

    def stack_tensor(self) -> torch.Tensor:
        return torch.cat([p.stack_tensor() for p in self.paths], dim=0)

    def to_json(self, file_path: str):
        """Save the 3D sketch to a JSON file (Compact Format).

        Args:
            file_path (str): Path to the output JSON file.
        """
        paths_data = []
        for path in self.paths:
            curves_data = []
            for curve in path.curves:
                widths = (
                    curve.widths.tolist()
                    if hasattr(curve, "widths") and curve.widths is not None
                    else None
                )

                curves_data.append(
                    Curve3DSchema(
                        p_start=_serialize_point3d(curve.p_start),
                        p_ctrl1=_serialize_point3d(curve.p_ctrl1),
                        p_ctrl2=_serialize_point3d(curve.p_ctrl2),
                        p_end=_serialize_point3d(curve.p_end),
                        widths=widths,
                        attributes=_copy_attributes(curve.attributes),
                    )
                )

            paths_data.append(
                Path3DSchema(
                    curves=curves_data,
                    label=path.label,
                    attributes=_copy_attributes(path.attributes),
                )
            )

        sketch_data = Sketch3DSchema(
            paths=paths_data,
            attributes=_copy_attributes(self.attributes),
        )

        with open(file_path, "w") as f:
            f.write(sketch_data.model_dump_json(indent=None))

    @classmethod
    def from_json(cls, json_path: str) -> "Sketch3D":
        """Load a 3D sketch from a JSON file (Compact Format).

        Args:
            json_path (str): Path to the JSON file.

        Returns:
            Sketch3D: Loaded sketch object.
        """
        with open(json_path, "r") as f:
            json_content = f.read()

        sketch_data = Sketch3DSchema.model_validate_json(json_content)

        sketch = cls(attributes=sketch_data.attributes)
        for path_data in sketch_data.paths:
            curves = []
            for curve_data in path_data.curves:
                p_start = _deserialize_point3d(curve_data.p_start)
                ctrl1 = _deserialize_point3d(curve_data.p_ctrl1)
                ctrl2 = _deserialize_point3d(curve_data.p_ctrl2)
                p_end = _deserialize_point3d(curve_data.p_end)

                curves.append(
                    Curve3D(
                        p_start,
                        p_end,
                        ctrl1,
                        ctrl2,
                        attributes=curve_data.attributes,
                    )
                )
            sketch.add(
                Path3D(
                    curves,
                    label=path_data.label,
                    attributes=path_data.attributes,
                )
            )

        return sketch
