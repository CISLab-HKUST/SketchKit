import numpy as np
import torch
import json


from .schema import Sketch3DSchema, Path3DSchema, Vertex3DArray, Point3DArray

from .sketch import Point, Vertex, Curve, Path, Sketch
from .camera import Camera


# ================= 3D Sketch Data Structures =====================
class Point3D:
    def __init__(self, x: float, y: float, z: float, thickness: float | None = None):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.thickness = None if thickness is None else float(thickness)

    def as_array(self):
        return np.array([self.x, self.y, self.z], dtype=np.float32)

    def to_2D(self, camera: Camera) -> Point:
        """Project the 3D point to 2D using the given camera."""
        import numpy as np

        # Convert to homogeneous coordinates


class Vertex3D(Point3D):
    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        thickness: float | None = None,
        color: tuple[float, float, float] | np.ndarray | None = None,
        opacity: float | None = None,
    ):
        super().__init__(x, y, z)
        self.thickness = None if thickness is None else float(thickness)
        self.color = None if color is None else np.array(color, dtype=np.float32)
        self.opacity = None if opacity is None else float(opacity)

    def to_2D(self, camera: Camera) -> Vertex:
        """Project the 3D vertex to 2D using the given camera."""
        import numpy as np

        # Convert to homogeneous coordinates


class Curve3D:
    """Cubic Bézier curve in 3D defined by two vertices and two control points."""

    def __init__(
        self,
        p_start: Vertex3D,
        p_end: Vertex3D,
        p_ctrl1: Point3D,
        p_ctrl2: Point3D,
    ):
        self.p_start = p_start
        self.p_end = p_end
        self.p_ctrl1 = p_ctrl1
        self.p_ctrl2 = p_ctrl2

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
    def __init__(self, curves: list[Curve3D] | None = None):
        self.curves = curves if curves else []

    @property
    def curve_num(self):
        return len(self.curves)

    def add(self, curve: Curve3D):
        self.curves.append(curve)

    def stack_tensor(self) -> torch.Tensor:
        return torch.stack([c.as_tensor() for c in self.curves], dim=0)


class Sketch3D:
    def __init__(self, paths: list[Path3D] | None = None):
        self.paths = paths if paths else []

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
                v_start: Vertex3DArray = (
                    float(curve.p_start.x),
                    float(curve.p_start.y),
                    float(curve.p_start.z),
                    (
                        float(curve.p_start.thickness)
                        if curve.p_start.thickness is not None
                        else None
                    ),
                    (
                        tuple(curve.p_start.color.tolist())
                        if curve.p_start.color is not None
                        else None
                    ),
                    (
                        float(curve.p_start.opacity)
                        if curve.p_start.opacity is not None
                        else None
                    ),
                )
                v_end: Vertex3DArray = (
                    float(curve.p_end.x),
                    float(curve.p_end.y),
                    float(curve.p_end.z),
                    (
                        float(curve.p_end.thickness)
                        if curve.p_end.thickness is not None
                        else None
                    ),
                    (
                        tuple(curve.p_end.color.tolist())
                        if curve.p_end.color is not None
                        else None
                    ),
                    (
                        float(curve.p_end.opacity)
                        if curve.p_end.opacity is not None
                        else None
                    ),
                )
                p_ctrl1: Point3DArray = (
                    float(curve.p_ctrl1.x),
                    float(curve.p_ctrl1.y),
                    float(curve.p_ctrl1.z),
                    (
                        float(curve.p_ctrl1.thickness)
                        if curve.p_ctrl1.thickness is not None
                        else None
                    ),
                )
                p_ctrl2: Point3DArray = (
                    float(curve.p_ctrl2.x),
                    float(curve.p_ctrl2.y),
                    float(curve.p_ctrl2.z),
                    (
                        float(curve.p_ctrl2.thickness)
                        if curve.p_ctrl2.thickness is not None
                        else None
                    ),
                )

                widths = (
                    curve.widths.tolist()
                    if hasattr(curve, "widths") and curve.widths is not None
                    else None
                )

                curves_data.append((v_start, p_ctrl1, p_ctrl2, v_end, widths))
                # curves_data.append((v_start, p_ctrl1, p_ctrl2, v_end))

            paths_data.append(Path3DSchema(curves=curves_data))

        sketch_data = Sketch3DSchema(paths=paths_data)

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

        sketch = cls()
        for path_data in sketch_data.paths:
            curves = []
            for curve_tuple in path_data.curves:
                # curve_tuple: [v_start, ctrl1, ctrl2, v_end, widths]
                v_start_arr = curve_tuple[0]
                ctrl1_arr = curve_tuple[1]
                ctrl2_arr = curve_tuple[2]
                v_end_arr = curve_tuple[3]
                # widths = curve_tuple[4]

                v_start = Vertex3D(
                    x=v_start_arr[0],
                    y=v_start_arr[1],
                    z=v_start_arr[2],
                    thickness=v_start_arr[3],
                    color=v_start_arr[4],
                    opacity=v_start_arr[5],
                )
                ctrl1 = Point3D(
                    x=ctrl1_arr[0],
                    y=ctrl1_arr[1],
                    z=ctrl1_arr[2],
                    thickness=ctrl1_arr[3],
                )
                ctrl2 = Point3D(
                    x=ctrl2_arr[0],
                    y=ctrl2_arr[1],
                    z=ctrl2_arr[2],
                    thickness=ctrl2_arr[3],
                )
                v_end = Vertex3D(
                    x=v_end_arr[0],
                    y=v_end_arr[1],
                    z=v_end_arr[2],
                    thickness=v_end_arr[3],
                    color=v_end_arr[4],
                    opacity=v_end_arr[5],
                )

                curve = Curve3D(v_start, v_end, ctrl1, ctrl2)
                # if widths is not None:
                #     curve.widths = np.array(widths, dtype=np.float32)

                curves.append(curve)
            sketch.add(Path3D(curves))

        return sketch
