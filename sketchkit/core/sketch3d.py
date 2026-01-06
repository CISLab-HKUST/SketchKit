import numpy as np
import torch
import json



# ================= 3D Sketch Data Structures =====================
class Point3D:
    def __init__(self, x: float, y: float, z: float, thickness: float | None = None):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.thickness = None if thickness is None else float(thickness)

    def as_array(self):
        return np.array([self.x, self.y, self.z], dtype=np.float32)


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
        
        # Automatically collect widths from the 4 control points
        # [width_at_p_start, width_at_p_ctrl1, width_at_p_ctrl2, width_at_p_end]
        thicknesses = [
            self.p_start.thickness,
            self.p_ctrl1.thickness,
            self.p_ctrl2.thickness,
            self.p_end.thickness
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
                self.widths = None
            else:
                # Find first non-None value as default
                default_width = next((w for w in widths_list if w is not None), 1.0)
                final_widths = [w if w is not None else default_width for w in widths_list]
                self.widths = np.array(final_widths, dtype=np.float32)
        else:
            self.widths = None

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


def save_sketch3d_to_json(sketch: Sketch3D, filepath: str):
    """Save a Sketch3D object to JSON format."""
    data = {"paths": []}
    
    for path in sketch.paths:
        path_data = {"curves": []}
        for curve in path.curves:
            curve_data = {
                "v_start": {
                    "x": float(curve.p_start.x),
                    "y": float(curve.p_start.y),
                    "z": float(curve.p_start.z),
                    "thickness": float(curve.p_start.thickness) if curve.p_start.thickness is not None else None,
                    "color": curve.p_start.color.tolist() if curve.p_start.color is not None else None,
                    "opacity": float(curve.p_start.opacity) if curve.p_start.opacity is not None else None,
                },
                "v_end": {
                    "x": float(curve.p_end.x),
                    "y": float(curve.p_end.y),
                    "z": float(curve.p_end.z),
                    "thickness": float(curve.p_end.thickness) if curve.p_end.thickness is not None else None,
                    "color": curve.p_end.color.tolist() if curve.p_end.color is not None else None,
                    "opacity": float(curve.p_end.opacity) if curve.p_end.opacity is not None else None,
                },
                "ctrl1": {
                    "x": float(curve.p_ctrl1.x),
                    "y": float(curve.p_ctrl1.y),
                    "z": float(curve.p_ctrl1.z),
                    "thickness": float(curve.p_ctrl1.thickness) if curve.p_ctrl1.thickness is not None else None,
                },
                "ctrl2": {
                    "x": float(curve.p_ctrl2.x),
                    "y": float(curve.p_ctrl2.y),
                    "z": float(curve.p_ctrl2.z),
                    "thickness": float(curve.p_ctrl2.thickness) if curve.p_ctrl2.thickness is not None else None,
                },
            }
            # Add widths array if present
            if hasattr(curve, 'widths') and curve.widths is not None:
                curve_data["widths"] = curve.widths.tolist() if isinstance(curve.widths, np.ndarray) else list(curve.widths)
            
            path_data["curves"].append(curve_data)
        data["paths"].append(path_data)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved Sketch3D to {filepath}")


def load_sketch3d_from_json(filepath: str) -> Sketch3D:
    """Load a Sketch3D object from JSON format."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    sketch = Sketch3D()
    
    for path_data in data["paths"]:
        curves = []
        for curve_data in path_data["curves"]:
            v_start_data = curve_data["v_start"]
            v_end_data = curve_data["v_end"]
            ctrl1_data = curve_data["ctrl1"]
            ctrl2_data = curve_data["ctrl2"]
            
            v_start = Vertex3D(
                v_start_data["x"],
                v_start_data["y"],
                v_start_data["z"],
                thickness=v_start_data.get("thickness"),
                color=tuple(v_start_data["color"]) if v_start_data.get("color") else None,
                opacity=v_start_data.get("opacity"),
            )
            v_end = Vertex3D(
                v_end_data["x"],
                v_end_data["y"],
                v_end_data["z"],
                thickness=v_end_data.get("thickness"),
                color=tuple(v_end_data["color"]) if v_end_data.get("color") else None,
                opacity=v_end_data.get("opacity"),
            )
            ctrl1 = Point3D(
                ctrl1_data["x"],
                ctrl1_data["y"],
                ctrl1_data["z"],
                thickness=ctrl1_data.get("thickness"),
            )
            ctrl2 = Point3D(
                ctrl2_data["x"],
                ctrl2_data["y"],
                ctrl2_data["z"],
                thickness=ctrl2_data.get("thickness"),
            )
            
            curve = Curve3D(v_start, v_end, ctrl1, ctrl2)
            
            # Add widths if present
            if "widths" in curve_data:
                curve.widths = np.array(curve_data["widths"], dtype=np.float32)
            
            curves.append(curve)
        
        sketch.add(Path3D(curves))
    
    print(f"Loaded Sketch3D from {filepath} with {sketch.path_num} paths and {sketch.curve_num} curves")
    return sketch