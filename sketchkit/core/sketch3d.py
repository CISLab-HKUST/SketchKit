import numpy as np
import torch 

# ================= 3D Sketch Data Structures =====================
class Point3D:
    def __init__(self, x: float, y: float, z: float):
        self.x = float(x); self.y = float(y); self.z = float(z)
    def as_array(self):
        return np.array([self.x, self.y, self.z], dtype=np.float32)

class Vertex3D(Point3D):
    def __init__(self, x: float, y: float, z: float,
                 thickness: float | None = None,
                 color: tuple[float,float,float] | np.ndarray | None = None,
                 opacity: float | None = None):
        super().__init__(x,y,z)
        self.thickness = None if thickness is None else float(thickness)
        self.color = None if color is None else np.array(color, dtype=np.float32)
        self.opacity = None if opacity is None else float(opacity)

class Curve3D:
    """Cubic Bézier curve in 3D defined by two vertices and two control points."""
    def __init__(self, p_start: Vertex3D, p_end: Vertex3D, p_ctrl1: Point3D, p_ctrl2: Point3D):
        self.p_start = p_start; self.p_end = p_end; self.p_ctrl1 = p_ctrl1; self.p_ctrl2 = p_ctrl2

    def sample(self, samples: int = 32) -> np.ndarray:
        t = np.linspace(0,1,samples, dtype=np.float32)[:,None]
        P0 = self.p_start.as_array(); P1 = self.p_ctrl1.as_array(); P2 = self.p_ctrl2.as_array(); P3 = self.p_end.as_array()
        pts = (1-t)**3*P0 + 3*(1-t)**2*t*P1 + 3*(1-t)*t**2*P2 + t**3*P3
        return pts

    def as_tensor(self) -> torch.Tensor:
        ctrl_points = np.stack([
            self.p_start.as_array(),
            self.p_ctrl1.as_array(),
            self.p_ctrl2.as_array(),
            self.p_end.as_array(),
        ], axis=0)
        return torch.from_numpy(ctrl_points)

class Path3D:
    def __init__(self, curves: list[Curve3D] | None = None):
        self.curves = curves if curves else []
    @property
    def curve_num(self): return len(self.curves)
    def add(self, curve: Curve3D): self.curves.append(curve)
    def stack_tensor(self) -> torch.Tensor:
        return torch.stack([c.as_tensor() for c in self.curves], dim=0)

class Sketch3D:
    def __init__(self, paths: list[Path3D] | None = None):
        self.paths = paths if paths else []
    @property
    def path_num(self): return len(self.paths)
    @property
    def curve_num(self): return sum(p.curve_num for p in self.paths)
    def add(self, path: Path3D): self.paths.append(path)
    def stack_tensor(self) -> torch.Tensor:
        return torch.cat([p.stack_tensor() for p in self.paths], dim=0)