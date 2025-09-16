import numpy as np
from numpy.polynomial.legendre import leggauss
try:
    import torch
    has_torch = True
except ImportError:
    has_torch = False

def points_collinear(a, b, c, epsilon=1e-6):
    """Check if three points are collinear.

    Args:
        a (np.ndarray): First point (2,).
        b (np.ndarray): Second point (2,).
        c (np.ndarray): Third point (2,).
        epsilon (float, optional): Threshold for floating point comparison. Defaults to 1e-6.

    Returns:
        bool: True if points are collinear, False otherwise.
    """
    # Calculate cross product: (b.x - a.x)*(c.y - a.y) - (b.y - a.y)*(c.x - a.x)
    cross_product = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    # Consider floating point precision, check if absolute value is below threshold
    return abs(cross_product) < epsilon

def is_line(x: np.ndarray, epsilon: float = 1e-6):
    """Check if a Curve is a line.

    Args:
        x: [N, 4, 2] Control points (P0, P1, P2, P3) as numpy array
        epsilon (float, optional): Threshold for floating point comparison. Defaults to 1e-6.

    Returns:
        [N] bool: True if the curve is a line, False otherwise.
    """
    ret = []
    for i in range(x.shape[0]):
        t1 = points_collinear(x[i, 0], x[i, 1], x[i, 2], epsilon)
        t2 = points_collinear(x[i, 0], x[i, 1], x[i, 3], epsilon)
        t3 = points_collinear(x[i, 1], x[i, 2], x[i, 3], epsilon)
        t4 = points_collinear(x[i, 0], x[i, 2], x[i, 3], epsilon)
        ret.append(t1 and t2 and t3 and t4)

    return np.array(ret)


def bezier_lengths(x: np.ndarray, steps: int = 64) -> np.ndarray:
    """
    Calculate the length of Bézier curves.
    
    Args:
        x: [N, 4, 2] Control points (P0, P1, P2, P3) as numpy array
        steps: Number of sampling steps along the curve
        
    Returns:
        [N, 1] Lengths of the curves as numpy array
    """
    if has_torch:
        # print("Using PyTorch implementation for bezier_lengths")
        # PyTorch implementation (convert numpy array to tensor and use GPU if available)
        # Check if GPU is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        t, w = gauss_legendre_nodes_weights(steps, device)  
        ctrl = torch.from_numpy(x).to(device)
        # 差分向量 [N, 3, 2]
        delta = ctrl[:, 1:] - ctrl[:, :-1]

        # 二次 Bernstein 基函数矩阵  [m, 3]
        one_minus_t = 1 - t
        basis = torch.stack([
            one_minus_t * one_minus_t,
            2 * one_minus_t * t,
            t * t
        ], dim=-1)

        # 速度 B'(t) = 3 * sum_{k=0}^{2} b_{k,2}(t) * (P_{k+1}-P_k)
        # [N, m, 2]
        velocity = 3 * torch.einsum('nkd,mk->nmd', delta, basis)

        # 模长
        speed = velocity.norm(dim=-1)          # [N, m]

        # 高斯积分
        lengths = torch.einsum('nm,m->n', speed, w)
        # Convert numpy array to PyTorch tensor and move to device
        # x_tensor = torch.from_numpy(x).to(device)
        
        # N = x_tensor.shape[0]
        # dtype = x_tensor.dtype
        # t = torch.linspace(0, 1, steps + 1, device=device, dtype=dtype)  # [steps+1]

        # # Calculate curve point coordinates
        # t = t.view(1, -1)  
        # t = t.expand(N, -1) # [N, steps+1]
        # mt = 1 - t
        # mt3 = mt ** 3
        # mt2t = 3 * mt ** 2 * t
        # mtt2 = 3 * mt * t ** 2
        # t3 = t ** 3

        # # Broadcast x to [N, steps+1, 4, 2]
        # x_broadcasted = x_tensor.unsqueeze(1).expand(N, steps + 1, 4, 2)
        
        # # Calculate x/y coordinates [N, steps+1]
        # px = (mt3 * x_broadcasted[:, :, 0, 0] + mt2t * x_broadcasted[:, :, 1, 0] +
        #       mtt2 * x_broadcasted[:, :, 2, 0] + t3 * x_broadcasted[:, :, 3, 0])
        # py = (mt3 * x_broadcasted[:, :, 0, 1] + mt2t * x_broadcasted[:, :, 1, 1] +
        #       mtt2 * x_broadcasted[:, :, 2, 1] + t3 * x_broadcasted[:, :, 3, 1])

        # # Compute differences and calculate length
        # dx = torch.diff(px, dim=1)  # [N, steps]
        # dy = torch.diff(py, dim=1)
        # length = torch.sqrt(dx ** 2 + dy ** 2).sum(dim=1, keepdim=True)  # [N, 1]
        
        # Convert back to numpy array before returning
        return lengths.unsqueeze(-1).cpu().numpy()
    else:
        print("Using NumPy implementation for bezier_lengths")
        # NumPy implementation
        x, w = leggauss(steps)          # 默认区间 [-1,1]
        t = (x + 1) / 2.0           # 映射到 [0,1]
        w *= 0.5                    # 权重同步缩放

        # 2. 差分向量 [N, 3, 2]
        delta = x[:, 1:] - x[:, :-1]

        # 3. 二次 Bernstein 基函数矩阵  [m, 3]
        om = 1 - t
        basis = np.column_stack([om * om, 2 * om * t, t * t])

        # 4. 速度 B'(t) = 3 * sum_{k=0}^{2} b_{k,2}(t) * (P_{k+1}-P_k)
        #    利用 einsum 批量计算  [N, m, 2]
        velocity = 3.0 * np.einsum('nkd,mk->nmd', delta, basis)

        # 5. 模长
        speed = np.hypot(velocity[..., 0], velocity[..., 1])  # [N, m]

        # 6. 高斯积分
        lengths = np.dot(speed, w)      # [N]
        return lengths.reshape(-1, 1)

def gauss_legendre_nodes_weights(m: int, device: torch.device, dtype: torch.dtype = torch.float32):
    """
    返回 [0,1] 区间上的 m 阶 Gauss-Legendre 节点与权重
    """
    x, w = leggauss(m)          # 默认区间 [-1,1]
    x = (x + 1) / 2             # 映射到 [0,1]
    w = w / 2                   # 权重同步缩放
    return torch.as_tensor(x, device=device, dtype=dtype), \
           torch.as_tensor(w, device=device, dtype=dtype)