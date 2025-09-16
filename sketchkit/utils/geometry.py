import numpy as np

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

def bezier_lengths(x: np.ndarray, steps: int = 1000) -> np.ndarray:
    """
    Calculate the length of Bézier curves.
    
    Args:
        x: [N, 4, 2] Control points (P0, P1, P2, P3) as numpy array
        steps: Number of sampling steps along the curve
        
    Returns:
        [N, 1] Lengths of the curves as numpy array
    """
    if has_torch:
        print("Using PyTorch implementation for bezier_lengths")
        # PyTorch implementation (convert numpy array to tensor and use GPU if available)
        # Check if GPU is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Convert numpy array to PyTorch tensor and move to device
        x_tensor = torch.from_numpy(x).to(device)
        
        N = x_tensor.shape[0]
        dtype = x_tensor.dtype
        t = torch.linspace(0, 1, steps + 1, device=device, dtype=dtype)  # [steps+1]

        # Calculate curve point coordinates
        t = t.view(1, -1)  
        t = t.expand(N, -1) # [N, steps+1]
        mt = 1 - t
        mt3 = mt ** 3
        mt2t = 3 * mt ** 2 * t
        mtt2 = 3 * mt * t ** 2
        t3 = t ** 3

        # Broadcast x to [N, steps+1, 4, 2]
        x_broadcasted = x_tensor.unsqueeze(1).expand(N, steps + 1, 4, 2)
        
        # Calculate x/y coordinates [N, steps+1]
        px = (mt3 * x_broadcasted[:, :, 0, 0] + mt2t * x_broadcasted[:, :, 1, 0] +
              mtt2 * x_broadcasted[:, :, 2, 0] + t3 * x_broadcasted[:, :, 3, 0])
        py = (mt3 * x_broadcasted[:, :, 0, 1] + mt2t * x_broadcasted[:, :, 1, 1] +
              mtt2 * x_broadcasted[:, :, 2, 1] + t3 * x_broadcasted[:, :, 3, 1])

        # Compute differences and calculate length
        dx = torch.diff(px, dim=1)  # [N, steps]
        dy = torch.diff(py, dim=1)
        length = torch.sqrt(dx ** 2 + dy ** 2).sum(dim=1, keepdim=True)  # [N, 1]
        
        # Convert back to numpy array before returning
        return length.cpu().numpy()
    else:
        print("Using NumPy implementation for bezier_lengths")
        # NumPy implementation
        N = x.shape[0]
        t = np.linspace(0, 1, steps + 1)  # [steps+1]

        # Calculate curve point coordinates
        t = np.expand_dims(t, axis=0)  # [1, steps+1]
        t = np.repeat(t, N, axis=0)  # [N, steps+1]
        mt = 1 - t
        mt3 = mt ** 3
        mt2t = 3 * mt ** 2 * t
        mtt2 = 3 * mt * t ** 2
        t3 = t ** 3

        # Calculate x/y coordinates [N, steps+1]
        px = (mt3 * x[:, np.newaxis, 0, 0] + mt2t * x[:, np.newaxis, 1, 0] +
              mtt2 * x[:, np.newaxis, 2, 0] + t3 * x[:, np.newaxis, 3, 0])
        py = (mt3 * x[:, np.newaxis, 0, 1] + mt2t * x[:, np.newaxis, 1, 1] +
              mtt2 * x[:, np.newaxis, 2, 1] + t3 * x[:, np.newaxis, 3, 1])

        # Compute differences and calculate length
        dx = np.diff(px, axis=1)  # [N, steps]
        dy = np.diff(py, axis=1)
        length = np.sqrt(dx ** 2 + dy ** 2).sum(axis=1, keepdims=True)  # [N, 1]
        
        return length