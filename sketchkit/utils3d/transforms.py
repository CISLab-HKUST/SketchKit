import numpy as np
import torch

def yaw_pitch_roll_from_wxyz(wxyz: np.ndarray) -> tuple[float, float, float]:
    """Convert quaternion (w,x,y,z) to yaw, pitch, roll angles in radians."""
    w, x, y, z = wxyz

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return yaw, pitch, roll


def wxyz_from_yaw_pitch_roll(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Convert yaw, pitch, roll angles to quaternion (w,x,y,z)."""
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])


def translation_matrix_from_transform(transform: np.ndarray) -> np.ndarray:
    """Extract 4x4 translation matrix from 4x4 transformation matrix."""
    translation = np.eye(4)
    translation[:3, 3] = transform[:3, 3]
    return translation


def rotation_matrix_from_transform(transform: np.ndarray) -> np.ndarray:
    """Extract 4x4 rotation matrix from 4x4 transformation matrix."""
    rotation = np.eye(4)
    rotation[:3, :3] = transform[:3, :3]
    return rotation


def translation_matrix_from_xyz(xyz) -> np.ndarray:
    """Create 4x4 translation matrix from xyz translation vector."""
    matrix = np.eye(4)
    matrix[:3, 3] = xyz
    return matrix


def xyz_from_translation_matrix(matrix: np.ndarray) -> np.ndarray:
    """Extract xyz translation vector from 4x4 translation matrix."""
    return matrix[:3, 3]


def rotation_matrix_from_wxyz(wxyz: np.ndarray) -> np.ndarray:
    """Convert quaternion (w,x,y,z) to 4x4 rotation matrix."""
    w, x, y, z = wxyz

    # Normalize quaternion
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    # Create rotation matrix
    matrix = np.eye(4)
    matrix[0, 0] = 1 - 2 * (y * y + z * z)
    matrix[0, 1] = 2 * (x * y - w * z)
    matrix[0, 2] = 2 * (x * z + w * y)
    matrix[1, 0] = 2 * (x * y + w * z)
    matrix[1, 1] = 1 - 2 * (x * x + z * z)
    matrix[1, 2] = 2 * (y * z - w * x)
    matrix[2, 0] = 2 * (x * z - w * y)
    matrix[2, 1] = 2 * (y * z + w * x)
    matrix[2, 2] = 1 - 2 * (x * x + y * y)

    return matrix


def wxyz_from_rotation_matrix(matrix: np.ndarray) -> np.ndarray:
    """Extract quaternion (w, x, y, z) from a 4x4 or 3x3 rotation matrix.
    Robust near 180° and for slightly non-orthonormal inputs.
    """
    R = np.asarray(matrix, dtype=float)[:3, :3]

    # Re-orthogonalize for numerical stability (optional but helpful)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    trace = np.trace(R)
    q = np.empty(4, dtype=float)

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0  # s = 4 * qw
        q[0] = 0.25 * s
        q[1] = (R[2, 1] - R[1, 2]) / s
        q[2] = (R[0, 2] - R[2, 0]) / s
        q[3] = (R[1, 0] - R[0, 1]) / s
    else:
        # Pick the largest diagonal to avoid division by tiny numbers (handles ~180°)
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0  # s = 4*qx
            q[0] = (R[2, 1] - R[1, 2]) / s
            q[1] = 0.25 * s
            q[2] = (R[0, 1] + R[1, 0]) / s
            q[3] = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0  # s = 4*qy
            q[0] = (R[0, 2] - R[2, 0]) / s
            q[1] = (R[0, 1] + R[1, 0]) / s
            q[2] = 0.25 * s
            q[3] = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0  # s = 4*qz
            q[0] = (R[1, 0] - R[0, 1]) / s
            q[1] = (R[0, 2] + R[2, 0]) / s
            q[2] = (R[1, 2] + R[2, 1]) / s
            q[3] = 0.25 * s

    # Normalize and make the representation canonical (w >= 0)
    q /= np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return q


def wxyz_from_look_at(
    eye: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0, 0, 1])
) -> np.ndarray:
    """Compute quaternion (w,x,y,z) from eye position and look-at target."""
    forward = (target - eye).astype(np.float64)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)

    true_up = np.cross(right, forward)

    rotation_matrix = np.eye(4)
    rotation_matrix[:3, 0] = right
    rotation_matrix[:3, 1] = forward
    rotation_matrix[:3, 2] = true_up

    return wxyz_from_rotation_matrix(rotation_matrix)


def transform_matrix_from_xyz_wxyz(xyz: np.ndarray, wxyz: np.ndarray) -> np.ndarray:
    """Create 4x4 transformation matrix from xyz translation and quaternion (w,x,y,z)."""
    translation = translation_matrix_from_xyz(xyz)
    rotation = rotation_matrix_from_wxyz(wxyz)
    return translation @ rotation


dir_vecs = {
    "R": (1, 0, 0),
    "L": (-1, 0, 0),
    "U": (0, 1, 0),
    "D": (0, -1, 0),
    "F": (0, 0, -1),
    "B": (0, 0, 1),
}


def get_coordinate_RUB_matrix(coordinate_type: str) -> np.ndarray:
    assert (
        len(coordinate_type) == 3
    ), "Coordinate system type should be 3-character-long, e.g. RUB"
    for c in coordinate_type:
        assert (
            c in dir_vecs.keys()
        ), f"Unknown direction {c} - Should be in {dir_vecs.keys()}"
    trans_matrix = [dir_vecs[c] for c in coordinate_type]
    trans_matrix = np.array(trans_matrix).T
    assert np.linalg.det(trans_matrix) != 0, f"{coordinate_type} is invalid."
    full_trans_matrix = np.eye(4, dtype=float)
    full_trans_matrix[:3, :3] = trans_matrix
    return full_trans_matrix


def get_coordinate_trans_matrix(src_type, dest_type="RUB"):
    src_RUB_matrix = get_coordinate_RUB_matrix(src_type)
    dest_RUB_matrix = get_coordinate_RUB_matrix(dest_type)
    trans_matrix = np.linalg.inv(dest_RUB_matrix) @ src_RUB_matrix
    return trans_matrix


def convert_coordinate(src, src_type, dest_type="RUB"):
    src = np.array(src)
    trans_matrix = get_coordinate_trans_matrix(src_type, dest_type)
    return (src.T @ trans_matrix).T

# for unit_test (tests/test_3Dsketch.py) in 3D sketch rendering (360 degree view synthesis)
def _rt_from_lookat_opencv(eye: np.ndarray, lookat: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build OpenCV extrinsics (R, t) from eye/lookat/up.

    Camera axes: x right, y up, z forward (toward scene). Transforms world to camera: Xc = R Xw + t.
    """
    z = lookat - eye
    z = z / (np.linalg.norm(z) + 1e-9)
    x = np.cross(z, up)
    x = x / (np.linalg.norm(x) + 1e-9)
    y = np.cross(x, z)
    R = np.stack([x, y, z], axis=0).astype(np.float32)  # rows
    t = (-R @ eye.astype(np.float32))
    return R, t

# transform to ndc space following the OpenCV convention
from sketchkit.utils3d.camera import intrinsics_from_fov_xy
def project_curves_ndc_opencv(curves: torch.Tensor,
                              cam_position: np.ndarray,
                              cam_lookat: np.ndarray,
                              fov_x: float,
                              width: int,
                              height: int) -> torch.Tensor:
    """Project curves using OpenCV-style K [normalized] and (R,t) from lookat.

    Returns V x N x 4 x 2 in [0,1]. y is up (invert later for Cairo/PIL).
    """
    device = curves.device
    V = cam_position.shape[0]
    N, P = curves.shape[0], curves.shape[1]

    # Compute fov_y from aspect ratio
    aspect = float(height) / float(width)
    fov_y = 2.0 * np.arctan(aspect * np.tan(fov_x / 2.0))
    K = intrinsics_from_fov_xy(fov_x, fov_y)  # normalized to [0,1]

    verts = curves.detach().cpu().numpy().reshape(-1, 3)  # [N*P,3]
    out = np.zeros((V, N, P, 2), dtype=np.float32)

    for vi in range(V):
        eye = cam_position[vi].astype(np.float32)
        tgt = cam_lookat[vi].astype(np.float32)
        R, t = _rt_from_lookat_opencv(eye, tgt, np.array([0.0, 1.0, 0.0], dtype=np.float32))

        # Transform to camera
        Xc = (R @ verts.T + t[:, None]).T  # [N*P,3]
        # Perspective divide
        z = np.clip(Xc[:, 2:3], 1e-6, None)
        x_norm = Xc[:, 0:1] / z
        y_norm = Xc[:, 1:2] / z
        # Apply K (already normalized to [0,1])
        u = K[0, 0] * x_norm + K[0, 2]
        v = K[1, 1] * y_norm + K[1, 2]
        uv = np.concatenate([u, v], axis=1)  # [N*P,2]
        out[vi] = uv.reshape(N, P, 2)

    return torch.from_numpy(out).to(device)