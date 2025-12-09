import numpy as np
from typing import overload

from sketchkit.utils3d.camera import intrinsics_from_fov_xy
from sketchkit.utils3d.transforms import (
    yaw_pitch_roll_from_wxyz,
    wxyz_from_yaw_pitch_roll,
    translation_matrix_from_transform,
    rotation_matrix_from_transform,
    translation_matrix_from_xyz,
    xyz_from_translation_matrix,
    rotation_matrix_from_wxyz,
    wxyz_from_rotation_matrix,
    transform_matrix_from_xyz_wxyz,
    wxyz_from_look_at,
    get_coordinate_trans_matrix,
)


# We use COLMAP/OPENCV coordinate conventions


class Object3D:
    def __init__(
        self,
        xyz: tuple[float, float, float] | np.ndarray = np.array([0.0, 0.0, 0.0]),
        wxyz: tuple[float, float, float, float] | np.ndarray = np.array([1.0, 0, 0, 0]),
    ):
        self.xyz = np.array(xyz, dtype=float)
        self.wxyz = np.array(wxyz, dtype=float)
        self._look_at = None

    @classmethod
    def from_transform_matrix(cls, transform_matrix):
        ret = cls()
        ret.translation_matrix = translation_matrix_from_transform(transform_matrix)
        ret.rotation_matrix = rotation_matrix_from_transform(transform_matrix)
        return ret

    @property
    def yaw(self):
        return yaw_pitch_roll_from_wxyz(self.wxyz)[0]

    @yaw.setter
    def yaw(self, value):
        yaw_pitch_roll = np.array(yaw_pitch_roll_from_wxyz(self.wxyz))
        yaw_pitch_roll[0] = value
        self.wxyz = wxyz_from_yaw_pitch_roll(**yaw_pitch_roll)

    @property
    def pitch(self):
        return yaw_pitch_roll_from_wxyz(self.wxyz)[1]

    @pitch.setter
    def pitch(self, value):
        yaw_pitch_roll = np.array(yaw_pitch_roll_from_wxyz(self.wxyz))
        yaw_pitch_roll[1] = value
        self.wxyz = wxyz_from_yaw_pitch_roll(**yaw_pitch_roll)

    @property
    def roll(self):
        return yaw_pitch_roll_from_wxyz(self.wxyz)[2]

    @roll.setter
    def roll(self, value: float):
        yaw_pitch_roll = np.array(yaw_pitch_roll_from_wxyz(self.wxyz))
        yaw_pitch_roll[2] = value
        self.wxyz = wxyz_from_yaw_pitch_roll(**yaw_pitch_roll)

    @property
    def translation_matrix(self):
        return translation_matrix_from_xyz(self.xyz)

    @translation_matrix.setter
    def translation_matrix(self, value):
        self.xyz = xyz_from_translation_matrix(value)

    @property
    def rotation_matrix(self):
        return rotation_matrix_from_wxyz(self.wxyz)

    @rotation_matrix.setter
    def rotation_matrix(self, value):
        self.wxyz = wxyz_from_rotation_matrix(value)

    @property
    def transform_matrix(self):
        return transform_matrix_from_xyz_wxyz(self.xyz, self.wxyz)

    def set_look_at(self, look_at: list[float] | np.ndarray):
        if isinstance(look_at, list) and len(look_at) != 3:
            raise ValueError("look_at list must have exactly 3 elements")
        self._look_at = np.array(look_at, dtype=float)

    def set_xyz(self, xyz: tuple[float, float, float] | np.ndarray):
        self.xyz = np.array(xyz, dtype=float)
        if self._look_at is not None:
            self.wxyz = wxyz_from_look_at(self.xyz, self._look_at)

    def set_wxyz(self, wxyz: tuple[float, float, float, float] | np.ndarray):
        self.wxyz = wxyz
        self._look_at = None


class Camera(Object3D):
    def __init__(
        self,
        xyz: tuple[float, float, float] | np.ndarray = np.array([0, 0, 0]),
        wxyz: tuple[float, float, float, float] | np.ndarray = np.array([1, 0, 0, 0]),
        fov_x: float = 40,
        fov_y: float = 40,
        z_far: float = 1000,
        z_near: float = 0.1,
        coordinate_type="RDF",
    ):
        super().__init__(xyz, wxyz)
        self.fov_x = fov_x
        self.fov_y = fov_y
        self.z_far = z_far
        self.z_near = z_near
        self.coordinate_type = coordinate_type

    @property
    def intrinsics(self):
        return intrinsics_from_fov_xy(self.fov_x, self.fov_y)

    @intrinsics.setter
    def intrinsics(self, value):
        pass

    @property
    def extrinsics(self):
        coordinate_trans_matrix = get_coordinate_trans_matrix(
            "RFU", self.coordinate_type
        )

        c2w_matrix = self.transform_matrix @ coordinate_trans_matrix.T 
        return np.linalg.inv(c2w_matrix)
 
    @extrinsics.setter
    def extrinsics(self, value):
        pass

    @overload
    def set_fov(self, *, fov: float) -> None: ...

    @overload
    def set_fov(self, *, fov_x: float, fov_y: float) -> None: ...

    def set_fov(
        self,
        *,
        fov: None | float = None,
        fov_x: None | float = None,
        fov_y: None | float = None,
    ):
        if fov is not None:
            self.fov_x = fov
            self.fov_y = fov
        else:
            self.fov_x = fov_x
            self.fov_y = fov_y
