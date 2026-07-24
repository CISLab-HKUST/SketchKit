import numpy as np
from sketchkit.core.camera import Object3D


def test_object3d_initialization():
    obj = Object3D()
    np.testing.assert_array_equal(obj.xyz, np.array([0.0, 0.0, 0.0]))
    np.testing.assert_array_equal(obj.wxyz, np.array([1.0, 0.0, 0.0, 0.0]))


def test_object3d_custom_initialization():
    xyz = [1.0, 2.0, 3.0]
    wxyz = [0.0, 1.0, 0.0, 0.0]
    obj = Object3D(xyz=xyz, wxyz=wxyz)
    np.testing.assert_array_equal(obj.xyz, np.array(xyz))
    np.testing.assert_array_equal(obj.wxyz, np.array(wxyz))


def test_object3d_set_xyz():
    obj = Object3D()
    obj._look_at = None
    obj.set_xyz([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(obj.xyz, np.array([1.0, 2.0, 3.0]))


def test_object3d_set_wxyz():
    obj = Object3D()
    obj.set_wxyz([0.0, 1.0, 0.0, 0.0])
    np.testing.assert_array_equal(obj.wxyz, np.array([0.0, 1.0, 0.0, 0.0]))
