from types import SimpleNamespace
import subprocess
import numpy as np
import pytest
from sketchkit.renderer.npr_renderer import NPRRenderer


def renderer():
    obj = NPRRenderer.__new__(NPRRenderer)
    obj.renderer_path = "/tmp/path with spaces/renderer"
    obj.obj_path = "/tmp/model with spaces.obj"
    return obj


def test_renderer_passes_paths_as_arguments(tmp_path, monkeypatch):
    output = tmp_path / "frame with spaces.ppm"
    def run(args, **kwargs):
        assert args[:3] == [renderer().renderer_path, renderer().obj_path, "-o"]
        assert str(output) in args
        assert kwargs["check"] is True
        output.write_bytes(b"P6\n1 1\n255\n\xff\xff\xff")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(subprocess, "run", run)
    renderer().render_suggestive_countour(str(output), np.eye(4))


def test_renderer_propagates_native_failure(tmp_path, monkeypatch):
    def run(args, **kwargs):
        raise subprocess.CalledProcessError(1, args, stderr="Failed to initialize GLFW")
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(RuntimeError, match="Failed to initialize GLFW"):
        renderer().render_suggestive_countour(str(tmp_path / "out.ppm"), np.eye(4))


def test_renderer_rejects_missing_output(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    with pytest.raises(RuntimeError, match="produced no image"):
        renderer().render_ridges_and_valleys(str(tmp_path / "out.ppm"), np.eye(4))
