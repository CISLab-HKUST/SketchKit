import pytest
from sketchkit.datasets import QuickDraw, PhotoSketching


@pytest.mark.dataset
@pytest.mark.slow
def test_quickdraw_dataset_loading(tmp_path):
    try:
        dataset = QuickDraw(root=str(tmp_path))
    except Exception as exc:
        pytest.skip(f"QuickDraw dataset unavailable: {exc}")

    assert len(dataset) > 0
    sample = dataset[0]
    assert sample is not None


@pytest.mark.dataset
@pytest.mark.slow
def test_quickdraw_dataset_loading_all(tmp_path):
    try:
        dataset = QuickDraw(root=str(tmp_path), load_all=True)
    except Exception as exc:
        pytest.skip(f"QuickDraw dataset unavailable: {exc}")

    assert len(dataset) > 0
    sample = dataset[0]
    assert sample is not None


@pytest.mark.dataset
@pytest.mark.slow
def test_photosketching_dataset_loading(tmp_path):
    try:
        dataset = PhotoSketching(root=str(tmp_path), cislab_source=True)
    except Exception as exc:
        pytest.skip(f"PhotoSketching dataset unavailable: {exc}")

    assert len(dataset) > 0
    sample = dataset[0]
    assert sample is not None
