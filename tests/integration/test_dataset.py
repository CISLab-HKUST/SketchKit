import pytest
from sketchkit.datasets import QuickDraw, PhotoSketching


@pytest.mark.dataset
@pytest.mark.slow
def test_quickdraw_dataset_loading(tmp_path):
    try:
        dataset = QuickDraw()
    except Exception as exc:
        pytest.skip(f"QuickDraw dataset unavailable: {exc}")

    assert len(dataset) > 0
    sample = dataset[0]
    assert sample is not None


@pytest.mark.dataset
@pytest.mark.slow
def test_quickdraw_dataset_loading_all(tmp_path):
    try:
        dataset = QuickDraw(load_all=True)
    except Exception as exc:
        pytest.skip(f"QuickDraw dataset unavailable: {exc}")

    assert len(dataset) > 0
    sample = dataset[0]
    assert sample is not None


@pytest.mark.dataset
@pytest.mark.slow
def test_photosketching_dataset_loading(tmp_path):
    try:
        dataset = PhotoSketching(cislab_source=True)
    except Exception as exc:
        pytest.skip(f"PhotoSketching dataset unavailable: {exc}")

    assert len(dataset) > 0
    sample = dataset[0]
    assert sample is not None
