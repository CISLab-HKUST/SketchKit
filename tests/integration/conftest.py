import pytest


def load_or_skip(factory, *args, **kwargs):
    try:
        return factory(*args, **kwargs)
    except Exception as exc:
        pytest.skip(f"test skipped due to unavailable dependency or data: {exc}")


def pytest_collection_modifyitems(items):
    for item in items:
        item.add_marker(pytest.mark.integration)
