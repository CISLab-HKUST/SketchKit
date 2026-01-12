"""
Test script for the new Stylizer wrapper API.

This script tests the unified Stylizer wrapper class that supports multiple stylization methods.
"""

import sys
import os
from pathlib import Path
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sketchkit.stylization import Stylizer
from sketchkit.datasets import PhotoSketching

"""
Stylizer parameters (passed via kwargs):
 - method: Stylization method (currently only "brushstrokesengine")
 - model_name: Model name ('style1' or 'style2') for brushstrokesengine
 - style_id: Style ID (e.g., "playdoh10") for brushstrokesengine
 - feature_blending_level: Feature blending level (default 2)
 - color_mode: Color mode ('1'/'2' for library colors, or "r,g,b;..." for custom)
 - crop_margin: Crop margin (default 10)
 - on_white: Render on white background (default True)
 - render_mode: Render mode (default "clear")
 - stitching_mode: Stitching mode (default "all")
 - canvas_size: Canvas size for rendering Sketch (None uses sketch's actual dimensions)
 - library_path: Optional brush library path
 - encoder_checkpoint: Optional encoder checkpoint path
 - device: Device to use ('cpu'/'cuda', default 'cuda')
 - auto_download: Automatically download models if not found (default True)
"""


def test_stylizer_from_png(png_path: Path, output_dir: Path, **kwargs) -> bool:
    """Test stylization from an input PNG using Stylizer wrapper."""
    try:
        print("\nTesting Stylizer from PNG...")
        stylizer = Stylizer(
            method="brushstrokesengine",
            model_name=kwargs.get("model_name", "style2"),
            style_id=kwargs.get("style_id", "playdoh10"),
            auto_download=True,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"stylizer_png_{stylizer.model_name}_{stylizer.style_id}.png"

        stylizer.run_from_png(
            png_path,
            output_path=out_path,
            feature_blending_level=kwargs.get("feature_blending_level", 2),
            color_mode=kwargs.get("color_mode", "1"),
            crop_margin=kwargs.get("crop_margin", 10),
            on_white=kwargs.get("on_white", True),
        )

        print(f"✅ PNG stylization succeeded! Saved to {out_path}")
        return True

    except Exception as e:
        print(f"❌ PNG stylization failed: {e}")
        traceback.print_exc()
        return False


def test_stylizer_from_sketch(sketch, output_dir: Path, **kwargs) -> bool:
    """Test stylization from a Sketch object using Stylizer wrapper."""
    try:
        print("\nTesting Stylizer from Sketch...")
        stylizer = Stylizer(
            method="brushstrokesengine",
            model_name=kwargs.get("model_name", "style2"),
            style_id=kwargs.get("style_id", "playdoh10"),
            auto_download=True,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"stylizer_sketch_{stylizer.model_name}_{stylizer.style_id}.png"

        stylizer.run(
            sketch,
            output_path=out_path,
            canvas_size=kwargs.get("canvas_size", None),  # None means use sketch's actual dimensions
            feature_blending_level=kwargs.get("feature_blending_level", 2),
            color_mode=kwargs.get("color_mode", "1"),
            crop_margin=kwargs.get("crop_margin", 10),
            on_white=kwargs.get("on_white", True),
        )

        print(f"✅ Sketch stylization succeeded! Saved to {out_path}")
        return True

    except Exception as e:
        print(f"❌ Sketch stylization failed: {e}")
        traceback.print_exc()
        return False


def test_stylizer_initialization():
    """Test Stylizer initialization with different parameters."""
    try:
        print("\nTesting Stylizer initialization...")
        
        # Test default initialization
        stylizer1 = Stylizer(
            method="brushstrokesengine",
            auto_download=False,  # Don't download for initialization test
        )
        assert stylizer1.method == "brushstrokesengine", "Method should be brushstrokesengine"
        print("   ✅ Default initialization successful")
        
        # Test with custom parameters
        stylizer2 = Stylizer(
            method="brushstrokesengine",
            model_name="style1",
            style_id="playdoh05",
            auto_download=False,
        )
        assert stylizer2.method == "brushstrokesengine", "Method should be brushstrokesengine"
        assert stylizer2.model_name == "style1", "Model name should be style1"
        assert stylizer2.style_id == "playdoh05", "Style ID should be playdoh05"
        print("   ✅ Custom parameter initialization successful")
        
        # Test invalid method
        try:
            stylizer3 = Stylizer(method="invalid_method", auto_download=False)
            print("   ❌ Should have raised ValueError for invalid method")
            return False
        except ValueError:
            print("   ✅ Invalid method correctly raised ValueError")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Initialization test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    print("=" * 60)
    print("Stylizer Wrapper API Test")
    print("=" * 60)
    
    successes = []
    failures = []
    
    # Test initialization
    if test_stylizer_initialization():
        successes.append("initialization")
    else:
        failures.append("initialization")
    
    # Load a sample sketch for functional tests
    print("\n" + "=" * 60)
    print("Functional Tests")
    print("=" * 60)
    
    print("\nLoading PhotoSketching dataset...")
    try:
        dataset = PhotoSketching()
        sketch = dataset[5]
        print(f"Loaded sketch with {len(sketch.paths)} paths")
    except Exception as e:
        print(f"⚠️  Could not load dataset: {e}")
        print("Skipping functional tests that require dataset")
        sketch = None
    
    # Prepare output directory and sample PNG
    output_dir = Path("tmp_stylizer_test")
    png_path = Path("input/0-0.png")
    
    if png_path.exists() and sketch is not None:
        # Test from PNG
        if test_stylizer_from_png(png_path, output_dir):
            successes.append("from_png")
        else:
            failures.append("from_png")
        
        # Test from Sketch
        if test_stylizer_from_sketch(sketch, output_dir):
            successes.append("from_sketch")
        else:
            failures.append("from_sketch")
    else:
        if not png_path.exists():
            print(f"⚠️  Sample PNG not found at {png_path}, skipping PNG test")
        if sketch is None:
            print("⚠️  Could not load sketch, skipping Sketch test")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    print(f"✅ Successful: {successes}")
    if failures:
        print(f"❌ Failed: {failures}")
    
    if len(failures) == 0:
        print("\n✅ All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {len(failures)} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

