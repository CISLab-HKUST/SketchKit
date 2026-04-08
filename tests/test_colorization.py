import pytest
import os
import sys
import traceback
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sketchkit.datasets import QuickDraw
from sketchkit.colorization import Colorizer
from sketchkit.renderer.cairo_renderer import CairoRenderer, CairoRenderOptions

TEST_RESULTS = {}

def convert_sketch_to_image(sketch, size=(1024, 1024)):
    """Convert sketch to image for testing setup."""
    original_w, original_h = sketch.width, sketch.height
    sketch.resize(size[0], size[1])
    
    render_options = CairoRenderOptions(
        canvas_size=size,
        background_color=(1, 1, 1),
        stroke_width=3
    )
    renderer = CairoRenderer(render_options=render_options)
    pil_image = renderer.render(sketch, render_options=render_options)
    sketch.resize(original_w, original_h)
    return pil_image

@pytest.fixture(scope="module")
def sample_data():
    """Fixture to load QuickDraw dataset and prepare common test inputs."""
    print("\n[Fixture] Loading QuickDraw dataset...")
    dataset = QuickDraw()
    sketch = dataset[150]
    sketch_image = convert_sketch_to_image(sketch, size=(1024, 1024))
    return sketch, sketch_image

def run_and_save(colorizer, input_data, method_name, test_type, prompt, input_size=None, **kwargs):
    try:
        print(f"\nTesting {method_name} with {test_type} input...")
        img = colorizer.run(input_data, prompt=prompt, size=1024, input_size=input_size, **kwargs)
        filename = f"output_colorization_{method_name}_{test_type}.png"
        img.save(filename)
        print(f"✅ {method_name} with {test_type} succeeded! Saved to {filename}")
        return True
    except Exception as e:
        print(f"❌ {method_name} with {test_type} failed: {str(e)}")
        traceback.print_exc()
        return False

COLORIZATION_TEST_PARAMS = [
    (
        'controlnet_lineart', 
        "cagliostrolab/animagine-xl-3.1", 
        "1girl, solo, cute cat, masterpiece, best quality, highly detailed, vibrant colors, flat shading, anime illustration", 
        {'num_inference_steps': 15, 'guidance_scale': 7.5} 
    ),
    (
        'controlnet_scribble', 
        "Lykon/dreamshaper-xl-1-0", 
        "colorful cat illustration, digital painting, thick impasto, beautiful lighting", 
        {'num_inference_steps': 15, 'guidance_scale': 7.5}
    )
]

@pytest.mark.parametrize("method, sd_model, prompt, kwargs", COLORIZATION_TEST_PARAMS)
def test_colorizer_methods(sample_data, method, sd_model, prompt, kwargs):
    """Test all modes of Colorizer with given models."""
    sketch, sketch_image = sample_data
    TEST_RESULTS[method] = {}

    try:
        colorizer = Colorizer(method=method, sd_model=sd_model)
    except Exception as e:
        print(f"❌ Failed to initialize {method}: {str(e)}")
        TEST_RESULTS[method] = {'sketch': False, 'sketch_resized': False, 'image': False, 'image_resized': False}
        pytest.fail(f"Initialization failed for {method}")

    r1 = run_and_save(colorizer, sketch, method, "sketch", prompt, input_size=None, **kwargs)
    r2 = run_and_save(colorizer, sketch, method, "sketch_resized", prompt, input_size=(768, 768), **kwargs)
    r3 = run_and_save(colorizer, sketch_image, method, "image", prompt, input_size=None, **kwargs)
    r4 = run_and_save(colorizer, sketch_image, method, "image_resized", prompt, input_size=(768, 768), **kwargs)

    # 记录结果
    TEST_RESULTS[method] = {
        'sketch': r1,
        'sketch_resized': r2,
        'image': r3,
        'image_resized': r4
    }

    assert all([r1, r2, r3, r4]), f"Some test modes failed for {method}."

def teardown_module(module):
    summary = ["\n" + "="*50, "SUMMARY REPORT (Colorization)", "="*50]
    
    for method, results in TEST_RESULTS.items():
        summary.append(f"\n{method}:")
        for test_type, success in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            summary.append(f"  {test_type}: {status}")
            
    summary_str = "\n".join(summary)
    print(summary_str)
    
    report_path = os.path.join(os.path.dirname(__file__), "summary_colorization.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(summary_str)
    print(f"\n📝 Test summary successfully saved to: {report_path}")

if __name__ == "__main__":
    pytest.main(["-s", "-v", __file__])