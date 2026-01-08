import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sketchkit.datasets import QuickDraw
from sketchkit.colorization import Colorizer
from PIL import Image
import numpy as np
import traceback


def test_with_sketch_input(colorizer, sketch, method_name, prompt, input_size=None, **kwargs):
    """Test colorization with sketch input."""
    try:
        print(f"\nTesting {method_name} with Sketch input...")
        
        # Run colorization
        image = colorizer.run(
            sketch, 
            prompt=prompt, 
            size=512,
            input_size=input_size,
            **kwargs
        )
        
        # Save output
        filename = f"output_colorization_{method_name}_sketch.png"
        image.save(filename)
        print(f"✅ {method_name} with Sketch succeeded! Saved to {filename}")
        return True
        
    except Exception as e:
        print(f"❌ {method_name} with Sketch failed: {str(e)}")
        traceback.print_exc()
        return False


def test_with_image_input(colorizer, image, method_name, prompt, input_size=None, **kwargs):
    """Test colorization with image input."""
    try:
        print(f"\nTesting {method_name} with Image input...")
        
        # Run colorization
        image_output = colorizer.run(
            image, 
            prompt=prompt, 
            size=512,
            input_size=input_size,
            **kwargs
        )
        
        # Save output
        filename = f"output_colorization_{method_name}_image.png"
        image_output.save(filename)
        print(f"✅ {method_name} with Image succeeded! Saved to {filename}")
        return True
        
    except Exception as e:
        print(f"❌ {method_name} with Image failed: {str(e)}")
        traceback.print_exc()
        return False


def convert_sketch_to_image(sketch, size=(512, 512)):
    """Convert sketch to image for testing."""
    from sketchkit.renderer.cairo_renderer import CairoRenderer, CairoRenderOptions
    
    render_options = CairoRenderOptions(
        canvas_size=size,
        background_color=(1, 1, 1),
        stroke_width=2
    )
    
    renderer = CairoRenderer(render_options=render_options)
    pil_image = renderer.render(sketch, render_options=render_options)
    return pil_image


def main():
    # Load QuickDraw dataset
    print("Loading QuickDraw dataset...")
    dataset = QuickDraw()
    
    # Get a cat sketch (index 100 as in original test)
    sketch = dataset[100]
    print(f"Loaded sketch: height={sketch.height}, width={sketch.width}, paths={len(sketch.paths)}")
    
    # Convert sketch to image for testing image input
    sketch_image = convert_sketch_to_image(sketch, size=(512, 512))
    
    # Test different colorization methods
    methods_to_test = [
        ('controlnet_lineart', "vibrant anime cat with bright colors", 
        {'num_inference_steps': 20, 'guidance_scale': 7.5}),
        ('controlnet_scribble', "colorful cat illustration", 
        {'num_inference_steps': 20, 'guidance_scale': 7.5}),
    ]
    
    all_results = {}
    
    for method, prompt, kwargs in methods_to_test:
        print(f"\n{'='*50}")
        print(f"Testing {method} method")
        print(f"{'='*50}")
        
        try:
            colorizer = Colorizer(method=method)
            
            # Test with sketch input (no resize)
            success_sketch = test_with_sketch_input(
                colorizer, sketch, method, 
                prompt, None, **kwargs
            )
            
            # Test with sketch input (with resize)
            success_sketch_resized = test_with_sketch_input(
                colorizer, sketch, f"{method}_resized", 
                prompt, (256, 256), **kwargs
            )
            
            # Test with image input (no resize)
            success_image = test_with_image_input(
                colorizer, sketch_image, f"{method}_image", 
                prompt, None, **kwargs
            )
            
            # Test with image input (with resize)
            success_image_resized = test_with_image_input(
                colorizer, sketch_image, f"{method}_image_resized", 
                prompt, (128, 128), **kwargs
            )
            
            all_results[method] = {
                'sketch': success_sketch,
                'sketch_resized': success_sketch_resized,
                'image': success_image,
                'image_resized': success_image_resized
            }
            
        except Exception as e:
            print(f"❌ Failed to initialize {method}: {str(e)}")
            all_results[method] = {
                'sketch': False,
                'sketch_resized': False,
                'image': False,
                'image_resized': False
            }
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    
    for method, results in all_results.items():
        print(f"\n{method}:")
        for test_type, success in results.items():
            status = "✅" if success else "❌"
            print(f"  {test_type}: {status}")


if __name__ == "__main__":
    main()