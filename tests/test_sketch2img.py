import sys
import os
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sketchkit.sketch2image import ImageGenerator
from PIL import Image
import numpy as np
import traceback


def create_test_image(size=(512, 512)):
    """Create a simple test image for testing."""
    img = Image.new('RGB', (size[1], size[0]), color=(255, 255, 255))
    return img


def test_sketch_input(generator, sketch_factory, method_name, prompt, input_size=None, **kwargs):
    """Test with sketch input."""
    try:
        print(f"\nTesting {method_name} with Sketch input...")
        
        sketch = sketch_factory()
        
        image = generator.run(
            sketch, 
            prompt=prompt, 
            size=512,
            input_size=input_size,
            **kwargs
        )
        
        filename = f"output_sketch2img_{method_name}_sketch.png"
        image.save(filename)
        print(f"✅ {method_name} with Sketch succeeded! Saved to {filename}")
        return True
        
    except Exception as e:
        print(f"❌ {method_name} with Sketch failed: {str(e)}")
        traceback.print_exc()
        return False


def test_image_input(generator, image, method_name, prompt, input_size=None, **kwargs):
    """Test with image input."""
    try:
        print(f"\nTesting {method_name} with Image input...")
        image_output = generator.run(
            image, 
            prompt=prompt, 
            size=512,
            input_size=input_size,
            **kwargs
        )
        
        filename = f"output_sketch2img_{method_name}_image.png"
        image_output.save(filename)
        print(f"✅ {method_name} with Image succeeded! Saved to {filename}")
        return True
        
    except Exception as e:
        print(f"❌ {method_name} with Image failed: {str(e)}")
        traceback.print_exc()
        return False


def create_test_sketch():
    """Create a simple test sketch since we can't load the datasets."""
    from sketchkit.core.sketch import Sketch, Path, Curve, Vertex, Point
    
    # Create a simple house sketch
    paths = []
    
    # House base (square)
    curves = []
    curves.append(Curve.from_line([[100, 200], [200, 200]]))
    curves.append(Curve.from_line([[200, 200], [200, 100]]))
    curves.append(Curve.from_line([[200, 100], [100, 100]]))
    curves.append(Curve.from_line([[100, 100], [100, 200]]))
    paths.append(Path(curves))
    
    # Roof (triangle)
    curves = []
    curves.append(Curve.from_line([[100, 100], [150, 50]]))
    curves.append(Curve.from_line([[150, 50], [200, 100]]))
    paths.append(Path(curves))
    
    # Door (rectangle)
    curves = []
    curves.append(Curve.from_line([[135, 200], [165, 200]]))
    curves.append(Curve.from_line([[165, 200], [165, 160]]))
    curves.append(Curve.from_line([[165, 160], [135, 160]]))
    curves.append(Curve.from_line([[135, 160], [135, 200]]))
    paths.append(Path(curves))
    
    # Windows (two squares)
    # Left window
    curves = []
    curves.append(Curve.from_line([[115, 120], [130, 120]]))
    curves.append(Curve.from_line([[130, 120], [130, 135]]))
    curves.append(Curve.from_line([[130, 135], [115, 135]]))
    curves.append(Curve.from_line([[115, 135], [115, 120]]))
    paths.append(Path(curves))
    
    # Right window
    curves = []
    curves.append(Curve.from_line([[170, 120], [185, 120]]))
    curves.append(Curve.from_line([[185, 120], [185, 135]]))
    curves.append(Curve.from_line([[185, 135], [170, 135]]))
    curves.append(Curve.from_line([[170, 135], [170, 120]]))
    paths.append(Path(curves))
    
    return Sketch(height=256, width=256, paths=paths)


def main():
    # create test image
    test_img = create_test_image()
    
    methods_to_test = [
        ('controlnet', "a realistic house in the countryside", 
        {'num_inference_steps': 20, 'guidance_scale': 7.5, 'controlnet_conditioning_scale': 1.0}),
        ('t2i_adapter', "a beautiful house illustration", 
        {'num_inference_steps': 30, 'guidance_scale': 7.5, 'adapter_conditioning_scale': 1.0}),
        ('pix2pix', "a realistic house photo", 
        {'num_inference_steps': 10, 'guidance_scale': 7.5, 'image_guidance_scale': 1.0})
    ]
    
    all_results = {}
    
    for method, prompt, kwargs in methods_to_test:
        print(f"\n{'='*50}")
        print(f"Testing {method} method")
        print(f"{'='*50}")
        
        method_results = {}
        
        try:
            generator = ImageGenerator(method=method)
            
            # Test with sketch input (no resize) 
            success_sketch = test_sketch_input(
                generator, create_test_sketch, method, 
                prompt, None, **kwargs
            )
            
            # Test with sketch input (with resize) 
            success_sketch_resized = test_sketch_input(
                generator, create_test_sketch, f"{method}_resized", 
                prompt, (128, 128), **kwargs
            )
            
            # Test with image input (no resize)
            success_image = test_image_input(
                generator, test_img, f"{method}_image", 
                prompt, None, **kwargs
            )
            
            # Test with image input (with resize)
            success_image_resized = test_image_input(
                generator, test_img, f"{method}_image_resized", 
                prompt, (256, 256), **kwargs
            )
            
            method_results = {
                'sketch': success_sketch,
                'sketch_resized': success_sketch_resized,
                'image': success_image,
                'image_resized': success_image_resized
            }
            
        except Exception as e:
            print(f"❌ Failed to initialize {method}: {str(e)}")
            method_results = {
                'sketch': False,
                'sketch_resized': False,
                'image': False,
                'image_resized': False
            }
        
        all_results[method] = method_results
    
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