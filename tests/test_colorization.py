import sys
import os
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sketchkit.colorization import Colorizer
from PIL import Image
import numpy as np
import traceback


def create_test_image(size=(512, 512)):
    """Create a simple test image for testing."""
    img = Image.new('RGB', (size[1], size[0]), color=(255, 255, 255))
    return img


def test_sketch_input(colorizer, sketch_factory, method_name, prompt, input_size=None, **kwargs):
    """Test with sketch input."""
    try:
        print(f"\nTesting {method_name} with Sketch input...")

        sketch = sketch_factory()
        
        image = colorizer.run(
            sketch, 
            prompt=prompt, 
            size=512,
            input_size=input_size,
            **kwargs
        )
        
        filename = f"output_colorization_{method_name}_sketch.png"
        image.save(filename)
        print(f"✅ {method_name} with Sketch succeeded! Saved to {filename}")
        return True
        
    except Exception as e:
        print(f"❌ {method_name} with Sketch failed: {str(e)}")
        traceback.print_exc()
        return False


def test_image_input(colorizer, image, method_name, prompt, input_size=None, **kwargs):
    """Test with image input."""
    try:
        print(f"\nTesting {method_name} with Image input...")
        image_output = colorizer.run(
            image, 
            prompt=prompt, 
            size=512,
            input_size=input_size,
            **kwargs
        )
        
        filename = f"output_colorization_{method_name}_image.png"
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
    
    # Create a simple cat face sketch
    paths = []
    
    # Head (circle)
    curves = []
    # Simplified circle with 4 cubic curves
    # This is just a placeholder - in reality you'd use proper circle approximation
    curves.append(Curve(
        p_start=Vertex(100, 100),
        p_end=Vertex(150, 100),
        p_crtl1=Point(115, 100),
        p_crtl2=Point(135, 100)
    ))
    curves.append(Curve(
        p_start=Vertex(150, 100),
        p_end=Vertex(150, 150),
        p_crtl1=Point(150, 115),
        p_crtl2=Point(150, 135)
    ))
    curves.append(Curve(
        p_start=Vertex(150, 150),
        p_end=Vertex(100, 150),
        p_crtl1=Point(135, 150),
        p_crtl2=Point(115, 150)
    ))
    curves.append(Curve(
        p_start=Vertex(100, 150),
        p_end=Vertex(100, 100),
        p_crtl1=Point(100, 135),
        p_crtl2=Point(100, 115)
    ))
    paths.append(Path(curves))
    
    # Ears (two triangles)
    # Left ear
    curves = []
    curves.append(Curve.from_line([[100, 100], [90, 70]]))
    curves.append(Curve.from_line([[90, 70], [110, 80]]))
    curves.append(Curve.from_line([[110, 80], [100, 100]]))
    paths.append(Path(curves))
    
    # Right ear
    curves = []
    curves.append(Curve.from_line([[150, 100], [160, 70]]))
    curves.append(Curve.from_line([[160, 70], [140, 80]]))
    curves.append(Curve.from_line([[140, 80], [150, 100]]))
    paths.append(Path(curves))
    
    # Eyes (two dots)
    curves = []
    curves.append(Curve.from_line([[115, 120], [115, 120]]))
    paths.append(Path(curves))
    
    curves = []
    curves.append(Curve.from_line([[135, 120], [135, 120]]))
    paths.append(Path(curves))
    
    # Nose (triangle)
    curves = []
    curves.append(Curve.from_line([[125, 130], [120, 140]]))
    curves.append(Curve.from_line([[120, 140], [130, 140]]))
    curves.append(Curve.from_line([[130, 140], [125, 130]]))
    paths.append(Path(curves))
    
    # Mouth (smile)
    curves = []
    curves.append(Curve(
        p_start=Vertex(120, 145),
        p_end=Vertex(130, 145),
        p_crtl1=Point(122, 150),
        p_crtl2=Point(128, 150)
    ))
    paths.append(Path(curves))
    
    return Sketch(height=256, width=256, paths=paths)


def main():
    # create test image
    test_img = create_test_image()
    
    methods_to_test = [
        ('controlnet_lineart', "vibrant anime style with bright colors", 
        {'num_inference_steps': 20, 'guidance_scale': 7.5}),
        ('controlnet_scribble', "colorful illustration", 
        {'num_inference_steps': 20, 'guidance_scale': 7.5}),
    ]
    
    all_results = {}
    
    for method, prompt_suffix, kwargs in methods_to_test:
        print(f"\n{'='*50}")
        print(f"Testing {method} method")
        print(f"{'='*50}")
        
        method_results = {}
        
        try:
            colorizer = Colorizer(method=method)
            
            # Test with sketch input (no resize)
            success_sketch = test_sketch_input(
                colorizer, create_test_sketch, method, 
                prompt_suffix, None, **kwargs
            )
            
            # Test with sketch input (with resize) 
            success_sketch_resized = test_sketch_input(
                colorizer, create_test_sketch, f"{method}_resized", 
                prompt_suffix, (128, 128), **kwargs
            )
            
            # Test with image input (no resize)
            success_image = test_image_input(
                colorizer, test_img, f"{method}_image", 
                prompt_suffix, None, **kwargs
            )
            
            # Test with image input (with resize)
            success_image_resized = test_image_input(
                colorizer, test_img, f"{method}_image_resized", 
                prompt_suffix, (256, 256), **kwargs
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