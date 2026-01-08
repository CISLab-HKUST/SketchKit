import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sketchkit.datasets import QuickDraw
from sketchkit.colorization import Colorizer
import traceback


def test_method_safely(sketch, method, prompt, **kwargs):
    """Test a method safely with error handling"""
    try:
        print(f"\nTesting {method} method...")
        colorizer = Colorizer(method=method)
        image = colorizer.run(sketch, prompt=prompt, size=512, **kwargs)
        
        # Save the result
        filename = f"output_colorization_{method}.png"
        image.save(filename)
        print(f"✅ {method} succeeded! Saved to {filename}")
        return True
        
    except Exception as e:
        print(f"❌ {method} failed: {str(e)}")
        traceback.print_exc()
        return False


def main():
    # Load dataset
    print("Loading QuickDraw dataset...")
    dataset = QuickDraw()
    sketch = dataset[100]  # Cat sketch
    
    print(f"Loaded sketch with {len(sketch.paths)} paths")
    
    # Test different methods
    methods_to_test = [
        ('controlnet_lineart', "vibrant anime cat with bright colors", 
        {'num_inference_steps': 20, 'guidance_scale': 7.5}),
        ('controlnet_scribble', "colorful cat illustration", 
        {'num_inference_steps': 20, 'guidance_scale': 7.5}),
    ]
    
    successful_methods = []
    failed_methods = []
    
    for method, prompt, kwargs in methods_to_test:
        success = test_method_safely(sketch, method, prompt, **kwargs)
        if success:
            successful_methods.append(method)
        else:
            failed_methods.append(method)
    
    # Print summary
    print(f"\n📊 Summary:")
    print(f"✅ Successful methods: {successful_methods}")
    print(f"❌ Failed methods: {failed_methods}")


if __name__ == "__main__":
    main()