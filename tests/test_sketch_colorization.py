"""
Test sketch colorization functionality
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sketchkit.datasets import QuickDraw
from sketchkit.sketch_colorization import colorize_sketch, SketchColorizationConverter, convert_sketch, available_methods
from PIL import Image
import traceback


def test_method_safely(sketch, method, prompt, **kwargs):
    """Test a method safely with error handling"""
    try:
        print(f"\nTesting {method} method...")
        # 使用新的统一API
        image = convert_sketch(
            sketch, 
            method=method,
            prompt=prompt,
            size=512,
            **kwargs
        )
        
        # Save the result
        filename = f"output_colorization_{method}.png"
        image.save(filename)
        print(f"✅ {method} succeeded! Saved to {filename}")
        return True
        
    except Exception as e:
        print(f"❌ {method} failed: {str(e)}")
        traceback.print_exc()
        return False


def test_backward_compatibility(sketch):
    """Test backward compatibility with old API"""
    try:
        print("\nTesting backward compatibility (colorize_sketch)...")
        image = colorize_sketch(
            sketch,
            method="controlnet_lineart",
            prompt="vibrant anime cat",
            size=512,
            output_path="output_colorization_backward.png"
        )
        print("✅ Backward compatibility works!")
        return True
    except Exception as e:
        print(f"❌ Backward compatibility failed: {e}")
        return False

def test_magiccolor_with_references(sketch):
    """Test MagicColor with reference images"""
    try:
        reference_images = [
            Image.new('RGB', (256, 256), color=(255, 150, 150)),  # Pink
            Image.new('RGB', (256, 256), color=(150, 255, 150)),  # Green
        ]
        
        print("\nTesting magiccolor with reference images...")
        image = convert_sketch(
            sketch,
            prompt="consistent coloring",
            method="magiccolor",
            size=512,
            reference_images=reference_images
        )
        
        image.save("output_magiccolor_ref.png")
        print("✅ MagicColor with references succeeded!")
        return True
    except Exception as e:
        print(f"❌ MagicColor failed: {e}")
        return False


def main():
    # Load dataset
    print("Loading QuickDraw dataset...")
    dataset = QuickDraw()
    sketch = dataset[100]  # Cat sketch
    
    print(f"Loaded sketch with {len(sketch.paths)} paths")
    print(f"Available colorization methods: {available_methods()}")
    
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
        if method in available_methods():
            success = test_method_safely(sketch, method, prompt, **kwargs)
            if success:
                successful_methods.append(method)
            else:
                failed_methods.append(method)
        else:
            print(f"⚠️  Method '{method}' not available, skipping")
            failed_methods.append(method)
    
    # Test MagicColor if available
    if "magiccolor" in available_methods():
        test_magiccolor_with_references(sketch)
    
    # Test backward compatibility
    test_backward_compatibility(sketch)
    
    # Print summary
    print(f"\n📊 Summary:")
    print(f"✅ Successful methods: {successful_methods}")
    print(f"❌ Failed methods: {failed_methods}")
    
    if not successful_methods:
        print("\n⚠️  All methods failed. This might be due to:")
        print("   - Missing dependencies (diffusers, transformers, torch)")
        print("   - Insufficient GPU memory")
        print("   - Network issues downloading models")
        print("   - Model compatibility issues")
        print("\n💡 Try installing missing dependencies:")
        print("   pip install diffusers transformers accelerate torch torchvision")
        print("   pip install opencv-python")


if __name__ == "__main__":
    main()