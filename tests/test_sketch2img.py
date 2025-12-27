# tests/test_sketch2img.py
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# test_safe.py
from sketchkit.datasets import QuickDraw
from sketchkit.sketch2image import sketch2img, Sketch2ImageConverter, convert_sketch, available_methods
import traceback

def test_method_safely(sketch, method, prompt, **kwargs):
    """Test a method safely with error handling"""
    try:
        print(f"\nTesting {method} method...")
        # 使用新的统一API
        image = convert_sketch(
            sketch, 
            prompt=prompt,
            method=method,
            size=512,
            **kwargs
        )
        
        # Save the result
        filename = f"output_{method}.png"
        image.save(filename)
        print(f"✅ {method} succeeded! Saved to {filename}")
        return True
        
    except Exception as e:
        print(f"❌ {method} failed: {str(e)}")
        # Print full traceback for debugging
        traceback.print_exc()
        return False

def test_backward_compatibility(sketch):
    """Test backward compatibility with old API"""
    try:
        print("\nTesting backward compatibility (sketch2img)...")
        image = sketch2img(
            sketch,
            method="controlnet",
            prompt="a cute cat",
            size=512,
            output_path="output_backward.png"
        )
        print("✅ Backward compatibility works!")
        return True
    except Exception as e:
        print(f"❌ Backward compatibility failed: {e}")
        return False

def test_direct_converter(sketch):
    """Test using converter directly"""
    try:
        print("\nTesting direct converter usage...")
        converter = Sketch2ImageConverter(method="controlnet")
        image = converter.convert(sketch, "a photorealistic cat", size=512)
        image.save("output_direct.png")
        print("✅ Direct converter works!")
        return True
    except Exception as e:
        print(f"❌ Direct converter failed: {e}")
        return False

def main():
    # 加载数据集
    print("Loading QuickDraw dataset...")
    dataset = QuickDraw()
    sketch = dataset[100]  # 假设是猫的sketch
    
    print(f"Loaded sketch with {len(sketch.paths)} paths")
    print(f"Available methods: {available_methods()}")
    
    # Test different methods safely
    # 修改方法测试部分
    methods_to_test = [
        ('controlnet', "a cute cat sitting on grass, photorealistic", 
        {'num_inference_steps': 20, 'guidance_scale': 7.5}),
        ('t2i_adapter', "a beautiful cat portrait", 
        {'num_inference_steps': 30, 'guidance_scale': 7.5}),
        ('pix2pix', "a realistic cat photo", 
        {'num_inference_steps': 10, 'guidance_scale': 7.5, 'image_guidance_scale': 1.0})
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
    
    # Test additional functionality
    test_backward_compatibility(sketch)
    test_direct_converter(sketch)
    
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