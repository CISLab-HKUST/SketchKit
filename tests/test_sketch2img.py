import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sketchkit.datasets import QuickDraw
from sketchkit.sketch2image import ImageGenerator
import traceback

def test_method_safely(sketch, method, prompt, **kwargs):
    """Test a method safely with error handling"""
    try:
        print(f"\nTesting {method} method...")
        generator = ImageGenerator(method=method)
        image = generator.run(sketch, prompt=prompt, size=512, **kwargs)
        
        # Save the result
        filename = f"output_{method}.png"
        image.save(filename)
        print(f"✅ {method} succeeded! Saved to {filename}")
        return True
        
    except Exception as e:
        print(f"❌ {method} failed: {str(e)}")
        traceback.print_exc()
        return False

def test_direct_generator(sketch):
    """Test using generator directly"""
    try:
        print("\nTesting direct generator usage...")
        generator = ImageGenerator(method="controlnet")
        image = generator.run(sketch, "a photorealistic cat", size=512)
        image.save("output_direct.png")
        print("✅ Direct generator works!")
        return True
    except Exception as e:
        print(f"❌ Direct generator failed: {e}")
        return False

def main():
    # 加载数据集
    print("Loading QuickDraw dataset...")
    dataset = QuickDraw()
    sketch = dataset[100]  # 假设是猫的sketch
    
    print(f"Loaded sketch with {len(sketch.paths)} paths")
    
    # Test different methods safely
    methods_to_test = [
        ('controlnet', "a cute cat sitting on grass, photorealistic", 
        {'num_inference_steps': 20, 'guidance_scale': 7.5, 'controlnet_conditioning_scale': 1.0}),
        ('t2i_adapter', "a beautiful cat portrait", 
        {'num_inference_steps': 30, 'guidance_scale': 7.5, 'adapter_conditioning_scale': 1.0}),
        ('pix2pix', "a realistic cat photo", 
        {'num_inference_steps': 10, 'guidance_scale': 7.5, 'image_guidance_scale': 1.0})
    ]
    
    successful_methods = []
    failed_methods = []
    
    for method, prompt, kwargs in methods_to_test:
        success = test_method_safely(sketch, method, prompt, **kwargs)
        if success:
            successful_methods.append(method)
        else:
            failed_methods.append(method)
    
    # Test additional functionality
    test_direct_generator(sketch)
    
    # Print summary
    print(f"\n📊 Summary:")
    print(f"✅ Successful methods: {successful_methods}")
    print(f"❌ Failed methods: {failed_methods}")


if __name__ == "__main__":
    main()