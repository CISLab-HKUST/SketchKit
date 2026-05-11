import os
import pytest
import numpy as np
from PIL import Image
from sketchkit.image2sketch import image2sketch

def test_anilines_basic_generation():
    # Create a small dummy image
    dummy_img = Image.new("RGB", (64, 64), color="white")
    
    # Try the anilines generation
    try:
        out = image2sketch(dummy_img, method="anilines")
        assert isinstance(out, Image.Image)
        assert out.size == (64, 64)
        
        output_dir = "outputs/image2sketch"
        os.makedirs(output_dir, exist_ok=True)
        out.save(os.path.join(output_dir, "dummy_anilines_basic.png"))
    except Exception as e:
        pytest.fail(f"image2sketch failed with anilines method: {e}")

def test_anilines_detail_mode():
    dummy_img = Image.new("RGB", (32, 32), color="black")
    from sketchkit.image2sketch.anilines.generate import AniLinesModel
    
    # Directly initialize to pass mode kwargs (which aren't directly passed via image2sketch interface currently unless we pass (**kwargs) which might not be supported)
    model = AniLinesModel(mode="detail")
    try:
        out = model.generate(dummy_img)
        
        assert isinstance(out, Image.Image)
        
        output_dir = "outputs/image2sketch"
        os.makedirs(output_dir, exist_ok=True)
        out.save(os.path.join(output_dir, "dummy_anilines_detail.png"))
    except Exception as e:
        pytest.fail(f"Direct generate failed with detail mode: {e}")

if __name__ == "__main__":
    image_path = "test_images/madoka.jpg"
    
    if os.path.exists(image_path):
        # Load a real input image
        with Image.open(image_path) as pil_image:
            input_image = pil_image.convert("RGB")

        output_dir = "outputs/image2sketch"
        os.makedirs(output_dir, exist_ok=True)

        print("Testing AniLines Basic Mode...")
        basic_result = image2sketch(input_image, method="anilines")
        basic_path = os.path.join(output_dir, "camel_anilines_basic.png")
        basic_result.save(basic_path)
        print(f"Basic mode sketch saved to: {basic_path}")

        print("Testing AniLines Detail Mode...")
        from sketchkit.image2sketch.anilines.generate import AniLinesModel
        detail_model = AniLinesModel(mode="detail")
        detail_result = detail_model.generate(input_image)
        detail_path = os.path.join(output_dir, "camel_anilines_detail.png")
        detail_result.save(detail_path)
        print(f"Detail mode sketch saved to: {detail_path}")
    else:
        print(f"Test image {image_path} not found. Run with pytest to test with dummy images.")
