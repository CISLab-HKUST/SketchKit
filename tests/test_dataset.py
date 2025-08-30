from sketchkit.datasets import TUBerlin
from sketchkit.datasets import SketchyDatabase
from sketchkit.datasets import GMUSketchCleanup
from sketchkit.datasets import CreativeSketch
from sketchkit.datasets import PhotoSketch


if __name__ == "__main__":
    dataset = TUBerlin()
    print(len(dataset))
    sketch = dataset[0]
    print(sketch.curve_num, sketch.path_num)

    # my_data2 = QuickDraw(category="cat", split="train")
    # print(len(my_data2))
    # print(my_data2[0])


    # data = SketchyDatabase()
    # print(f"Total sketches: {len(data)}")
    # sketch = data[0]
    # print(sketch)
    # print(sketch.stroke_num, sketch.path_num)
    # print("Loaded first sketch successfully.")

    # data = GMUSketchCleanup(
    #     split="GT", # "GT" or "Rough"
    #     modality="SVG" # Only SVG is supported now
    # )

    # print(len(data))
    # sketchGMU = data[0]
    # print(sketchGMU)
    # print(sketchGMU.curve_num, sketchGMU.path_num)


    data = CreativeSketch()
    print(f"Total sketches: {len(data)}")
    sketch = data[0]
    print(sketch)
    print(sketch.curve_num, sketch.path_num)
    print("Loaded first sketch successfully.")

    # PhotoSketch dataset test----------------begin----------------
    # print("\n" + "="*50)
    # print("Testing PhotoSketch dataset...")
    # photosketch_data = PhotoSketch()
    # print(f"Total sketches: {len(photosketch_data)}")
    # photosketch_sketch = photosketch_data[0]
    # print(photosketch_sketch)
    # print(f"Sketch has {photosketch_sketch.curve_num} curves and {photosketch_sketch.path_num} paths")
    # print("Loaded first PhotoSketch successfully.")
    
    # # Test SVG path access
    # svg_path = photosketch_data.get_sketch_svg_path(0)
    # print(f"SVG path: {svg_path}")
    
    # # Test PNG path access (note: may not exist)
    # try:
    #     png_path = photosketch_data.get_sketch_png_path(0)
    #     print(f"PNG path: {png_path}")
    # except Exception as e:
    #     print(f"PNG path access: {e}")
    
    # # Test photo path access (note: may not exist)
    # try:
    #     photo_path = photosketch_data.get_photo_path(0)
    #     print(f"Photo path: {photo_path}")
    # except Exception as e:
    #     print(f"Photo path access: {e}")
    # PhotoSketch dataset test----------------end----------------
