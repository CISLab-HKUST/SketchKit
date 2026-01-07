import sys
import os
from pathlib import Path
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sketchkit.brushstrokeengine import BrushstrokeStylizer
from sketchkit.datasets import PhotoSketching

"""
Optional parameters you can override (via kwargs in test functions, or at BrushstrokeStylizer init):
 - style_id: e.g. "playdoh10"; can blend with style_id2 + style_blend_alpha
 - feature_blending_level: default 2
 - color_mode: '1'/'2' use library colors; or "r,g,b;..." for custom
 - crop_margin: default 10
 - on_white: default True
 - render_mode: default "clear"
 - stitching_mode: default "all"
 - canvas_size: used when rendering Sketch via CairoRenderer
 - library_path: optional; defaults to model's projected_training_styles.pkl; you can pass other libs
 - encoder_checkpoint: optional; defaults to auto-select geo_encoders/weak.pt or strong.pt
 - device: optional; cpu / cuda / device index; default auto GPU if available
"""


def test_brushstroke_from_png(png_path: Path, output_dir: Path, **kwargs) -> bool:
    """Run stylization from an input PNG."""
    try:
        print("\nTesting brushstroke from PNG...")
        stylizer = BrushstrokeStylizer(
            model_name="style2",
            style_id=kwargs.get("style_id", "playdoh10"),
            auto_download=True,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"from_png_{stylizer.model_name}_{stylizer.style_id}.png"

        stylizer.run_from_png(
            png_path,
            output_path=out_path,
            feature_blending_level=kwargs.get("feature_blending_level", 2),
            color_mode=kwargs.get("color_mode", "1"),
            crop_margin=kwargs.get("crop_margin", 10),
            on_white=kwargs.get("on_white", True),
        )

        print(f"✅ PNG stylization succeeded! Saved to {out_path}")
        return True

    except Exception as e:
        print(f"❌ PNG stylization failed: {e}")
        traceback.print_exc()
        return False


def test_brushstroke_from_sketch(sketch, output_dir: Path, **kwargs) -> bool:
    """Render a Sketch via CairoRenderer, then stylize."""
    try:
        print("\nTesting brushstroke from Sketch...")
        stylizer = BrushstrokeStylizer(
            model_name="style2",
            style_id=kwargs.get("style_id", "playdoh10"),
            auto_download=True,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"from_sketch_{stylizer.model_name}_{stylizer.style_id}.png"

        stylizer.run(
            sketch,
            output_path=out_path,
            canvas_size=kwargs.get("canvas_size", 512),
            feature_blending_level=kwargs.get("feature_blending_level", 2),
            color_mode=kwargs.get("color_mode", "1"),
            crop_margin=kwargs.get("crop_margin", 10),
            on_white=kwargs.get("on_white", True),
        )

        print(f"✅ Sketch stylization succeeded! Saved to {out_path}")
        return True

    except Exception as e:
        print(f"❌ Sketch stylization failed: {e}")
        traceback.print_exc()
        return False


def main():
    # Load a sample sketch
    print("Loading PhotoSketching dataset...")
    dataset = PhotoSketching()
    sketch = dataset[5]
    print(f"Loaded sketch with {len(sketch.paths)} paths")

    # Prepare output dir and sample PNG
    output_dir = Path("tmp_brushstroke_test")
    png_path = Path("input/PhotoSketching.png")
    if not png_path.exists():
        print(f"❌ Sample PNG not found at {png_path}")
        return

    successes = []
    failures = []

    if test_brushstroke_from_png(png_path, output_dir):
        successes.append("from_png")
    else:
        failures.append("from_png")

    if test_brushstroke_from_sketch(sketch, output_dir):
        successes.append("from_sketch")
    else:
        failures.append("from_sketch")

    print("\n📊 Summary:")
    print(f"✅ Successful: {successes}")
    print(f"❌ Failed: {failures}")


if __name__ == "__main__":
    main()

# brushstrokeengine涉及的参数：
# 1. MODEL===
    # 示例：style2 或 style1
    # 含义：选择使用哪一个 NeuBE 训练好的模型目录，即 models/neube/<MODEL>/。
    # 区别：
        # style1：偏传统画风的模型
        # style2：偏现代/多样风格的模型
    # model 决定：
        # 使用哪一套权重与结构（style1/style2），也就决定了“风格空间”的整体分布。
# 2. STYLE（或当作 Z 使用）======
    # 示例：playdoh10（风格名）或 594（随机种子）
    # 含义有两种用法：
        # 1. 风格 ID 模式：
            # 当你传入 STYLE=playdoh10 且配合 LIBRARY 使用时，STYLE 是 brush library 里的一个 key，用于选择特定风格 latent（固定的笔触风格）。
        # 2. 随机种子 Z 模式：
            # 当你只传 Z=594（第三个参数，不给 LIBRARY），脚本会用这个整数作为随机种子，生成一个随机 style latent，而不是从库里查名字。
    # style（style_id / Z） 决定：
        # 在选定的 library 中，具体选哪一个风格（或者用哪个 seed 生成）。
    # 可选风格：
        # Loaded w library with 239 styles: dict_keys(['misc05', 'string17_take4', 'playdoh10', 'misc05_take2', 'lego00_take2', 'pompoms02', 'playdoh03', 'string23', 'string22_take2', 'beads03_take2', 'playdoh12', 'ribbon15', 'playdoh09_take2', 'beads03', 'string05_take2', 'lego02_take2', 'playdoh05', 'ribbon19', 'ribbon00_take2', 'beads01_take2', 'ribbon27', 'beads15_take3', 'string12', 'ribbon18_take3', 'beads14_take3', 'pompoms01_take2', 'lego00', 'beads01', 'beads15_take2', 'playdoh09', 'misc06', 'paper01', 'ribbon05', 'beads12', 'pompoms03', 'misc02', 'chain00', 'beads05', 'ribbon29_take4', 'beads17_take2', 'playdoh00', 'playdoh07_take2', 'ribbon23', 'ribbon23_take2', 'ribbon29_take3', 'beads04', 'playdoh05_take2', 'string18', 'string05', 'beads11_take4', 'ribbon19_take3', 'beads00_take2', 'ribbon31_take2', 'ribbon06', 'misc03_take2', 'ribbon15_take2', 'string23_take3', 'string04', 'beads21', 'playdoh03_take2', 'ribbon16_take3', 'beads07_take2', 'ribbon25', 'string22', 'string08', 'ribbon13', 'ribbon15_take3', 'ribbon21_take4', 'playdoh06', 'misc06_take2', 'misc09', 'ribbon28', 'string03', 'ribbon20', 'beads02_take2', 'ribbon31_take3', 'ribbon14', 'playdoh08', 'beads09_take2', 'chain01', 'paper02', 'pompoms02_take2', 'beads14', 'string20', 'pompoms00', 'beads10_take2', 'paper06', 'misc04_take2', 'string00_take2', 'string15', 'ribbon13_take2', 'string19', 'beads09', 'playdoh02_take2', 'beads10', 'ribbon03_take3', 'string10', 'ribbon01_take3', 'ribbon07', 'pompoms03_take2', 'misc05_take3', 'beads12_take2', 'ribbon23_take3', 'playdoh08_take2', 'beads19', 'misc07', 'paper05', 'ribbon06_take2', 'beads11', 'beads02', 'ribbon21_take3', 'beads08', 'paper05_take2', 'string13', 'ribbon29', 'misc07_take2', 'paper01_take2', 'beads13_take2', 'string17', 'paper03', 'ribbon20_take2', 'string17_take2', 'playdoh04', 'ribbon14_take3', 'beads20', 'string19_take2', 'ribbon30', 'beads11_take3', 'ribbon03_take2', 'string06', 'playdoh01', 'ribbon01_take2', 'ribbon19_take4', 'ribbon15_take4', 'beads11_take2', 'ribbon08', 'string00_take3', 'ribbon14_take2', 'ribbon18', 'string18_take3', 'beads06', 'beads15', 'ribbon07_take2', 'paper02_take2', 'ribbon18_take4', 'beads07', 'ribbon28_take4', 'string21', 'chain02_take2', 'misc10', 'string16', 'beads17', 'ribbon02', 'lego01_take2', 'string11', 'ribbon22', 'string09', 'beads21_take2', 'pompoms03_take3', 'beads16', 'ribbon00', 'beads05_take2', 'beads13', 'ribbon00_take3', 'paper03_take2', 'playdoh11', 'string02_take2', 'paper04_take2', 'ribbon21_take2', 'ribbon24', 'ribbon11_take2', 'string18_take2', 'string12_take2', 'string01', 'ribbon17', 'paper06_take2', 'lego01', 'ribbon28_take2', 'ribbon21', 'ribbon11', 'ribbon05_take3', 'string17_take3', 'misc11', 'beads20_take2', 'ribbon02_take2', 'ribbon19_take2', 'ribbon07_take3', 'ribbon22_take2', 'ribbon04', 'beads18', 'playdoh06_take2', 'string00_take4', 'string20_take2', 'pompoms00_take2', 'paper04', 'ribbon28_take3', 'string14', 'ribbon12', 'ribbon10', 'lego02', 'ribbon01', 'misc08', 'ribbon08_take2', 'ribbon16', 'lego02_take3', 'paper00', 'string23_take2', 'string07', 'ribbon05_take2', 'misc01', 'beads06_take2', 'string00', 'ribbon11_take3', 'ribbon17_take2', 'ribbon14_take4', 'ribbon09', 'beads00', 'ribbon24_take2', 'pompoms01', 'playdoh04_take2', 'ribbon26', 'ribbon27_take2', 'ribbon29_take2', 'ribbon18_take2', 'ribbon16_take2', 'chain02', 'playdoh01_take2', 'playdoh12_take2', 'ribbon31', 'ribbon03', 'playdoh02', 'beads14_take2', 'string04_take2', 'chain01_take2', 'playdoh07', 'string18_take4', 'misc00', 'misc04', 'misc03'])
# 3. LIBRARY======
    # 示例：models/neube/style2/brush_libs/projected_training_styles.pkl
    # 含义：风格库（brush library）的路径，里面存放了许多预先计算好的风格 latent（一个或多个 .pkl）。
    # 作用：STYLE 作为 key，在这个库中取出对应的 style 向量（或多向量），再交给 BrushLibrary / GanBrushOptions 来设置笔触风格。
    # 要求：必须和 MODEL 对应，比如 MODEL=style2 时，要用 models/neube/style2/brush_libs/... 下面的库。
    # library 决定：
        # 在这个 model 的风格空间里，要从哪个“字典”（库）里拿风格向量：
            # 训练集投影库（projected_training_styles.pkl）
            # 新风格库（projected_novel_styles.pkl）
            # CLIP 风格库（clip_brushes.pkl）
            # 随机库（rand100, 1000 等）