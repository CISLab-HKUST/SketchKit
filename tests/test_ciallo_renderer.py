from sketchkit.datasets import DifferSketching
from sketchkit.renderer.ciallo_renderer import CialloRenderer, CialloRenderOptions

if __name__ == "__main__":
    my_data = DifferSketching(skip_integrity_check=True)
    print(len(my_data))
    
    render_options = CialloRenderOptions()

    renderer = CialloRenderer()    

    render_options.brush_texture_path = "./assets/stamp1.png"
    render_options.stroke_color = (1, 0, 0, 255)
    img = renderer.render(my_data[13014], render_options)
    img.save("ciallo_output_1111.png")

    render_options.canvas_size = (1024, 1024)
    render_options.brush_texture_path = "./assets/stamp2.png"
    render_options.background_color = (2, 0, 0, 1.0)
    img = renderer.render(my_data[13014], render_options)
    img.save("ciallo_output_2222.png")
    # render_options = CairoRenderOptions()
    # renderer = CairoRenderer()    
    # renderer.render(my_data[13014], render_options)