from sketchkit.datasets import DifferSketching
from sketchkit.renderer.ciallo_renderer import CialloRenderer, CialloRenderOptions
from sketchkit.renderer.cairo_renderer import CairoRenderer, CairoRenderOptions

if __name__ == "__main__":
    my_data = DifferSketching()
    print(len(my_data))
    
    render_options = CialloRenderOptions
    renderer = CialloRenderer(render_options)    
    renderer.render(my_data[13014], render_options)

    render_options = CairoRenderOptions
    renderer = CairoRenderer(render_options)    
    renderer.render(my_data[13014], render_options)