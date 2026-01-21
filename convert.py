from sketchkit.core import Sketch, Point, Curve, Vertex, Path
import numpy as np
from sketchkit.renderer.cairo_renderer import CairoRenderer, CairoRenderOptions

path_before_prune = "media/svg_iter800_before_prune.svg"
path_after_prune = "media/svg_iter800_after_prune.svg"
path_after_add = "media/svg_iter800_after_add.svg"


render_options = CairoRenderOptions(background_color=(1.0, 1.0, 1.0))
renderer = CairoRenderer(render_options)

def equal_curve(curve1: Curve, curve2: Curve) -> bool:
    return np.allclose(curve1.np_points, curve2.np_points)

def find_curve_in_sketch(sketch: Sketch, curve: Curve) -> bool:
    for path in sketch.paths:
        for curve_in_path in path.curves:
            if equal_curve(curve, curve_in_path):
                return True
    return False

def equal_vertex(vertex1, vertex2) -> bool:
    return vertex1.x == vertex2.x and vertex1.y == vertex2.y
    

def find_split_curve(sketch: Sketch, curve: Curve) -> bool:
    for path in sketch.paths:
        for curve_in_path in path.curves:
            if not equal_curve(curve, curve_in_path) and (equal_vertex(curve_in_path.p_start, curve.p_start) or equal_vertex(curve_in_path.p_end, curve.p_end)):
                return True
    return False

def find_add_curve(sketch: Sketch, curve: Curve) -> bool:
    for path in sketch.paths:
        for curve_in_path in path.curves:
            if not equal_curve(curve, curve_in_path) and (not equal_vertex(curve_in_path.p_start, curve.p_end) and not equal_vertex(curve_in_path.p_end, curve.p_start)):
                return True
    return False

if __name__ == '__main__':
    sketch_before_prune = Sketch.from_svg(path_before_prune)
    sketch_after_prune = Sketch.from_svg(path_after_prune)
    sketch_after_add = Sketch.from_svg(path_after_add)
    
    raster_image_before_prune = renderer.render(sketch_before_prune)
    raster_image_before_prune.save('media/before_prune.png', 'PNG')
    for path in sketch_before_prune.paths:
        curve = path.curves[0]
        curve.thickness = 5.0
        if not find_curve_in_sketch(sketch_after_prune, curve):
            curve.color = (0.0, 0.0, 1.0)
    raster_image_pruning = renderer.render(sketch_before_prune)
    raster_image_pruning.save('media/pruning.png', 'PNG')
    # 
    raster_image_after_prune = renderer.render(sketch_after_prune)
    raster_image_after_prune.save('media/after_prune.png', 'PNG')

    for path in sketch_after_add.paths[-5:]:
        path.color = (0.0, 1.0, 0.0)
        
    
    for path in sketch_after_add.paths:
        curve = path.curves[0]
        curve.thickness = 5.0
    
    raster_image_after_add = renderer.render(sketch_after_add)
    raster_image_after_add.save('media/after_add.png', 'PNG')

    sketch_after_add.paths = sketch_after_add.paths[:-5]

    for path in sketch_after_add.paths:
        curve = path.curves[0]
        if not find_curve_in_sketch(sketch_after_prune, curve):
            
            if find_split_curve(sketch_after_prune, curve):
                curve.color = (1.0, 0.0, 0.0)
                curve.thickness = 1.0
    

    raster_image_after_add = renderer.render(sketch_after_add)
    raster_image_after_add.save('media/spliting.png', 'PNG')
    


