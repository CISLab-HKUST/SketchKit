import os
import sys
import math
from typing import List, Tuple

import numpy as np
import pygame
import moderngl
from PIL import Image
from sketchkit.core.sketch import Sketch, Curve

def bezier_point(p0, p1, p2, p3, t):
    u = 1.0 - t
    return (
        u*u*u*p0[0] + 3*u*u*t*p1[0] + 3*u*t*t*p2[0] + t*t*t*p3[0],
        u*u*u*p0[1] + 3*u*u*t*p1[1] + 3*u*t*t*p2[1] + t*t*t*p3[1],
    )
# -----------------------------------------------------------------------------
# Utility: equidistant stamps along a polyline (CPU)
# -----------------------------------------------------------------------------

def polyline_equidistant_stamps(
    polyline: List[Tuple[float, float]],
    spacing_px: float,
    canvas_px: int
) -> List[Tuple[float, float]]:
    """
    Compute equidistant stamp centers along a polyline in NDC coordinates.
    spacing_px is in pixels; convert to NDC using canvas size (assumes square canvas).
    """
    if len(polyline) < 2:
        return polyline[:]

    spacing_ndc = (2.0 * spacing_px) / float(canvas_px)

    out = []
    acc = 0.0
    t_next = 0.0

    p_prev = np.array(polyline[0], dtype=np.float32)
    out.append((float(p_prev[0]), float(p_prev[1])))

    for i in range(1, len(polyline)):
        p_cur = np.array(polyline[i], dtype=np.float32)
        seg = p_cur - p_prev
        seg_len = float(np.linalg.norm(seg))
        if seg_len <= 1e-8:
            continue
        seg_dir = seg / seg_len
        while acc + seg_len >= t_next + spacing_ndc:
            remain = (t_next + spacing_ndc) - acc
            stamp_pos = p_prev + seg_dir * remain
            out.append((float(stamp_pos[0]), float(stamp_pos[1])))
            t_next += spacing_ndc
        acc += seg_len
        p_prev = p_cur

    # ensure last vertex stamped
    if out and (abs(out[-1][0] - polyline[-1][0]) > 1e-5 or abs(out[-1][1] - polyline[-1][1]) > 1e-5):
        out.append(polyline[-1])
    return out


def polylines_to_stamps(
    polylines: List[List[Tuple[float, float]]],
    spacing_px: float,
    canvas_px: int
) -> List[Tuple[float, float]]:
    stamps = []
    for pl in polylines:
        stamps.extend(polyline_equidistant_stamps(pl, spacing_px, canvas_px))
    return stamps


# -----------------------------------------------------------------------------
# ModernGL shader sources (textured stamp)
# -----------------------------------------------------------------------------

VERT_SRC = """
#version 330 core
// Per-vertex: unit quad centered at origin, covering [-1,1]^2 in local coords
layout (location = 0) in vec2 in_pos;

// Per-instance: center in NDC, stamp radius (NDC), and aspect (tex_w/tex_h)
layout (location = 1) in vec2 i_center_ndc;
layout (location = 2) in float i_radius_ndc;
layout (location = 3) in float i_aspect;

out vec2 v_local; // -1..1 local coordinates, used for UV mapping

void main() {
    // Preserve texture aspect: scale X by aspect, Y by 1
    vec2 scaled = vec2(in_pos.x * i_aspect, in_pos.y);
    // Radius_ndc corresponds to half the stroke width in NDC
    vec2 world = i_center_ndc + scaled * (i_radius_ndc * 2.0);
    v_local = in_pos; // map to UV in fragment
    gl_Position = vec4(world, 0.0, 1.0);
}
"""

FRAG_SRC = """
#version 330 core
in vec2 v_local;
out vec4 out_color;

uniform sampler2D u_tex;

void main() {
    // Map local [-1,1] to [0,1] UVs
    vec2 uv = v_local * 0.5 + 0.5;

    // If you want hard clipping, discard outside the quad — but letting the
    // texture alpha decide shape typically looks better:
    vec4 texel = texture(u_tex, uv);

    // Discard fully transparent texel (prevents small alpha bleed)
    if (texel.a <= 0.001)
        discard;

    out_color = texel;
}
"""

class CialloRenderer:
    def __init__(self, canvas_size=512, canvas_color=(1, 1, 1)):
        """
        Initializes the CairoRenderer with a given canvas size and color.

        Args:
            canvas_size (int, optional): The size of the canvas (both width and height) in pixels. Default is 512.
            canvas_color (tuple, optional): The background color of the canvas as an (R, G, B) tuple. Default is (1, 1, 1) (white).
        """
        self.canvas_size = canvas_size
        self.canvas_color = canvas_color

    def render(
        self,
        sketch: Sketch, #List[List[Tuple[float, float]]]
        texture_path: str = "texture.png",
        canvas_size: int = 1024,
        stroke_width_px: float = 24.0,
        spacing_px: float = 12.0,
        out_path: str = "output.png",
        clear_color=(0, 0, 0, 0),
    ):
        """
        Render polylines with a given RGBA texture as stamps.
        - polylines: in NDC coordinates [-1,1]
        - stroke_width_px: fixed stroke width (the *short* dimension of the stamp)
        - spacing_px: distance between consecutive stamp centers
        - texture preserves aspect ratio; if non-square, the stamp quad stretches to keep it.
        - Renders off-screen and writes out_path (PNG).

        Returns: None (writes PNG).
        """
        samples = 100
        polylines_raw = []
        all_pts = []
        for each_path in sketch.paths:
            for curve in each_path.curves:
                if isinstance(curve, Curve):
                    p0_t, p1_t, p2_t, p3_t = curve.p_start, curve.p_crtl1, curve.p_crtl2, curve.p_end
                    p0, p1, p2, p3 = (p0_t.x, p0_t.y), (p1_t.x, p1_t.y), (p2_t.x, p2_t.y), (p3_t.x, p3_t.y)
                    ts = [i/(samples-1) for i in range(samples)]
                    pts = [bezier_point(p0,p1,p2,p3,t) for t in ts]
                    polylines_raw.append(pts)
                    all_pts.extend(pts)

        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        def normalize_point(p):
            x = (p[0] - min_x) / (max_x - min_x) * 2 - 1
            y = (p[1] - min_y) / (max_y - min_y) * 2 - 1
            return (x, y)

        polylines = [[normalize_point(p) for p in poly] for poly in polylines_raw]

        print(f"Rendering {len(polylines)} polylines with {sum(len(pl) for pl in polylines)} total points.")
        # ---- Create hidden window / GL context ----
        os.environ.setdefault("SDL_WINDOWS_DPI_AWARENESS", "permonitorv2")
        pygame.init()
        pygame.display.init()

        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, True)

        pygame.display.set_mode(
            (canvas_size, canvas_size),
            flags=pygame.OPENGL | pygame.DOUBLEBUF | pygame.HIDDEN,
        )

        ctx = moderngl.get_context()
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        ctx.viewport = (0, 0, canvas_size, canvas_size)

        # ---- Load texture (RGBA), build GL texture, set filtering ----
        tex_img = Image.open(texture_path).convert("RGBA")
        tex_w, tex_h = tex_img.size
        tex_data = tex_img.tobytes()
        tex = ctx.texture((tex_w, tex_h), components=4, data=tex_data)
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        tex.repeat_x = False
        tex.repeat_y = False
        # Optional: tex.build_mipmaps()

        # Aspect ratio (w/h); stamp quad scales X by this factor.
        aspect = float(tex_w) / float(tex_h)

        # ---- Convert stroke width from px to NDC radius ----
        radius_ndc = (stroke_width_px * 0.5) * (2.0 / canvas_size)

        # ---- Build equidistant stamp centers (instances) ----
        centers = polylines_to_stamps(polylines, spacing_px=spacing_px, canvas_px=canvas_size)

        # If nothing to draw, just save a blank
        if not centers:
            ctx.clear(*[c/255.0 if c > 1 else c for c in clear_color])
            pixels = ctx.screen.read(components=4, alignment=1)
            img = Image.frombytes("RGBA", (canvas_size, canvas_size), pixels)
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            img.save(out_path)
            pygame.quit()
            return

        # Instance buffer: (cx, cy, radius_ndc, aspect)
        inst = np.array([(cx, cy, radius_ndc, aspect) for (cx, cy) in centers], dtype="f4")

        # Unit quad covering [-1,1]^2
        quad = np.array([
            [-1.0, -1.0],
            [ 1.0, -1.0],
            [ 1.0,  1.0],
            [-1.0, -1.0],
            [ 1.0,  1.0],
            [-1.0,  1.0],
        ], dtype="f4")

        vbo_quad = ctx.buffer(quad.tobytes())
        vbo_inst = ctx.buffer(inst.tobytes())

        prog = ctx.program(vertex_shader=VERT_SRC, fragment_shader=FRAG_SRC)
        vao = ctx.vertex_array(
            prog,
            [
                (vbo_quad, "2f", "in_pos"),
                (vbo_inst, "2f 1f 1f/i", "i_center_ndc", "i_radius_ndc", "i_aspect"),  # per-instance
            ],
        )

        # Bind texture to unit 0
        tex.use(location=0)
        # If your ModernGL version requires explicit uniform binding:
        # prog["u_tex"].value = 0  # (ModernGL usually auto-binds the first sampler to 0)

        # Clear + draw
        if isinstance(clear_color[0], int):  # allow (0,0,0,0) or (0.0,...)
            r, g, b, a = [c/255.0 for c in clear_color]
        else:
            r, g, b, a = clear_color
        ctx.clear(r, g, b, a)

        vao.render(instances=len(centers))
        ctx.finish()

        # Readback and save
        pixels = ctx.screen.read(components=4, alignment=1)  # RGBA
        img = Image.frombytes("RGBA", (canvas_size, canvas_size), pixels)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img.save(out_path)
        print(f"Saved: {out_path} (stamps={len(centers)}, stroke={stroke_width_px}px, spacing={spacing_px}px, tex={tex_w}x{tex_h})")

        pygame.quit()
        return None