import math
import moderngl
import numpy as np
from typing import List
from PIL import Image
from dataclasses import dataclass

from sketchkit.renderer.renderer import Renderer, RenderOptions
from sketchkit.core.sketch import Sketch, Curve, Path, Vertex, Point


def _sample_cubic_bezier(points: np.ndarray, num_samples: int) -> np.ndarray:
    """Vectorized sampling of a cubic Bezier curve."""
    if num_samples < 2:
        return points[[0, -1]]
    t = np.linspace(0.0, 1.0, num_samples, dtype=np.float32).reshape(-1, 1)
    mt = 1.0 - t
    P0, P1, P2, P3 = points
    # B(t) = (1-t)^3 P0 + ...
    B = (mt**3) * P0 + 3 * (mt**2) * t * P1 + 3 * mt * (t**2) * P2 + (t**3) * P3
    return B.astype(np.float32)


def sketch_to_polylines(
    sketch: Sketch, samples_per_curve: int = 20
) -> List[np.ndarray]:
    """Converts a sketch into a list of polylines (N, 3) [x, y, pressure]."""
    polylines = []
    for path in sketch.paths:
        current_poly_pts = []
        for curve in path.curves:
            if not isinstance(curve, Curve):
                continue

            pts = curve.np_points
            p_vals = curve.pressure
            p0 = p_vals[0] if p_vals[0] is not None else 1.0
            p1 = p_vals[1] if p_vals[1] is not None else 1.0

            positions = _sample_cubic_bezier(pts, samples_per_curve)
            t_vals = np.linspace(0.0, 1.0, samples_per_curve).reshape(-1, 1)
            pressures = (1.0 - t_vals) * p0 + t_vals * p1

            segment = np.hstack([positions, pressures]).astype(np.float32)
            if len(current_poly_pts) > 0:
                current_poly_pts.append(segment[1:])  # Avoid duplicating joints
            else:
                current_poly_pts.append(segment)

        if current_poly_pts:
            polylines.append(np.vstack(current_poly_pts))
    return polylines


def sample_points_by_distance(
    polyline: np.ndarray,
    spacing: float,
    spacing_jitter: float = 0.0,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Resamples a polyline [x, y, p] to have points spaced by 'spacing' (with optional jitter).
    Returns: (N, 4) -> [x, y, pressure, angle]
    """
    if len(polyline) < 2:
        return np.zeros((0, 4), dtype=np.float32)
    if rng is None:
        rng = np.random.default_rng()

    # Calculate segment lengths
    diffs = polyline[1:, :2] - polyline[:-1, :2]
    dists = np.linalg.norm(diffs, axis=1)

    total_len = np.sum(dists)
    if total_len <= 0:
        return np.zeros((0, 4), dtype=np.float32)

    # Walk along the line
    result = []
    current_dist = 0.0
    seg_idx = 0
    seg_progress = 0.0  # Distance into current segment

    # Start at 0 (or random offset)
    current_dist = rng.uniform(0.0, spacing) if spacing_jitter > 0 else 0.0

    while seg_idx < len(dists):
        p0 = polyline[seg_idx]
        p1 = polyline[seg_idx + 1]
        seg_len = dists[seg_idx]

        if seg_len < 1e-5:
            seg_idx += 1
            continue

        angle = math.atan2(p1[1] - p0[1], p1[0] - p0[0])

        while current_dist <= seg_progress + seg_len:
            # Interpolate
            t = (current_dist - seg_progress) / seg_len
            pos_x = p0[0] + (p1[0] - p0[0]) * t
            pos_y = p0[1] + (p1[1] - p0[1]) * t
            pressure = p0[2] + (p1[2] - p0[2]) * t

            result.append([pos_x, pos_y, pressure, angle])

            # Advance
            step = spacing
            if spacing_jitter > 0:
                # Vary spacing: spacing * (1.0 +/- jitter)
                noise = rng.uniform(-spacing_jitter, spacing_jitter)
                step *= max(0.1, 1.0 + noise)

            current_dist += step

        seg_progress += seg_len
        seg_idx += 1

    return np.array(result, dtype=np.float32)


def build_vanilla_sdf_geometry(
    polylines: List[np.ndarray], base_thickness: float
) -> np.ndarray:
    all_verts = []
    corners = np.array(
        [[-1, -1], [1, -1], [1, 1], [-1, -1], [1, 1], [-1, 1]], dtype=np.float32
    )

    for poly in polylines:
        if len(poly) < 2:
            continue
        p0_data, p1_data = poly[:-1], poly[1:]

        r0 = p0_data[:, 2] * base_thickness * 0.5
        r1 = p1_data[:, 2] * base_thickness * 0.5

        lens = np.linalg.norm(p1_data[:, :2] - p0_data[:, :2], axis=1)
        valid = lens > 1e-5

        p0_v, p1_v = p0_data[valid], p1_data[valid]
        r0_v, r1_v = r0[valid], r1[valid]

        count = len(p0_v)
        if count == 0:
            continue

        attr_corner = np.tile(corners[None, :, :], (count, 1, 1))
        attr_p0 = np.tile(p0_v[:, None, :2], (1, 6, 1))
        attr_p1 = np.tile(p1_v[:, None, :2], (1, 6, 1))
        attr_r0 = np.tile(r0_v[:, None], (1, 6))
        attr_r1 = np.tile(r1_v[:, None], (1, 6))

        attributes = np.dstack(
            [attr_corner, attr_p0, attr_p1, attr_r0[..., None], attr_r1[..., None]]
        ).reshape(-1, 8)
        all_verts.append(attributes)

    return np.vstack(all_verts) if all_verts else np.zeros((0, 8), dtype=np.float32)


def build_stamp_instance_data(
    polylines: List[np.ndarray],
    spacing: float,
    base_thickness: float,
    rotation_randomness: float,
    noise_factor: float,
    seed: int,
) -> np.ndarray:
    """
    Builds instance buffer for glDrawArraysInstanced with randomness parameters.
    """
    rng = np.random.default_rng(seed)
    all_instances = []

    for poly in polylines:
        # Pass jitter parameters to sampler for spacing variation
        spacing_jitter = noise_factor if noise_factor > 0 else 0.0

        samples = sample_points_by_distance(
            poly, spacing, spacing_jitter, rng
        )  # (N, 4)
        count = len(samples)
        if count == 0:
            continue

        pos = samples[:, 0:2]
        pressure = samples[:, 2]
        angle = samples[:, 3]

        # 1. Size Jitter
        # Base Size = thickness * pressure * 2.0 (diameter)
        sizes = pressure * base_thickness * 2.0
        if noise_factor > 0:
            scale_noise = rng.uniform(
                1.0 - noise_factor, 1.0 + noise_factor, size=count
            )
            sizes *= np.maximum(0.1, scale_noise)

        # 2. Angle Jitter
        if rotation_randomness > 0:
            angle_noise = rng.uniform(
                -rotation_randomness, rotation_randomness, size=count
            )
            angle += angle_noise

        # 3. Position Jitter (Perpendicular)
        if noise_factor > 0:
            # Normal vector (-sin(a), cos(a)) relative to path direction
            # angle is atan2(dy, dx), so tangent is (cos, sin)
            normals = np.column_stack([-np.sin(angle), np.cos(angle)])

            # Offset amount: noise_factor * thickness
            jitter_mag = (
                rng.uniform(-1.0, 1.0, size=count) * noise_factor * base_thickness
            )
            pos += normals * jitter_mag[:, None]

        instances = np.column_stack([pos, sizes, angle]).astype(np.float32)
        all_instances.append(instances)

    return (
        np.vstack(all_instances)
        if all_instances
        else np.zeros((0, 4), dtype=np.float32)
    )


@dataclass
class CialloRenderOptions(RenderOptions):
    background_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)

    stroke_thickness: float = 12.0
    stroke_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    # stroke_opacity: float | None = None

    brush_type: str = "textured"
    brush_texture_path: str = "./assets/stamp2.png"
    spacing: float | None = 1.0
    rotation_randomness: float = 0.0
    noise_factor: float = 0.0
    rng_seed: int = 42


class CialloRenderer(Renderer):
    """
    A class for brush-based rendering of vector sketches using the method of paper "GPU-Accelerated Rendering of Vector
    Brush Strokes" in SIGGRAPH 2024.
    """
    def __init__(self, render_options: CialloRenderOptions | None = None):
        if render_options is None:
            render_options = CialloRenderOptions()  # Default if None is passed
        self.render_options = render_options
        self.new_ctx(render_options)
        # self.size = (
        #     self.render_options.canvas_size[0],
        #     self.render_options.canvas_size[1],
        # )

    def new_ctx(self, render_options: CialloRenderOptions):
        if hasattr(self, "ctx"):
            del self.ctx
        if hasattr(self, "fbo"):
            del self.fbo
        if hasattr(self, "fbo_tex"):
            del self.fbo_tex
        if hasattr(self, "depth_tex"):
            del self.depth_tex

        try:
            self.ctx = moderngl.create_standalone_context(backend="egl")
        except:
            self.ctx = moderngl.create_standalone_context()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        canvas_size = (
            (512, 512)
            if render_options.canvas_size is None
            else render_options.canvas_size
        )

        # Framebuffer with Depth
        self.fbo_tex = self.ctx.texture(canvas_size, 4)
        self.depth_tex = self.ctx.depth_texture(canvas_size)
        self.fbo = self.ctx.framebuffer(self.fbo_tex, self.depth_tex)

        # Vanilla Shader
        self.prog_vanilla = self.ctx.program(
            vertex_shader="""
            #version 330
            in vec2 in_corner; in vec2 in_p0; in vec2 in_p1;
            in float in_r0; in float in_r1;
            uniform vec2 u_res; uniform float u_depth;
            out vec2 v_p0; out vec2 v_p1;
            out float v_r0; out float v_r1; out vec2 v_local;
            void main() {
                float max_r = max(in_r0, in_r1);
                float pad = max_r + 10.0; 
                vec2 dir = in_p1 - in_p0;
                if (length(dir) < 1e-5) dir = vec2(1,0);
                vec2 t = normalize(dir);
                vec2 n = vec2(-t.y, t.x);
                vec2 l_pos = (in_corner.x < 0.0) ? in_p0 : in_p1;
                vec2 pos = l_pos + (t * in_corner.x * pad) + (n * in_corner.y * pad);
                v_p0 = in_p0; v_p1 = in_p1; v_r0 = in_r0; v_r1 = in_r1; v_local = pos;
                vec2 ndc = (pos / u_res) * 2.0 - 1.0; ndc.y = -ndc.y;
                gl_Position = vec4(ndc, u_depth, 1.0);
            }
            """,
            fragment_shader="""
            #version 330
            in vec2 v_p0; in vec2 v_p1; in float v_r0; in float v_r1; in vec2 v_local;
            uniform vec4 u_color; out vec4 f_color;
            float distToSeg(vec2 p, vec2 a, vec2 b) {
                vec2 pa = p - a, ba = b - a;
                float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
                return length(pa - ba * h);
            }
            void main() {
                vec2 ba = v_p1 - v_p0;
                float h = clamp(dot(v_local - v_p0, ba) / dot(ba, ba), 0.0, 1.0);
                float r = mix(v_r0, v_r1, h);
                float d = distToSeg(v_local, v_p0, v_p1);
                float alpha = 1.0 - smoothstep(r - 0.5, r + 0.5, d);
                if (alpha <= 0.0) discard;
                f_color = vec4(u_color.rgb, u_color.a * alpha);
            }
            """,
        )

        # Stamp Shader
        self.prog_stamp = self.ctx.program(
            vertex_shader="""
            #version 330
            in vec2 in_vert; in vec2 in_uv; // Quad
            in vec2 in_pos; in float in_size; in float in_angle; // Instance
            uniform vec2 u_res; uniform float u_depth;
            out vec2 v_uv;
            void main() {
                float s = sin(in_angle); float c = cos(in_angle);
                vec2 p = in_vert * in_size; 
                vec2 rotated = vec2(p.x * c - p.y * s, p.x * s + p.y * c);
                vec2 final_pos = in_pos + rotated;
                vec2 ndc = (final_pos / u_res) * 2.0 - 1.0; ndc.y = -ndc.y;
                gl_Position = vec4(ndc, u_depth, 1.0);
                v_uv = in_uv;
            }
            """,
            fragment_shader="""
            #version 330
            in vec2 v_uv;
            uniform sampler2D u_tex; uniform vec4 u_color;
            out vec4 f_color;
            void main() {
                vec4 tex = texture(u_tex, v_uv);
                // Tint texture with stroke color
                f_color = vec4(u_color.rgb, tex.a);
                if (f_color.a < 0.01) discard;
            }
            """,
        )

        quad_data = np.array(
            [
                [-0.5, -0.5, 0, 0],
                [0.5, -0.5, 1, 0],
                [0.5, 0.5, 1, 1],
                [-0.5, -0.5, 0, 0],
                [0.5, 0.5, 1, 1],
                [-0.5, 0.5, 0, 1],
            ],
            dtype=np.float32,
        )
        self.vbo_quad = self.ctx.buffer(quad_data.tobytes())

    def _render(
        self, sketch: Sketch, render_options: CialloRenderOptions
    ) -> Image.Image:
        """Renders the given sketch with the provided CialloRenderOptions."""

        if (
            self.fbo_tex.width != render_options.canvas_size[0]
            or self.fbo_tex.height != render_options.canvas_size[1]
        ):
            self.new_ctx(render_options)

        thickness = render_options.stroke_thickness
        brush_type = render_options.brush_type
        brush_texture_path = render_options.brush_texture_path
        spacing = render_options.spacing
        rotation_randomness = render_options.rotation_randomness
        noise_factor = render_options.noise_factor
        rng_seed = render_options.rng_seed
        color = render_options.stroke_color

        self.fbo.use()
        self.fbo.clear(*render_options.background_color)
        # print(*render_options.background_color)
        # self.ctx.clear(depth=1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.depth_func = "<"

        polylines = sketch_to_polylines(sketch)

        start_z = 0.9
        z_step = 1.8 / (len(polylines) + 1)
        current_z = start_z

        if brush_type == "vanilla":
            print("Rendering with vanilla brush...")

        elif brush_type == "textured":
            if not brush_texture_path:
                raise ValueError("Missing brush_texture_path")

            img = Image.open(brush_texture_path).convert("RGBA")
            tex = self.ctx.texture(img.size, 4, img.tobytes())
            tex.use(0)
            self.prog_stamp["u_tex"].value = 0
            self.prog_stamp["u_res"].value = render_options.canvas_size
            self.prog_stamp["u_color"].value = color

            # Disable depth test for blending accumulation
            self.ctx.disable(moderngl.DEPTH_TEST)

            for poly in polylines:
                actual_spacing = (
                    spacing if spacing is not None and spacing > 0 else thickness * 0.1
                )

                instance_data = build_stamp_instance_data(
                    [poly],
                    actual_spacing,
                    thickness,
                    rotation_randomness,
                    noise_factor,
                    rng_seed,
                )
                if len(instance_data) == 0:
                    continue

                vbo_inst = self.ctx.buffer(instance_data.tobytes())
                vao = self.ctx.vertex_array(
                    self.prog_stamp,
                    [
                        (self.vbo_quad, "2f 2f", "in_vert", "in_uv"),
                        (vbo_inst, "2f 1f 1f /i", "in_pos", "in_size", "in_angle"),
                    ],
                )
                vao.render(moderngl.TRIANGLES, instances=len(instance_data))
                vbo_inst.release()
                vao.release()

        data = self.fbo.read(components=4)
        img = Image.frombytes("RGBA", render_options.canvas_size, data).transpose(
            Image.FLIP_TOP_BOTTOM
        )
        return img
