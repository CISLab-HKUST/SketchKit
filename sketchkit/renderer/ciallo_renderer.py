import math
import random
from typing import List, Tuple

import moderngl
import numpy as np
from PIL import Image
from sketchkit.core.sketch import Sketch, Curve


Point = Tuple[float, float, float]  # (x, y, pressure)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _sample_stamps_along_polyline(
    points: List[Point],
    base_spacing: float,
    thickness: float,
    rotation_randomness: float,
    noise_factor: float,
    rng: random.Random,
) -> List[Tuple[float, float, float, float]]:
    """
    Sample stamp instances along one polyline.

    Returns a list of tuples: (cx, cy, angle, scale)
    where scale is a scalar radius factor in pixels.
    """
    stamps = []
    if len(points) < 2:
        return stamps

    # Precompute segment lengths
    seg_lengths = []
    for i in range(len(points) - 1):
        x0, y0, _ = points[i]
        x1, y1, _ = points[i + 1]
        dx = x1 - x0
        dy = y1 - y0
        seg_len = math.hypot(dx, dy)
        seg_lengths.append(seg_len)

    if not seg_lengths:
        return stamps

    total_len = sum(seg_lengths)
    if total_len <= 1e-6:
        return stamps

    # Start at a small random offset so strokes don't all align identically
    t_global = rng.uniform(0.0, base_spacing)
    seg_index = 0
    seg_start_len = 0.0

    while t_global <= total_len and seg_index < len(seg_lengths):
        # Advance seg_index until t_global is within this segment
        while seg_index < len(seg_lengths) - 1 and t_global > seg_start_len + seg_lengths[seg_index]:
            seg_start_len += seg_lengths[seg_index]
            seg_index += 1

        seg_len = seg_lengths[seg_index]
        if seg_len <= 1e-6:
            # Degenerate segment, skip
            seg_index += 1
            continue

        local_t = (t_global - seg_start_len) / seg_len
        local_t = _clamp(local_t, 0.0, 1.0)

        x0, y0, p0 = points[seg_index]
        x1, y1, p1 = points[seg_index + 1]

        cx = _lerp(x0, x1, local_t)
        cy = _lerp(y0, y1, local_t)
        pressure = _lerp(p0, p1, local_t)
        pressure = _clamp(pressure, 0.0, 1.0)

        # Tangent and angle
        dx = x1 - x0
        dy = y1 - y0
        if dx == 0.0 and dy == 0.0:
            # Fallback angle
            angle = 0.0
        else:
            angle = math.atan2(dy, dx)

        # Random rotation around tangent
        angle += rng.uniform(-rotation_randomness, rotation_randomness)

        # Base radius/scale from thickness and pressure
        scale = thickness * (1.0 + pressure) * 0.5  # scale is radius

        # Radius noise (multiplicative)
        if noise_factor > 0.0:
            noise = rng.uniform(-noise_factor, noise_factor)
            scale *= max(0.0, 1.0 + noise)

        # Jitter position slightly along the normal for extra organic feel
        if noise_factor > 0.0:
            # Normal to (dx, dy)
            length = math.hypot(dx, dy)
            if length > 1e-6:
                nx = -dy / length
                ny = dx / length
                offset = noise_factor * thickness * rng.uniform(-1.0, 1.0)
                cx += nx * offset
                cy += ny * offset

        stamps.append((cx, cy, angle, scale))

        # Next target distance along stroke, with jitter
        spacing_jitter = 1.0
        if noise_factor > 0.0:
            spacing_jitter += noise_factor * rng.uniform(-1.0, 1.0)
            spacing_jitter = max(0.1, spacing_jitter)

        t_global += base_spacing * spacing_jitter

    return stamps

def _sample_cubic_bezier(curve: Curve, ts: np.ndarray) -> np.ndarray:
    """
    Sample positions along a cubic Bézier defined by curve.np_points
    at parameter values ts ∈ [0, 1].

    Returns
    -------
    np.ndarray
        Array of shape (len(ts), 2) with (x, y) positions.
    """
    # curve.np_points: shape (4, 2) -> P0, P1, P2, P3
    pts = curve.np_points.astype(np.float32)
    P0, P1, P2, P3 = pts  # each is (2,)

    # Degenerate curve: all control points identical
    if np.allclose(pts, pts[0]):
        return np.repeat(pts[0][None, :], len(ts), axis=0)

    t = ts.reshape(-1, 1).astype(np.float32)  # (N, 1)
    mt = 1.0 - t

    # Cubic Bézier: B(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3
    B = (
        (mt ** 3) * P0
        + 3.0 * (mt ** 2) * t * P1
        + 3.0 * mt * (t ** 2) * P2
        + (t ** 3) * P3
    )  # shape (N, 2)

    return B


def sketch_to_polylines(
    sketch: Sketch,
    samples_per_curve: int = 10,
    default_pressure: float = 1.0,
) -> List[List[Tuple[float, float, float]]]:
    """
    Convert a Sketch into polylines with per-point pressure.

    Each Curve in each Path becomes one polyline sampled along its cubic Bézier.

    Parameters
    ----------
    sketch : Sketch
        The sketch to convert.
    samples_per_curve : int, optional
        Number of samples per curve (including endpoints).
        More samples → smoother polylines. Default is 32.
    default_pressure : float, optional
        Pressure used when a vertex has no pressure set. Default is 1.0.

    Returns
    -------
    List[List[Tuple[float, float, float]]]
        A list of polylines. Each polyline is a list of (x, y, pressure) tuples.
    """
    polylines: List[List[Tuple[float, float, float]]] = []

    # Parameter positions along each curve in [0, 1]
    ts = np.linspace(0.0, 1.0, samples_per_curve, dtype=np.float32)

    for path in sketch.paths:
        for curve in path.curves:
            if not isinstance(curve, Curve):
                continue

            # Sample positions along this curve (avoid svgpathtools)
            pts = _sample_cubic_bezier(curve, ts)  # (N, 2)

            # Endpoint pressures
            p_start = curve.p_start.pressure
            p_end = curve.p_end.pressure

            # Fallback logic:
            # - both None → default_pressure
            # - one None → use the other for both
            if p_start is None and p_end is None:
                p_start = p_end = np.float32(default_pressure)
            elif p_start is None and p_end is not None:
                p_start = p_end
            elif p_end is None and p_start is not None:
                p_end = p_start

            p_start = float(p_start)
            p_end = float(p_end)

            polyline: List[Tuple[float, float, float]] = []

            # Build (x, y, pressure) with linear pressure interpolation
            for i, (x, y) in enumerate(pts):
                t = float(ts[i])  # in [0, 1]
                pressure = (1.0 - t) * p_start + t * p_end
                polyline.append((float(x), float(y), float(pressure)))

            polylines.append(polyline)

    return polylines


def _build_vanilla_vertices(
    polylines: List[List[Tuple[float, float, float]]],
    thickness: float,
) -> np.ndarray:
    
    """
    Build vertex buffer for articulated 'vanilla' strokes.

    Each polyline edge (p_i -> p_{i+1}) becomes one capsule segment
    rendered as TWO triangles (6 vertices).

    Vertex layout per-vertex:
        in_corner (2)  : -1/+1, local corner id
        in_p0     (2)  : segment start point in canvas space
        in_p1     (2)  : segment end point in canvas space
        in_r0     (1)  : radius at p0 (pixels)
        in_r1     (1)  : radius at p1 (pixels)
        in_len    (1)  : segment length (pixels)

    Total = 2 + 2 + 2 + 1 + 1 + 1 = 9 floats per vertex.
    """

    segments: List[Tuple[float, float, float, float, float, float, float]] = []

    for poly in polylines:
        if len(poly) < 2:
            continue
        for i in range(len(poly) - 1):
            x0, y0, p0 = poly[i]
            x1, y1, p1 = poly[i + 1]

            # Radii from thickness * pressure. You can tweak mapping.
            r0 = 0.5 * thickness * ( 1.0 + float(p0))
            r1 = 0.5 * thickness * ( 1.0 + float(p1))

            dx = x1 - x0
            dy = y1 - y0
            seg_len = float(np.hypot(dx, dy))
            if seg_len < 1e-4:
                # Degenerate edge; skip (or treat as a dot if you want)
                continue

            segments.append((x0, y0, x1, y1, r0, r1, seg_len))

    if not segments:
        return np.zeros((0, 9), dtype="f4")

    # Two triangles per segment: 6 corners
    quad_corners = np.array(
        [
            [-1.0, -1.0],
            [+1.0, -1.0],
            [+1.0, +1.0],
            [-1.0, -1.0],
            [+1.0, +1.0],
            [-1.0, +1.0],
        ],
        dtype="f4",
    )

    num_segments = len(segments)
    num_vertices = num_segments * 6
    vertices = np.zeros((num_vertices, 9), dtype="f4")

    v = 0
    for (x0, y0, x1, y1, r0, r1, seg_len) in segments:
        for cx, cy in quad_corners:
            vertices[v, 0] = cx          # in_corner.x
            vertices[v, 1] = cy          # in_corner.y
            vertices[v, 2] = x0          # in_p0.x
            vertices[v, 3] = y0          # in_p0.y
            vertices[v, 4] = x1          # in_p1.x
            vertices[v, 5] = y1          # in_p1.y
            vertices[v, 6] = r0          # in_r0
            vertices[v, 7] = r1          # in_r1
            vertices[v, 8] = seg_len     # in_len
            v += 1

    return vertices.astype("f4")


class CialloRenderer:
    def __init__(self, canvas_size=512, canvas_color=(1, 1, 1, 0)):
        """
        Args:
            canvas_size: width/height in pixels.
            canvas_color: RGBA clear color in [0,1].
        """
        self.canvas_size = canvas_size
        self.canvas_color = canvas_color

    def render(
        self,
        sketch: Sketch,
        brush_image_path: str | None = None,
        brush_type: str = "stamp",
        rotation_randomness: float = 0.3,
        thickness: float = 12.0,
        noise_factor: float = 0.25,
        interval: float = 0.7,
        color: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        output_path: str = "output.png",
        rng_seed: int = 42,
    ) -> Image.Image:
        """
        Render a sketch using Ciallo in ModernGL.

        - brush_type="stamp"  : texture stamp
        """
        width = height = self.canvas_size

        polylines = sketch_to_polylines(sketch)

        ctx = moderngl.create_standalone_context()
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        color_tex = ctx.texture((width, height), components=4)
        fbo = ctx.framebuffer(color_attachments=[color_tex])
        fbo.use()
        ctx.viewport = (0, 0, width, height)
        fbo.clear(*self.canvas_color)

        if brush_type == "stamp":
            img = self._render_stamp(
                ctx,
                fbo,
                polylines,
                brush_image_path=brush_image_path,
                rotation_randomness=rotation_randomness,
                thickness=thickness,
                noise_factor=noise_factor,
                interval=interval,
                color=color,
                rng_seed=rng_seed,
                output_path=output_path,
            )
        else:
            raise ValueError("brush_type must be 'vanilla' or 'stamp'")

        return img

    # ---------------- STAMP: textured dot brush ---------------- #

    def _render_stamp(
        self,
        ctx: moderngl.Context,
        fbo: moderngl.Framebuffer,
        polylines: List[List[Tuple[float, float, float]]],
        brush_image_path: str | None,
        rotation_randomness: float,
        thickness: float,
        noise_factor: float,
        interval: float,
        color: Tuple[float, float, float, float],
        rng_seed: int,
        output_path: str,
    ) -> Image.Image:
        """
        Same idea as before: place stamps (quads) along polylines.
        This is *supposed* to be dots – increase density (smaller `interval`)
        if you want it to look more continuous.
        """
        if brush_image_path is None:
            raise ValueError("brush_image_path must be provided for stamp brush type.")

        width, height = self.canvas_size, self.canvas_size
        rng = random.Random(rng_seed)

        # Load brush texture
        brush_img = Image.open(brush_image_path).convert("RGBA")
        bw, bh = brush_img.size
        brush_tex = ctx.texture((bw, bh), components=4, data=brush_img.tobytes())
        brush_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        brush_tex.repeat_x = False
        brush_tex.repeat_y = False

        # Shaders
        prog = ctx.program(
            vertex_shader="""
            #version 330

            in vec2 in_pos;      // unit quad corner [-0.5,0.5]
            in vec2 in_uv;       // brush texture UV
            in vec2 in_center;   // canvas-space center
            in float in_angle;   // rotation
            in float in_scale;   // size in pixels

            uniform vec2 u_canvas_size;

            out vec2 v_uv;

            void main() {
                float s = sin(in_angle);
                float c = cos(in_angle);

                vec2 p = in_pos * in_scale;
                vec2 pr = vec2(
                    p.x * c - p.y * s,
                    p.x * s + p.y * c
                );
                vec2 world = in_center + pr;

                vec2 ndc = (world / u_canvas_size) * 2.0 - 1.0;
                ndc.y = -ndc.y;

                gl_Position = vec4(ndc, 0.0, 1.0);
                v_uv = in_uv;
            }
            """,
            fragment_shader="""
            #version 330

            in vec2 v_uv;

            uniform sampler2D u_brush;
            uniform vec4 u_color;
            uniform float u_opacity;

            out vec4 f_color;

            void main() {
                vec4 tex = texture(u_brush, v_uv);
                float coverage = tex.a;
                float alpha = coverage * u_opacity * u_color.a;
                if (alpha <= 0.001) discard;
                f_color = vec4(u_color.rgb, alpha);
            }
            """,
        )

        prog["u_canvas_size"].value = (float(width), float(height))
        prog["u_color"].value = color
        prog["u_opacity"].value = 1.0

        brush_tex.use(location=0)
        prog["u_brush"].value = 0

        # Sample stamps along stroke
        base_spacing = interval
        all_stamps = []
        for poly in polylines:
            if len(poly) < 2:
                continue
            stamps = _sample_stamps_along_polyline(
                poly,
                base_spacing=base_spacing,
                thickness=thickness,
                rotation_randomness=rotation_randomness,
                noise_factor=noise_factor,
                rng=rng,
            )
            all_stamps.extend(stamps)

        if not all_stamps:
            raw = fbo.read(components=4)
            img = Image.frombytes("RGBA", (width, height), raw, "raw", "RGBA", 0, -1)
            img.save(output_path)
            return img

        quad_positions = np.array(
            [
                [-0.5, -0.5],
                [+0.5, -0.5],
                [+0.5, +0.5],
                [-0.5, -0.5],
                [+0.5, +0.5],
                [-0.5, +0.5],
            ],
            dtype="f4",
        )
        quad_uvs = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ],
            dtype="f4",
        )

        num_stamps = len(all_stamps)
        num_vertices = num_stamps * 6
        vertices = np.zeros((num_vertices, 8), dtype="f4")

        v = 0
        for (cx, cy, angle, scale) in all_stamps:
            for i in range(6):
                px, py = quad_positions[i]
                u, vv = quad_uvs[i]
                vertices[v, 0] = px
                vertices[v, 1] = py
                vertices[v, 2] = u
                vertices[v, 3] = vv
                vertices[v, 4] = cx
                vertices[v, 5] = cy
                vertices[v, 6] = angle
                vertices[v, 7] = scale
                v += 1

        vbo = ctx.buffer(vertices.tobytes())
        vao = ctx.vertex_array(
            prog,
            [
                (
                    vbo,
                    "2f 2f 2f 1f 1f",  # in_pos, in_uv, in_center, in_angle, in_scale
                    "in_pos",
                    "in_uv",
                    "in_center",
                    "in_angle",
                    "in_scale",
                )
            ],
        )

        vao.render(mode=moderngl.TRIANGLES)

        raw = fbo.read(components=4)
        img = Image.frombytes("RGBA", (width, height), raw, "raw", "RGBA", 0, -1)
        img.save(output_path)
        return img