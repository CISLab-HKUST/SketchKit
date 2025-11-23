import os
import sys
import pygame
import moderngl
from PIL import Image

# (Optional) DPI awareness on Windows
os.environ.setdefault('SDL_WINDOWS_DPI_AWARENESS', 'permonitorv2')

pygame.init()
pygame.display.init()

# Request a core GL context
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, True)

# Create a hidden window (required to get GL context, but it won't show)
win_size = (800, 800)
pygame.display.set_mode(
    win_size,
    flags=pygame.OPENGL | pygame.DOUBLEBUF | pygame.HIDDEN,
)

# ModernGL context
ctx = moderngl.get_context()
drawable_w, drawable_h = ctx.screen.size
ctx.viewport = (0, 0, drawable_w, drawable_h)

VERTEX_SRC = """
#version 330 core
const vec3 vertices[3] = vec3[3](
    vec3( 0.0,  0.4, 0.0),
    vec3(-0.4, -0.3, 0.0),
    vec3( 0.4, -0.3, 0.0)
);
void main() { gl_Position = vec4(vertices[gl_VertexID], 1.0); }
"""

FRAGMENT_SRC = """
#version 330 core
out vec4 out_color;
void main() { out_color = vec4(1.0); }  // white
"""

program = ctx.program(vertex_shader=VERTEX_SRC, fragment_shader=FRAGMENT_SRC)
vao = ctx.vertex_array(program, [])

# Render once
ctx.clear(0.0, 0.0, 0.0, 1.0)
vao.render(mode=moderngl.TRIANGLES, vertices=3)
ctx.finish()

# Read framebuffer into PIL image
pixels = ctx.screen.read(components=3, alignment=1)  # RGB
img = Image.frombytes("RGB", (drawable_w, drawable_h), pixels)
img = img.transpose(Image.FLIP_TOP_BOTTOM)  # OpenGL origin is bottom-left
img.save("output.png")
print(f"Saved off-screen render as output.png ({drawable_w}x{drawable_h})")

pygame.quit()
sys.exit()
