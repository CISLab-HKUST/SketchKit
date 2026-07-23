from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    AutoencoderKL,
)
from diffusers import EulerAncestralDiscreteScheduler
from PIL import Image
import torch
import numpy as np
import cv2


def nms(x, t, s):
    x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), s)

    f1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    f2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    f3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    f4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)

    y = np.zeros_like(x)

    for f in [f1, f2, f3, f4]:
        np.putmask(y, cv2.dilate(x, kernel=f) == x, x)

    z = np.zeros_like(y, dtype=np.uint8)
    z[y > t] = 255
    return z


controlnet_conditioning_scale = 1.0
# prompt = "intricate, photorealistic, girl, black hair, closed mouth,looking at viewer, short hair, simple background, upper body"
# prompt = "dynamic action pose, a medieval knight warrior in full armor, crouching and leaning forward, left arm holding a large kite shield in front, right hand gripping a long sword held vertically, battle-ready stance, detailed steel plate armor with straps and cloth underlayer, realistic proportions, dramatic lighting, cinematic, ultra-detailed, sharp focus, high contrast, dust particles, 35mm film look, photorealistic, plain background"
# prompt = "An image of a classical Greco-Roman style marble bust, a young male, close-up left-side facial profile, with thick curly hair, perfectly idealized facial features, marble material, sharp focus, 4K resolution."
prompt = "A ultra-detailed macro photograph of a butterfly perched on a dew-covered flower, translucent wings with visible veins, iridescent scales, soft morning sunlight, shallow depth of field, creamy bokeh background, natural colors, 8k, photorealistic, high clarity, cinematic composition"
# prompt = "A dramatic Baroque-style marble bust of a young male, close-up left facial profile, thick curly hair sculpted in deep grooves, high-contrast chiaroscuro lighting, moody dark background, subtle dust particles, realistic marble veining, sharp focus, ultra-detailed, 4K."
# prompt = "A classical Greco-Roman marble bust of a young male, close-up left profile, thick curly hair, illuminated by cyberpunk neon rim lights (magenta and cyan glow), dark futuristic background, glossy marble highlights, crisp edges, sharp focus, ultra-detailed, 4K."
# prompt = "A minimalist contemporary museum photograph of a Greco-Roman marble bust of a young male, close-up left profile, thick curly hair, pristine white marble with gentle veining, soft diffused studio lighting, clean seamless background, centered composition, sharp focus, ultra-high detail, 4K."
negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"


eulera_scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
    "./hf_models/models--stabilityai--stable-diffusion-xl-base-1.0",
    subfolder="scheduler",
)


controlnet = ControlNetModel.from_pretrained(
    "xinsir/controlnet-scribble-sdxl-1.0", torch_dtype=torch.float16
)

# when test with other base model, you need to change the vae also.
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "./hf_models/models--stabilityai--stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    safety_checker=None,
    torch_dtype=torch.float16,
    scheduler=eulera_scheduler,
)

# Method 2
# if you use a sketch image total draw by yourself
control_path = "test_files/frame_0396.png"
controlnet_img = Image.open(control_path)

width, height = controlnet_img.size
ratio = np.sqrt(1024.0 * 1024.0 / (width * height))
new_width, new_height = int(width * ratio), int(height * ratio)
controlnet_img = controlnet_img.resize((new_width, new_height))

images = pipe(
    prompt,
    negative_prompt=negative_prompt,
    image=controlnet_img,
    controlnet_conditioning_scale=controlnet_conditioning_scale,
    width=new_width,
    height=new_height,
    num_inference_steps=30,
).images

images[0].save(f"gen-images/gen-butterfly-11-3.png")
