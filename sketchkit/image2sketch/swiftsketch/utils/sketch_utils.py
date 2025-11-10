
import torch
from sketchkit.image2sketch import pydiffvg
from PIL import Image
import os
import numpy as np
from svgpathtools import CubicBezier
from svgpathtools import svg2paths2
from io import StringIO
import torch.nn.functional as F
import wandb
import matplotlib  # Import the main matplotlib module
matplotlib.use('Agg')  # Set the backend to 'Agg' for faster, non-interactive plotting
import matplotlib.pyplot as plt
import io
import cairosvg
from torchvision.transforms.functional import normalize
import tempfile
from scipy.ndimage import  binary_erosion, binary_dilation
import re


def fix_image_scale(im):
    im_np = np.array(im) / 255
    height, width = im_np.shape[0], im_np.shape[1]
    max_len = max(height, width) + 20
    new_background = np.ones((max_len, max_len, 3))
    y, x = max_len // 2 - height // 2, max_len // 2 - width // 2
    new_background[y: y + height, x: x + width] = im_np
    new_background = (new_background / new_background.max()
                      * 255).astype(np.uint8)
    new_im = Image.fromarray(new_background)
    return new_im


def denormalize_points(points, scaling_factor, canvas_width):
    points = points.clone()
    points = points / scaling_factor
    points = (points + 1) / 2
    points = points * canvas_width
    return points


def save_key(target_file, data, key):
    ext = os.path.splitext(os.path.basename(target_file))[-1]
    if ext == ".npy":
        read_dictionary = np.load(target_file, allow_pickle=True).item()
        read_dictionary[key] = data
        np.save(target_file, read_dictionary)
    else:
        read_dictionary = dict(np.load(target_file, allow_pickle=True))
        read_dictionary[key] = data
        np.savez_compressed(target_file, **read_dictionary)
    


def load_compressed_npz(file_path, svg_keys=[], features_key="CLIPMiddle_layer4_features"):
    """Load compressed .npz file and reconstruct the original objects."""
    data = np.load(file_path, allow_pickle=True)

    # Reconstruct the image from bytes
    img_bytes = data["image"].tobytes()
    image = Image.open(io.BytesIO(img_bytes))

    result = {"image": image}

    if "mask" in data:
        result["mask"] = torch.from_numpy(data["mask"]).float()

    if "attn_map" in data:
        result["attn_map"] = torch.from_numpy(data["attn_map"])

    if features_key in data:
        result[features_key] = torch.from_numpy(data[features_key])

    for key in svg_keys:
        if key in data:
            result[key] = data[key].item()

    return result

def load_entry(file_path, svg_keys=[], features_key="CLIPMiddle_layer4_features"):
    """
    Load a .npy or .npz file into a Python dict.
    """
    ext = os.path.splitext(os.path.basename(file_path))[-1]
    if ext == ".npy":
        return np.load(file_path, allow_pickle=True).item()
    elif ext == ".npz":
        return load_compressed_npz(file_path, svg_keys, features_key)
    else:
        raise ValueError(f"Unsupported file type: {file_path}. Only '.npy' and '.npz' are allowed.")
  

def svg_from_points(control_points, canvas_width, canvas_height):
    #control_points_batch shape (num_paths, 4,2 )

    shapes = []
    shape_groups = []
    num_paths = control_points.shape[0]  # control_points is [num_paths, 4, 2]
    
    for i in range(num_paths):
        path_points = control_points[i] # shape [4, 2]
        path = pydiffvg.Path(num_control_points=torch.tensor([2]),
                        points=path_points,
                        stroke_width=torch.tensor(1.0),
                        is_closed=False)
        shapes.append(path)
        shape_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]),
                                        fill_color=None,
                                        stroke_color=torch.tensor([0.0, 0.0, 0.0, 1.0]))
        shape_groups.append(shape_group)

    svg_content= generate_svg_content(canvas_width, canvas_height, shapes, shape_groups)
    return svg_content





def svg_to_pil(svg_data, size=(224, 224)):
    # Convert SVG data to PNG using cairosvg
    png_data = cairosvg.svg2png(bytestring=svg_data)
    
    # Load the PNG data into a PIL Image
    image = Image.open(io.BytesIO(png_data))
    
    # Resize the image to the desired size (224x224 by default)
    # image = image.resize(size, Image.ANTIALIAS)
    
    return image


def generate_svg_content(canvas_width, canvas_height, shapes, shape_groups):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".svg") as temp_file:
        temp_filename = temp_file.name
    
    # Save the SVG to the temporary file
    pydiffvg.save_svg(temp_filename, canvas_width, canvas_height, shapes, shape_groups)
    
    # Read the SVG content from the temporary file
    with open(temp_filename, "r") as file:
        svg_content = file.read()
    
    # Clean up the temporary file
    os.remove(temp_filename)
    
    return svg_content


def render_paths(control_points, canvas_width, canvas_height, save_svg=False, return_svg_content=False, output_dir="", name=""):
    shapes = []
    shape_groups = []
    num_paths = control_points.shape[0]  # control_points is [num_paths, 4, 2]
    
    for i in range(num_paths):
        path_points = control_points[i] # shape [4, 2]
        path = pydiffvg.Path(num_control_points=torch.tensor([2]),
                        points=path_points,
                        stroke_width=torch.tensor(1.0),
                        is_closed=False)
        shapes.append(path)
        shape_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]),
                                        fill_color=None,
                                        stroke_color=torch.tensor([0.0, 0.0, 0.0, 1.0]))
        shape_groups.append(shape_group)

    scene_args = pydiffvg.RenderFunction.serialize_scene(canvas_width, canvas_height, shapes, shape_groups)
    render = pydiffvg.RenderFunction.apply
    img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)

    svg_content= None 
    if return_svg_content:
        svg_content = generate_svg_content(canvas_width, canvas_height, shapes, shape_groups)

    if save_svg:
        pydiffvg.save_svg('{}/{}.svg'.format(output_dir, name), canvas_width, canvas_height, shapes, shape_groups)
    
    return img ,svg_content



def rander_image_from_points(control_points_batch, canvas_width, canvas_height, return_svg_content=False):
    #control_points_batch shape (bs, num_paths, 4,2 )
    # Render the images from the control points in parallel using map
    device = control_points_batch.device
    bs = control_points_batch.shape[0]


    results = list(map(lambda args: render_paths(*args, return_svg_content=return_svg_content), 
                   zip(control_points_batch, 
                       [canvas_width] * bs, 
                       [canvas_height] * bs)))

    # Separate the results into two lists: images and svg_content_list
    images, svg_content_list = zip(*results)

    # Convert the tuples to lists if needed
    images = list(images)
    svg_content_list = list(svg_content_list)

    
    # Stack the rendered images and move to device
    rendered_images = torch.stack(images).to(device)
    
    # batch processing to the images
    opacity = rendered_images[:, :, :, 3:4]
    rendered_images_final = opacity * rendered_images[:, :, :, :3] + torch.ones(rendered_images.shape[0], rendered_images.shape[1], rendered_images.shape[2], 3, device=device) * (1 - opacity)
    rendered_images_final = rendered_images_final[:, :, :, :3]
    return rendered_images_final, svg_content_list



def save_svg_from_points(control_points, canvas_width, canvas_height, save_path):
    #control_points_batch shape (num_paths, 4,2 )

    shapes = []
    shape_groups = []
    num_paths = control_points.shape[0]  # control_points is [num_paths, 4, 2]
    
    for i in range(num_paths):
        path_points = control_points[i] # shape [4, 2]
        path = pydiffvg.Path(num_control_points=torch.tensor([2]),
                        points=path_points,
                        stroke_width=torch.tensor(1.0),
                        is_closed=False)
        shapes.append(path)
        shape_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]),
                                        fill_color=None,
                                        stroke_color=torch.tensor([0.0, 0.0, 0.0, 1.0]))
        shape_groups.append(shape_group)

    
    # Save the SVG to the temporary file
    pydiffvg.save_svg(save_path, canvas_width, canvas_height, shapes, shape_groups)
    
    return 

   



def render_image_from_norm_points(points, scaling_factor, canvas_size= 224): 
    #return list of rendered images in the length of the batch
    #convert the normalized points back to the original range 
    points = points / scaling_factor
    points = (points + 1) / 2
    points = points * canvas_size
    # Create rander image from control_points (sample)
    net_rendered_image, _ = rander_image_from_points(points,canvas_size, canvas_size)
    output_sketch_lst= [convert_image_to_pil(img) for img in net_rendered_image]
    return output_sketch_lst

def render_image_from_norm_points_svg(points, scaling_factor, canvas_size=224): 
    #return list of rendered images in the length of the batch
    #convert the normalized points back to the original range 
    points = points / scaling_factor
    points = (points + 1) / 2
    points = points * canvas_size
    # Create rander image from control_points (sample)
    _ , svg_content  = rander_image_from_points(points,canvas_size, canvas_size, return_svg_content=True)
    return svg_content #list 



def calculate_highest_points(strokes):
    """
    Vectorized function to calculate the maximum y-coordinate (highest point) for each Bézier curve in the strokes.
    Args:
        strokes: Tensor of shape [Nstrokes, 4, 2], representing strokes with 4 control points (x, y).
    Returns:
        max_y_points: Tensor of shape [Nstrokes] representing the highest point (max y) for each stroke.
    """
    device = strokes.device  # Get the device of the input tensor

    # Extract y-coordinates from the control points
    y0 = strokes[:, 0, 1]
    y1 = strokes[:, 1, 1]
    y2 = strokes[:, 2, 1]
    y3 = strokes[:, 3, 1]
    
    # Calculate coefficients for the y(t) cubic polynomial for each stroke
    a = -y0 + 3 * y1 - 3 * y2 + y3
    b = 3 * y0 - 6 * y1 + 3 * y2
    c = -3 * y0 + 3 * y1
    d = y0

    # Compute discriminants to find real roots for t in [0, 1]
    discriminant = b**2 - 3 * a * c
    valid_discriminant = discriminant >= 0

    # Initialize t with boundary values 0 and 1
    roots = torch.zeros((strokes.size(0), 3), device=device)  # Ensure roots tensor is on the same device
    roots[:, 1] = 1

    # Calculate real roots for cases where discriminant is non-negative
    if valid_discriminant.any():
        sqrt_discriminant = torch.sqrt(discriminant[valid_discriminant].to(device))
        t1 = (-b[valid_discriminant] + sqrt_discriminant) / (3 * a[valid_discriminant])
        t2 = (-b[valid_discriminant] - sqrt_discriminant) / (3 * a[valid_discriminant])
        
        roots[valid_discriminant, 0] = t1
        roots[valid_discriminant, 2] = t2
    
    # Clamp roots to [0, 1] and evaluate y at these points
    roots = roots.clamp(0, 1)

    # Evaluate y(t) at the roots for each stroke
    y_values = (
        (1 - roots)**3 * y0.unsqueeze(-1) + 
        3 * (1 - roots)**2 * roots * y1.unsqueeze(-1) + 
        3 * (1 - roots) * roots**2 * y2.unsqueeze(-1) + 
        roots**3 * y3.unsqueeze(-1)
    )

    # Take the max y-value among roots for each stroke
    max_y_points, _ = y_values.max(dim=-1)
    
    return max_y_points



def calculate_length(strokes, num_samples=100):
    """
    Approximates the length of each Bézier curve using sampling.
    Args:
        strokes: Tensor of shape [Nstrokes, 4, 2], representing strokes with 4 control points (x, y).
        num_samples: Number of samples along the curve to approximate length.
    
    Returns:
        lengths: Tensor of shape [Nstrokes] representing the length of each stroke.
    """
    device = strokes.device
    Nstrokes = strokes.shape[0]
    
    # Generate t values from 0 to 1
    t_values = torch.linspace(0, 1, num_samples, device=device).view(1, -1, 1)
    
    # Control points P0, P1, P2, P3 for Bézier curves
    P0, P1, P2, P3 = strokes[:, 0], strokes[:, 1], strokes[:, 2], strokes[:, 3]
    
    # Calculate points on the curve for each t
    curve_points = (1 - t_values)**3 * P0[:, None, :] + \
                   3 * (1 - t_values)**2 * t_values * P1[:, None, :] + \
                   3 * (1 - t_values) * t_values**2 * P2[:, None, :] + \
                   t_values**3 * P3[:, None, :]
    
    # Calculate distances between consecutive points
    distances = torch.sqrt(torch.sum((curve_points[:, 1:] - curve_points[:, :-1])**2, dim=-1))
    
    # Sum up distances to approximate the length of each stroke
    lengths = torch.sum(distances, dim=1)
    
    return lengths


def get_thick_contour_tensor(mask, canvas_width, canvas_height):
    # Resize to (canvas_width, canvas_height)
    mask_resized = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(canvas_width, canvas_height), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
    # mask to binary
    mask_resized[mask_resized < 0.5] = 0
    mask_resized[mask_resized >= 0.5] = 1

        # Convert tensor to NumPy array
    binary_array = mask_resized.numpy()
    
    # Perform binary erosion (This shrinks the foreground region, leaving the core area.)
    eroded = binary_erosion(binary_array, structure=np.ones((10, 10)))
    
    # Perform binary dilation (This expands the foreground region, enlarging the boundary.)
    dilated = binary_dilation(binary_array, structure=np.ones((5, 5)))
    
    # Find the thick contour
    thick_contour = dilated.astype(np.uint8) - eroded.astype(np.uint8)
    
    # Convert back to PyTorch tensor
    thick_contour_tensor = torch.tensor(thick_contour)
    return thick_contour_tensor
    


def sort_by_contour(opt_svg, mask):
    # Save SVG text to a .svg file to read it with pydiffvg
    with tempfile.NamedTemporaryFile(suffix=".svg", delete=True) as temp_svg:
        # Write the SVG content to the temporary file
        temp_svg.write(opt_svg.encode('utf-8'))
        temp_svg.flush()  # Ensure the content is written to the file

        # Use pydiffvg to parse the SVG
        canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(temp_svg.name)

    
    # Render each stroke separately to count intersecting pixels
    stroke_to_pixels = {}  # Map stroke index to its pixel locations
    intersection_pixel_count = []  # To store number of intersecting pixels

    thick_contour_tensor= get_thick_contour_tensor(mask, canvas_width, canvas_height)
    
    contour_mask = thick_contour_tensor.numpy() 
    
    for i, shape in enumerate(shapes):
        # Create a copy of shape groups with only the current stroke active
        # single_shape_group = [shape_groups[i]]
        path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
                                        fill_color = None,
                                        stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0]))
        single_shape_group = [path_group]
    
        # Render the single stroke
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_width, canvas_height, [shape], single_shape_group
        )
        render = pydiffvg.RenderFunction.apply
        img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)
    
        # Convert rendered image to binary mask
        img = img[:, :, 3:4] * img[:, :, :3] + \
            torch.ones(img.shape[0], img.shape[1], 3,
                       device=pydiffvg.device) * (1 - img[:, :, 3:4])
        img = img[:, :, :3].cpu().numpy()
        mask = ~np.any(img > 0, axis=-1)  # Binary mask for stroke pixels, note that it shoule be inverted with ~
    
        # Store the mask
        stroke_to_pixels[i] = mask
    
        # Count intersecting pixels
        intersection_pixels = np.logical_and(mask, contour_mask)
        intersection_count = np.sum(intersection_pixels)
        intersection_pixel_count.append((i, intersection_count))
    
    # Sort strokes by intersection pixel count in descending order
    sorted_strokes = sorted(intersection_pixel_count, key=lambda x: x[1], reverse=True)
    sorted_indices = [stroke[0] for stroke in sorted_strokes]
    return sorted_indices

# This is the main finction, we will rnder each stroke and count how many pixels intersect with the contour

def sort_by_contour_and_attn(opt_svg, mask, attn_map):

    with tempfile.NamedTemporaryFile(suffix=".svg", delete=True) as temp_svg:
        # Write the SVG content to the temporary file
        temp_svg.write(opt_svg.encode('utf-8'))
        temp_svg.flush()  # Ensure the content is written to the file

        # Use pydiffvg to parse the SVG
        canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(temp_svg.name)

    
    # Render each stroke separately to count intersecting pixels
    stroke_to_pixels = {}  # Map stroke index to its pixel locations
    intersection_pixel_count = []  # To store number of intersecting pixels

    thick_contour_tensor= get_thick_contour_tensor(mask, canvas_width, canvas_height)
    contour_mask = thick_contour_tensor.numpy() 

    attn_resized = F.interpolate(attn_map.unsqueeze(0).unsqueeze(0), size=(canvas_width, canvas_height), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
    attn_map= attn_resized.numpy() 

    for i, shape in enumerate(shapes):
        # Create a copy of shape groups with only the current stroke active
        path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([0]),
                                        fill_color = None,
                                        stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0]))
        single_shape_group = [path_group]
    
        # Render the single stroke
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_width, canvas_height, [shape], single_shape_group
        )
        render = pydiffvg.RenderFunction.apply
        img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)
    
        # Convert rendered image to binary mask
        img = img[:, :, 3:4] * img[:, :, :3] + \
            torch.ones(img.shape[0], img.shape[1], 3,
                       device=pydiffvg.device) * (1 - img[:, :, 3:4])
        img = img[:, :, :3].cpu().numpy()
        mask = ~np.any(img > 0, axis=-1)  # Binary mask for stroke pixels, note that it shoule be inverted with ~
    
        # Store the mask
        stroke_to_pixels[i] = mask
    
        # Count intersecting pixels
        intersection_pixels = np.logical_and(mask, contour_mask)
        intersection_count = np.sum(intersection_pixels)
        intersection_pixel_count.append((i, intersection_count))
    
    # Separate strokes with intersection count 0 and non-zero
    sorted_non_zero_intersection_strokes = sorted([(i, count) for i, count in intersection_pixel_count if count > 0], key=lambda x: x[1], reverse=True)
    zero_intersection_strokes = [(i, count) for i, count in intersection_pixel_count if count == 0]

    
    # To store average attention values for zero intersection strokes
    avg_attention_values = []
    
    for stroke_index, _ in zero_intersection_strokes:
        # Render stroke in white on a black background
        shape = shapes[stroke_index]
        path_group = pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([0]),
            fill_color=None,
            stroke_color=torch.tensor([1.0, 1.0, 1.0, 1.0])  # White color for the stroke
        )
        single_shape_group = [path_group]

        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_width, canvas_height, [shape], single_shape_group
        )
        render = pydiffvg.RenderFunction.apply
        img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)
    
        # Convert to binary mask for stroke pixels (white stroke on black background)
        img = img[:, :, 3:4] * img[:, :, :3] + \
            torch.zeros(img.shape[0], img.shape[1], 3, device=pydiffvg.device) * (1 - img[:, :, 3:4])
        img = img[:, :, :3].cpu().numpy()
        stroke_mask = np.any(img > 0.9, axis=-1)  # Pixels belonging to the stroke
    
        # Number of stroke pixels
        num_stroke_pixels = np.sum(stroke_mask)
    
        # Multiply stroke mask with attention map and calculate average attention value
        stroke_attention_values = attn_map * stroke_mask
        total_attention_value = np.sum(stroke_attention_values)
        avg_attention_value = total_attention_value / num_stroke_pixels if num_stroke_pixels > 0 else 0
    
        avg_attention_values.append((stroke_index, avg_attention_value))
    
    # Sort zero intersection strokes by average attention value (descending order)
    sorted_zero_intersection_strokes = sorted(avg_attention_values, key=lambda x: x[1], reverse=True)
    
    # Combine both sorted lists
    final_sorted_strokes = sorted_non_zero_intersection_strokes + sorted_zero_intersection_strokes
    
    # Output the final sorted stroke indices
    sorted_indices = [i for i, _ in final_sorted_strokes]
    return sorted_indices
    


def sort_strokes(strokes, by='highest_point', SVG_content=None, mask=None, attn_map=None):
    """
    Sorts strokes by the specified criterion ('highest_point' or 'length' or 'contour).
    Args:
        strokes: Tensor of shape [Nstrokes, 4, 2] representing strokes with 4 control points.
        by: Criterion for sorting, either 'highest_point' or 'length' or 'contour.
    Returns:
        Tensor of sorted strokes with the same shape as strokes.
    """
    if by == 'highest_point':
        # Calculate the highest point for each stroke
        max_y_points = calculate_highest_points(strokes)
        # Sort indices by the highest point (descending)
        sorted_indices = torch.argsort(max_y_points, descending=True)
    elif by == 'length':
        # Calculate the length for each stroke
        lengths = calculate_length(strokes)
        # Sort indices by the length (descending)
        sorted_indices = torch.argsort(lengths, descending=True)
    elif by == 'contour':
        sorted_indices= sort_by_contour(SVG_content, mask)
    elif by == 'contour_and_attn':
        sorted_indices= sort_by_contour_and_attn(SVG_content, mask, attn_map)
        
    else:
        raise ValueError("Invalid sorting criterion. Use 'highest_point' or 'length' or 'contour' or 'contour_and_attn'.")


    sorted_strokes = strokes[sorted_indices]

    return sorted_strokes





def convert_image_to_pil(img):
    sketch=img.cpu().detach().numpy()
    sketch = Image.fromarray((sketch * 255).astype('uint8'), 'RGB')
    return sketch

def create_masked_image(image, mask):
    """Apply the predicted mask to the input image."""

    # Convert the image to a numpy array and normalize
    im_np = np.array(image).astype(np.float32)
    if im_np.max() > 0:
        im_np = im_np / im_np.max()

    # Resize the mask to match the input resolution
    mask_np = mask.detach().cpu().numpy().astype(np.float32)
    if mask_np.ndim == 2:
        mask_np = np.expand_dims(mask_np, axis=-1)
    mask_image = Image.fromarray((mask_np.squeeze() * 255).astype(np.uint8))
    mask_resized = np.array(mask_image.resize(image.size, Image.BILINEAR)).astype(np.float32) / 255.0
    mask_resized = np.expand_dims(mask_resized, axis=-1)
    if mask_resized.shape[-1] != im_np.shape[-1]:
        mask_broadcast = np.repeat(mask_resized, im_np.shape[-1], axis=-1)
    else:
        mask_broadcast = mask_resized

    # Apply mask to the image
    im_np = mask_broadcast * im_np
    threshold = mask_resized.mean()
    im_np[mask_broadcast < threshold] = 1

    # Convert back to an image
    if im_np.max() > 0:
        im_final = (im_np / im_np.max() * 255).astype(np.uint8)
    else:
        im_final = np.zeros_like(im_np, dtype=np.uint8)
    masked_im = Image.fromarray(im_final)
    return masked_im


   

def get_mask(im: Image, device, model):
    # preprocess image
    orig_im = np.array(im)
    # orig_im_size = orig_im.shape[0:2]
    im_tensor = torch.tensor(orig_im, dtype=torch.float32).permute(2,0,1)
    im_tensor = F.interpolate(torch.unsqueeze(im_tensor,0), size=(1024,1024), mode='bilinear')
    image = torch.divide(im_tensor, 255.0)
    image_pre = normalize(image,[0.5,0.5,0.5],[1.0,1.0,1.0]).to(device)
    with torch.no_grad():
        result=model(image_pre)[0][0]
        result = result.squeeze().cpu()
    # postprocess image
    result = (result - torch.min(result)) / (torch.max(result) - torch.min(result))
    return result



def plot_row(axs, k, titles, row_num, lst):
    for i in range(k):
        axs[row_num, i].imshow(lst[i])
        axs[row_num, i].axis('off')
        if i == 0:
            axs[row_num, i].set_title(titles[row_num], loc='center', fontsize=14, pad=20)


def log_grid(input_images, target_sketches, output_sketches, epoch, output_dir, log_to_wandb, log_name="test"):
    # Titles for each row
    titles = ['Input Image', 'Target Sketch', 'Output Sketch']
    
    # Split images into two groups
    mid_point = len(input_images) // 2
    first_half_input = input_images[:mid_point]
    second_half_input = input_images[mid_point:]
    
    first_half_target = target_sketches[:mid_point]
    second_half_target = target_sketches[mid_point:]
    
    first_half_output = output_sketches[:mid_point]
    second_half_output = output_sketches[mid_point:]
    
    # Create two plots
    def create_plot(images, targets, outputs, group_num):
        k = len(images)
        
        # Set up the plot
        fig, axs = plt.subplots(3, k, figsize=(k * 3, 3 * 3))  # Adjust figsize as needed

        # If k is 1, axs will be 1-dimensional, handle this case separately
        if k == 1:
            axs = np.expand_dims(axs, axis=1)

        # Plot images in the first row
        plot_row(axs, k, titles, row_num=0, lst=images)
        plot_row(axs, k, titles, row_num=1, lst=targets)
        plot_row(axs, k, titles, row_num=2, lst=outputs)

        # Adjust layout
        plt.tight_layout()
        plot_name = f"{log_name}_group{group_num}"
        if epoch == -1:
            if log_to_wandb:
                wandb.log({plot_name: wandb.Image(plt)})
            plt.savefig(f"{output_dir}/{plot_name}")
        else:
            if log_to_wandb:
                wandb.log({plot_name: wandb.Image(plt)}, step=epoch)
            # plt.savefig(f"{output_dir}/iter_{epoch}_group{group_num}")
            plt.close()

    # Create and log the two plots
    create_plot(first_half_input, first_half_target, first_half_output, group_num=1)
    create_plot(second_half_input, second_half_target, second_half_output, group_num=2)




def log_grid_images_list(input_images, output_dir, log_to_wandb, log_name="test"):
    """
    Create a grid from a list of images, save it, and optionally log it to W&B.

    Args:
        input_images (list): List of image paths or PIL Images.
        output_dir (str): Directory to save the grid image.
        log_to_wandb (bool): Whether to log the grid to W&B.
        log_name (str): Name for the logged image in W&B.
    """


    # Define grid dimensions (1 row by len(images) columns)
    grid_cols = len(input_images)
    grid_rows = 1  # Single row of images

    # Resize images to a uniform size for the grid
    target_size = (256, 256)  # Adjust as needed
    resized_images = [img.resize(target_size) for img in input_images]

    # Create the grid canvas
    grid_width = target_size[0] * grid_cols
    grid_height = target_size[1] * grid_rows
    grid_image = Image.new("RGB", (grid_width, grid_height))

    # Paste images into the grid
    for idx, img in enumerate(resized_images):
        x_offset = idx * target_size[0]
        grid_image.paste(img, (x_offset, 0))

    # Save the grid image
    # grid_path = os.path.join(output_dir, f"{log_name}.png")
    # grid_image.save(grid_path)
    # print(f"Saved grid image to {grid_path}")

    # Optionally log to W&B
    if log_to_wandb:
        wandb.log({f"{log_name}": wandb.Image(grid_image)})
        print(f"Logged grid image to W&B with name {log_name}")



def log_grid_test(input_images, output_sketches, epoch, output_dir, log_to_wandb, log_name="test"):

    # Titles for each row
    titles = ['Input Image', 'Output Sketch']

    # Number of columns
    k = len(input_images)
    print("k" , k)

    # Set up the plot
    fig, axs = plt.subplots(2, k, figsize=(k * 2, 2 * 2))  # Adjust figsize as needed


    # If k is 1, axs will be 1-dimensional, handle this case separately
    if k == 1:
        axs = np.expand_dims(axs, axis=1)
    # Plot images in the first row
    plot_row(axs, k, titles, row_num=0, lst=input_images)
    plot_row(axs, k, titles, row_num=1, lst=output_sketches)


    # Adjust layout
    plt.tight_layout()
    if epoch==-1:
        if log_to_wandb:
            wandb.log({f"{log_name}": wandb.Image(plt)})
        # plt.savefig(f"{output_dir}/{log_name}")
    else:
        if log_to_wandb:
            wandb.log({f"{log_name}": wandb.Image(plt)}, step=epoch)
        # plt.savefig(f"{output_dir}/iter_{epoch}")
        plt.close()

def log_grid_model_sketches_all(diffusion_sketch_list, predict_sketch_list, target_sketch_list, step, log_name="test", mode="train"):
    # Titles for each row
    titles = ['Diffusion Sketch', 'Output Sketch', 'Target Sketch']

    # Use all data
    k = len(diffusion_sketch_list)

    # Set up the plot
    fig, axs = plt.subplots(3, k, figsize=(k * 3, 3 * 3))  # Adjust figsize as needed

    # If k is 1, axs will be 1-dimensional, handle this case separately
    if k == 1:
        axs = np.expand_dims(axs, axis=1)

    # Plot images in the rows
    plot_row(axs, k, titles, row_num=0, lst=diffusion_sketch_list)
    plot_row(axs, k, titles, row_num=1, lst=predict_sketch_list)
    plot_row(axs, k, titles, row_num=2, lst=target_sketch_list)

    # Adjust layout
    plt.tight_layout()
    if mode == "train":
        wandb.log({log_name: wandb.Image(plt)}, step=step)
    else:
        wandb.log({log_name: wandb.Image(plt)})

    plt.close()



def log_grid_model_sketches(diffusion_sketch_list, predict_sketch_list, target_sketch_list, step, log_name="test", mode="train", group_size=10):
    # Titles for each row
    titles = ['Diffusion Sketch', 'Output Sketch', 'Target Sketch']

    # Helper: chunk a list into pieces of size group_size
    def chunk_list(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    # Create a plot for each chunk
    def create_plot(images, targets, outputs, group_num):
        k = len(images)
        fig, axs = plt.subplots(3, k, figsize=(k * 3, 3 * 3))

        if k == 1:
            axs = np.expand_dims(axs, axis=1)

        plot_row(axs, k, titles, row_num=0, lst=images)
        plot_row(axs, k, titles, row_num=1, lst=targets)
        plot_row(axs, k, titles, row_num=2, lst=outputs)

        plt.tight_layout()
        plot_name = f"{log_name}_group{group_num}"
        if mode == "train":
            wandb.log({plot_name: wandb.Image(plt)}, step=step)
        else:
            wandb.log({plot_name: wandb.Image(plt)})
        plt.close()

    # Iterate over chunks of size group_size
    for group_num, (inputs, preds, targets) in enumerate(zip(
        chunk_list(diffusion_sketch_list, group_size),
        chunk_list(predict_sketch_list, group_size),
        chunk_list(target_sketch_list, group_size)
    ), start=1):
        create_plot(inputs, preds, targets, group_num)




def log_refine_model_prediction(diffusion_sketch, predict_sketch, target_sketch, step):
    # Create a figure with 1 row and 3 columns
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Display the images with titles
    axs[0].imshow(diffusion_sketch)
    axs[0].set_title("diffusion_sketch")
    axs[0].axis('off')  # Remove the axes

    axs[1].imshow(predict_sketch)
    axs[1].set_title("predicted_sketch")  # Replace 't' with the value of t
    axs[1].axis('off')

    axs[2].imshow(target_sketch)
    axs[2].set_title("target_sketch")
    axs[2].axis('off')

    # Tight layout to avoid overlapping
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Manually adjust the top spacing

    # Log the entire grid as one entity to WandB with the correct naming and step
    wandb.log({"model prediction": wandb.Image(fig)}, step=step)

    # Close the figure after logging to release memory
    plt.close(fig)

    



def log_model_prediction(x0_sketch, xt_sketch, predict_x0_sketch, t, quartile, step):
    # Create a figure with 1 row and 3 columns
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Display the images with titles
    axs[0].imshow(x0_sketch)
    axs[0].set_title("x0_sketch")
    axs[0].axis('off')  # Remove the axes

    axs[1].imshow(xt_sketch)
    axs[1].set_title(f"x{t}_sketch")  # Replace 't' with the value of t
    axs[1].axis('off')

    axs[2].imshow(predict_x0_sketch)
    axs[2].set_title("predict_x0_sketch")
    axs[2].axis('off')

    # Tight layout to avoid overlapping
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Manually adjust the top spacing

    # Log the entire grid as one entity to WandB with the correct naming and step
    wandb.log({f"model prediction quartile {quartile}": wandb.Image(fig)}, step=step)

    # Close the figure after logging to release memory
    plt.close(fig)


def log_diffusion_process_to_wandb(timesteps, xt_Denoising_Process, x0_Denoising_Process, title=""):
    # Ensure input lists have consistent lengths
    assert len(timesteps) == len(xt_Denoising_Process), "Timesteps and xt_Denoising_Process must have the same length."
    if x0_Denoising_Process:
        assert len(timesteps) == len(x0_Denoising_Process), \
            "If x0_Denoising_Process is provided, it must have the same length as timesteps."

    batch_size = len(xt_Denoising_Process[0])

    n_cols = len(timesteps)
    n_rows = 2 * batch_size if x0_Denoising_Process else batch_size
    fig, axes = plt.subplots(n_rows, n_cols + 1, figsize=(n_cols * 3, 3 * n_rows),
                             gridspec_kw={'width_ratios': [0.1] + [1] * n_cols})
    
    # Loop through each sample in the batch
    for batch_idx in range(batch_size):
        row_xt = 2 * batch_idx if x0_Denoising_Process else batch_idx
        if x0_Denoising_Process:
            row_x0 = 2 * batch_idx + 1
        
        # Add row titles on the left for xt
        axes[row_xt, 0].text(0.5, 0.5, "xt", va='center', ha='center', rotation=90,
                             fontsize=12, fontweight='bold', transform=axes[row_xt, 0].transAxes)
        axes[row_xt, 0].axis('off')  # Remove axis for the row title

        # Plot xt_Denoising_Process in the corresponding row (skip the first column for row titles)
        for i, timestep in enumerate(timesteps):
            axes[row_xt, i + 1].imshow(xt_Denoising_Process[i][batch_idx])
            axes[row_xt, i + 1].axis('off')
            axes[row_xt, i + 1].set_title(f't={timestep}', fontsize=10)
        
        # Add row titles for x0_Denoising_Process if provided
        if x0_Denoising_Process:
            axes[row_x0, 0].text(0.5, 0.5, "Predicted x0", va='center', ha='center', rotation=90,
                                 fontsize=12, fontweight='bold', transform=axes[row_x0, 0].transAxes)
            axes[row_x0, 0].axis('off')  # Remove axis for the row title

            # Plot x0_Denoising_Process in the corresponding row
            for i, timestep in enumerate(timesteps):
                axes[row_x0, i + 1].imshow(x0_Denoising_Process[i][batch_idx])
                axes[row_x0, i + 1].axis('off')

    # Adjust layout
    plt.tight_layout()
    
    # Log the grid to wandb
    wandb.log({title: wandb.Image(fig)})
    
    # Close the plot to free memory
    plt.close(fig)



def get_features_dim(image_features_type):
    if image_features_type=="CLIPMiddle_layer3":
        return (512,28)
    elif image_features_type=="CLIPMiddle_layer4":
        return (1024,14)
    elif image_features_type=="CLIPMiddle_layer5":
        return (2048,7)
    elif image_features_type=="DINO2":
        return (1536,16)
    

def parse_svg_size(val: str):
    if val is None:
        return None
    return float(re.sub(r"[a-zA-Z%]+", "", val))  # remove "px", "%", etc.


def extract_control_points_from_svg(svg_content):
    # Use StringIO to simulate a file object from the SVG string content
    paths, attributes, svg_attributes = svg2paths2(StringIO(svg_content))
    control_points = []
    for path in paths:
        path_control_points = []
        for segment in path:
            if isinstance(segment, CubicBezier):
                # Collect the 4 control points for this segment
                path_control_points.append([segment.start.real, segment.start.imag])  # Start point
                path_control_points.append([segment.control1.real, segment.control1.imag])  # Control point 1
                path_control_points.append([segment.control2.real, segment.control2.imag])  # Control point 2
                path_control_points.append([segment.end.real, segment.end.imag])  # End point
    
        control_points.append(path_control_points)

    canvas_size = parse_svg_size(svg_attributes.get("width"))
    points= torch.tensor(control_points) #(num_paths, 4, 2)
    return points, canvas_size


