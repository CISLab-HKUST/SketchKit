

import torch
import os
import numpy as np
from torch.utils.data import Dataset
import utils.sketch_utils as utils


def normalize_control_points(control_points, canvas_size, scaling_factor):
    norm = control_points / canvas_size
    norm = ((norm * 2) - 1) * scaling_factor
    return norm


class RefineImageSVGDataset(Dataset):
    def __init__(self,svg_control_points_lst, real_rendered_images_lst, diffusion_svg_control_points_lst,image_features_lst):
       self.svg_control_points_lst= svg_control_points_lst
       self.real_rendered_images_lst=real_rendered_images_lst
       self.diffusion_svg_control_points_lst=diffusion_svg_control_points_lst
       self.image_features_lst=image_features_lst
         

    def __len__(self):
        return len(self.svg_control_points_lst)
    

    def __getitem__(self, idx):
        target_control_points= self.svg_control_points_lst[idx]
        diffusion_control_points= self.diffusion_svg_control_points_lst[idx]
        target_rendered_images= self.real_rendered_images_lst[idx]
        image_features= self.image_features_lst[idx]

        return target_control_points, target_rendered_images, diffusion_control_points,image_features
    

def create_data_set_refine_model(dir_name_lst, target_key_name, diffusion_key_name, image_features_type, canvas_width,canvas_height,device, scaling_factor, cat_data_size, sort_by, use_cache=True, cache_path_dir="", data_name="data_refine"):
    svg_control_points_lst=[]
    real_rendered_images_lst=[]
    diffusion_svg_control_points_lst=[]
    image_features_lst=[]

    cache_path = os.path.join(cache_path_dir, f'{data_name}.pt')
    if use_cache and os.path.exists(cache_path):
        print("load data from cache", flush=True)
        print("cache path", cache_path, flush=True)
        _cache = torch.load(cache_path)
        svg_control_points_lst, real_rendered_images_lst, diffusion_svg_control_points_lst, image_features_lst =  _cache['svg_control_points_lst'], _cache['real_rendered_images_lst'], _cache['diffusion_svg_control_points_lst'], _cache['image_features_lst']
    
    else:

        for dir_name in dir_name_lst:
            category = os.path.basename(os.path.dirname(dir_name)).split('_')[-1]
            print(category,  flush=True)

            file_count = 0  # Counter for files processed in the current directory
            print(f"Processing directory: {dir_name}", flush=True)

            # Process all other files in the directory
            for f_ in os.listdir(dir_name):

                if file_count >= cat_data_size:  # Stop if we reach the limit
                    print(f"stopped at {f_}", flush=True)
                    break

                file_path = os.path.join(dir_name, f_)
                features_key= f"{image_features_type}_features"

                read_dictionary = utils.load_entry(file_path, [target_key_name, diffusion_key_name], features_key)
                if target_key_name in read_dictionary.keys() and features_key in read_dictionary.keys () and diffusion_key_name in read_dictionary.keys():
                    target_svg, diffusion_svg  = (
                        read_dictionary[target_key_name],
                        read_dictionary[diffusion_key_name],
                    )
                    
                    image_features_lst.append(read_dictionary[features_key].to("cpu"))
                    svg_control_points, target_canvas_size = utils.extract_control_points_from_svg(target_svg)
                    svg_control_points = svg_control_points.to(device)

                    if sort_by!= "no_sorting":
                        mask=None
                        attn_map=None
                        if sort_by== "contour_and_attn":
                            attn_map= read_dictionary["attn_map"]
                            mask= read_dictionary["mask"]
                        if sort_by== "contour":
                            mask= read_dictionary["mask"]
                        svg_control_points = utils.sort_strokes(svg_control_points, sort_by, target_svg, mask, attn_map)

                    svg_control_points = svg_control_points * (canvas_width/ target_canvas_size) # if the target canvas size is different from the training canvas size

                    normalized_svg_control_points=  normalize_control_points(svg_control_points, canvas_width, scaling_factor)
                    svg_control_points_lst.append(normalized_svg_control_points.to("cpu"))
                    real_rendered_images, _  = utils.rander_image_from_points(svg_control_points.unsqueeze(0),canvas_width, canvas_height)
                    real_rendered_images_lst.append(real_rendered_images[0].to("cpu"))

                    diffusion_svg_control_points, diffusion_canvas_size = utils.extract_control_points_from_svg(diffusion_svg)
                    diffusion_svg_control_points= diffusion_svg_control_points.to(device)
                    diffusion_svg_control_points = diffusion_svg_control_points * (canvas_width/ diffusion_canvas_size) # if the diffusion canvas size is different from the training canvas size

                    normalized_diffusion_svg_control_points= normalize_control_points(diffusion_svg_control_points, canvas_width, scaling_factor)
                    diffusion_svg_control_points_lst.append(normalized_diffusion_svg_control_points.to("cpu"))

                    file_count += 1  # Increment counter for the current directory

            print(f"Number of files taken from {dir_name}: {file_count}", flush=True)

        # Saving to cache
        print("saving data to cache", flush=True)
        torch.save({
            'svg_control_points_lst': svg_control_points_lst,
            'diffusion_svg_control_points_lst': diffusion_svg_control_points_lst,
            'real_rendered_images_lst': real_rendered_images_lst,
            'image_features_lst': image_features_lst
        }, cache_path)
      
    data_size=  len(image_features_lst)   
    print("data size:",data_size, flush=True)
    dataset = RefineImageSVGDataset(svg_control_points_lst, real_rendered_images_lst, diffusion_svg_control_points_lst,image_features_lst)
    print("created dataset", flush=True)
    return dataset 




class RefineImageSVGDatasetTest(Dataset):
    def __init__(self,image_features_lst, target_svg_lst,diffusion_svg_lst,mask_list,attn_list, canvas_width,canvas_height, device, scaling_factor, sort_by):
    
        self.image_features_lst=image_features_lst
        self.normalized_svg_control_points_lst=[]
        self.normalized_diffusion_svg_control_points_lst=[]
        self.real_rendered_images_lst=[]
        self.diffusion_rendered_images_lst=[]

        for i in range(len(target_svg_lst)):
            target_svg= target_svg_lst[i]
            mask= mask_list[i]
            svg_control_points, target_canvas_size = utils.extract_control_points_from_svg(target_svg)
            svg_control_points= svg_control_points.to(device)
            
            if sort_by!= "no_sorting":
                attn_map= None
                if sort_by=="contour_and_attn":
                    attn_map= attn_list[i]
                svg_control_points = utils.sort_strokes(svg_control_points, sort_by, target_svg, mask, attn_map)
            
            svg_control_points = svg_control_points * (canvas_width/ target_canvas_size) # if the target canvas size is different from the training canvas size

            normalized_svg_control_points=  normalize_control_points(svg_control_points, canvas_width, scaling_factor) 
            self.normalized_svg_control_points_lst.append(normalized_svg_control_points)
            real_rendered_images, _  = utils.rander_image_from_points(svg_control_points.unsqueeze(0),canvas_width, canvas_height)
            self.real_rendered_images_lst.append(real_rendered_images[0])


        for diffusion_svg in diffusion_svg_lst:
            diffusion_svg_control_points, diffusion_canvas_size = utils.extract_control_points_from_svg(diffusion_svg)
            diffusion_svg_control_points= diffusion_svg_control_points.to(device)
            diffusion_svg_control_points = diffusion_svg_control_points * (canvas_width/ diffusion_canvas_size) # if the diffusion canvas size is different from the training canvas size
            normalized_diffusion_svg_control_points= normalize_control_points(diffusion_svg_control_points, canvas_width, scaling_factor)
            self.normalized_diffusion_svg_control_points_lst.append(normalized_diffusion_svg_control_points)
            diffusion_rendered_images, _  = utils.rander_image_from_points(diffusion_svg_control_points.unsqueeze(0),canvas_width, canvas_height)
            self.diffusion_rendered_images_lst.append(diffusion_rendered_images[0])
         
      
    def __len__(self):
        return len(self.image_features_lst)
    

    def __getitem__(self, idx):
        target_control_points= self.normalized_svg_control_points_lst[idx]
        diffusion_control_points= self.normalized_diffusion_svg_control_points_lst[idx]
        target_rendered_images= self.real_rendered_images_lst[idx]
        diffusion_rendered_images= self.diffusion_rendered_images_lst[idx]
        image_features= self.image_features_lst[idx]

        return target_control_points, target_rendered_images, diffusion_control_points,diffusion_rendered_images,image_features
    



def create_data_set_refine_model_test(dir_name, target_key_name, diffusion_key_name, image_features_type, canvas_width,canvas_height,device, scaling_factor, sort_by):
    input_image_lst = []
    target_svg_lst = []
    mask_list=[]
    attn_list=[]
    image_features_lst=[]
    diffusion_svg_lst=[]
    resized_input_image_lst=[]

    for f_ in os.listdir(dir_name):
        file_path = os.path.join(dir_name, f_)
        features_key= f"{image_features_type}_features"
        read_dictionary = utils.load_entry(file_path, [target_key_name, diffusion_key_name], features_key)
        if target_key_name in read_dictionary.keys() and features_key in read_dictionary.keys() and diffusion_key_name in read_dictionary.keys() :
            input_image, target_svg, mask, diffusion_svg = (
                        read_dictionary["image"],
                        read_dictionary[target_key_name],
                        read_dictionary["mask"],
                        read_dictionary[diffusion_key_name],     
                    )
            if sort_by=="contour_and_attn":
                attn_map= read_dictionary["attn_map"]
                attn_list.append(attn_map)

            
            image_features_lst.append(read_dictionary[features_key].to("cpu"))
            input_image= utils.create_masked_image(input_image, mask)
            input_image_lst.append(input_image)
            resized_input_image = input_image.resize(( canvas_width, canvas_height)) #resize to 224 224
            resized_input_image_lst.append(resized_input_image)
            target_svg_lst.append(target_svg)
            mask_list.append(mask)
            diffusion_svg_lst.append(diffusion_svg)

        else:
            print("problematic_file")
            print(f_)
            print("keys")
            print(read_dictionary.keys())

    data_size=  len(input_image_lst)   
    print("data size:",data_size, flush=True)
   
    dataset = RefineImageSVGDatasetTest(image_features_lst, target_svg_lst,diffusion_svg_lst,mask_list,attn_list, canvas_width,canvas_height, device, scaling_factor, sort_by)
    print("created dataset", flush=True)
    return dataset , resized_input_image_lst



