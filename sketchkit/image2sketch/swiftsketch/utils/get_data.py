

import torch
import os
import numpy as np
from torch.utils.data import Dataset
import utils.sketch_utils as utils



def normalize_control_points(control_points, canvas_size, scaling_factor):
    norm = control_points / canvas_size
    norm = ((norm * 2) - 1) * scaling_factor
    return norm


class ImageSVGDataset(Dataset):
    def __init__(self, normalized_svg_control_points_lst,real_rendered_images_lst, image_features_lst):
        self.image_features_lst=image_features_lst
        self.normalized_svg_control_points_lst= normalized_svg_control_points_lst
        self.real_rendered_images_lst= real_rendered_images_lst

    def __len__(self):
        return len(self.image_features_lst)
    

    def __getitem__(self, idx):
        target_control_points= self.normalized_svg_control_points_lst[idx]
        target_rendered_images= self.real_rendered_images_lst[idx]
        image_features= self.image_features_lst[idx]

        return target_control_points, target_rendered_images, image_features
    


def create_data_set(dir_name_lst, target_key_name, image_features_type, canvas_width,canvas_height,device, scaling_factor, cat_data_size, sort_by, use_cache=True, cache_path_dir="", data_name="data"):
    
    image_features_lst=[]
    normalized_svg_control_points_lst=[]
    real_rendered_images_lst=[]
   
    cache_path = os.path.join(cache_path_dir, f'{data_name}.pt')
    if use_cache and os.path.exists(cache_path):
        print("load data from cache", flush=True)
        print("cache path", cache_path, flush=True)
        _cache = torch.load(cache_path)
        normalized_svg_control_points_lst, real_rendered_images_lst, image_features_lst = _cache['normalized_svg_control_points_lst'], _cache['real_rendered_images_lst'], _cache['image_features_lst']
    
    else:

        print("create data", flush=True)
        for dir_name in dir_name_lst:
            file_count = 0  # Counter for files processed in the current directory
            print(f"Processing directory: {dir_name}", flush=True)

            for f_ in os.listdir(dir_name):
                if file_count >= cat_data_size:  # Stop if we reach the limit
                    print(f"stopped at {f_}", flush=True)
                    break

                file_path = os.path.join(dir_name, f_)
                features_key= f"{image_features_type}_features"
                read_dictionary = utils.load_entry(file_path, [target_key_name], features_key)

                if target_key_name in read_dictionary.keys() and features_key in read_dictionary.keys():
                    target_svg= read_dictionary[target_key_name]

                    image_features_lst.append(read_dictionary[features_key].to("cpu"))
                    svg_control_points, target_canvas_size = utils.extract_control_points_from_svg(target_svg)
                    svg_control_points= svg_control_points.to(device)
                
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
                    normalized_svg_control_points_lst.append(normalized_svg_control_points.to("cpu"))
                    
                    real_rendered_images, _  = utils.rander_image_from_points(svg_control_points.unsqueeze(0),canvas_width, canvas_height)
                    real_rendered_images_lst.append(real_rendered_images[0].to("cpu"))

                    file_count += 1  # Increment counter for the current directory


            print(f"Number of files taken from {dir_name}: {file_count}", flush=True)

        # Saving to cache
        print("saving data to cache", flush=True)
        torch.save({
            'normalized_svg_control_points_lst': normalized_svg_control_points_lst,
            'real_rendered_images_lst': real_rendered_images_lst,
            'image_features_lst': image_features_lst
        }, cache_path)

    data_size=  len(normalized_svg_control_points_lst)   
    print("data size:",data_size, flush=True)
   
    dataset = ImageSVGDataset(normalized_svg_control_points_lst,real_rendered_images_lst, image_features_lst)
    return dataset 


