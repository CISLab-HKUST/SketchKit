
import torch
import os
import model.image_features_models as models
import utils.sketch_utils as utils
import numpy as np
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features using a specified network.")
    parser.add_argument("--dir_name", required=True, help="Directory containing input .npy or .npz files")
    parser.add_argument("--network_name", default="CLIPMiddle_layer4", 
                        choices=['DINO2', 'CLIPMiddle_layer3', 'CLIPMiddle_layer4', 'CLIPMiddle_layer5', 'CLIP_fc'], 
                        help="Name of the network to use")
    args = parser.parse_args()

    dir_name = args.dir_name
    network_name = args.network_name
    features_key= f"{network_name}_features"

    
    if torch.cuda.is_available():
        print("Using GPU")
    else:
        print("Using cpu")
    device = torch.device("cuda" if (
    torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")

    if network_name== "DINO2":
        features_model = models.DINO2Feutures(device).to(device) 
    elif  network_name== "CLIPMiddle_layer3":
        features_model = models.CLIPMidlleFeutures(device, 2).to(device) 
    elif  network_name== "CLIPMiddle_layer4":
        features_model = models.CLIPMidlleFeutures(device, 3).to(device) 
    elif  network_name== "CLIPMiddle_layer5":
        features_model = models.CLIPMidlleFeutures(device, 4).to(device) 
    elif  network_name== "CLIP_fc":
        features_model = models.CLIPMidlleFeutures(device, 5).to(device)       
    else:
        raise ValueError('Please choose correct fearures model [DINO2, CLIPMiddle_layer3, CLIPMiddle_layer4, CLIPMiddle_layer5, CLIP_fc]')


    count=1
    for f_ in os.listdir(dir_name):
        file_path = os.path.join(dir_name, f_)
        dictionary= utils.load_entry(file_path, features_key=features_key)
        if features_key not in dictionary.keys():
            input_image, mask = dictionary["image"], dictionary["mask"]
            input_image= utils.create_masked_image(input_image, mask)
            image_features= features_model(input_image).to("cpu")


            # Save the updated dictionary back to the file
            if os.path.splitext(os.path.basename(file_path))[-1] == ".npy":
                data = np.load(file_path, allow_pickle=True).item()
                data[features_key] = image_features
                np.save(file_path, data)
            else: #.npz
                data = dict(np.load(file_path, allow_pickle=True))
                data[features_key] = image_features.numpy()
                np.savez_compressed(file_path, **data)

            
            print(f"finish save features file {f_} number {count}")
            count+=1
 

    print("finish save features all files")

    
