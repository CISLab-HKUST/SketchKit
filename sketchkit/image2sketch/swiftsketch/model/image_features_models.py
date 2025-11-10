
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import clip
from transformers import  AutoImageProcessor, AutoModel


class DINO2Feutures(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.dino_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-giant')
        self.dino_model = AutoModel.from_pretrained('facebook/dinov2-giant').to(device)
        self.device=device
        self.transform = transforms.Compose([
        transforms.ToTensor(),
    ])
        
    def image_preprocessing(self,image):
        image=image.convert("RGB")
        image = self.transform(image) 
        image = image.to(self.device)
        image=image.unsqueeze(0)
        return image
        

    def get_dino_features(self, image):
        inputs = self.dino_processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.dino_model(**inputs)
        # outputs = outputs.last_hidden_state[:, 1:, :].mean(dim=1) #take mean of everything but cls
        # outputs = outputs.last_hidden_state[:,0,:] # take only cls
        outputs = outputs.last_hidden_state.mean(dim=1)   
        outputs /= outputs.norm(dim=-1, keepdim=True) 
        return outputs

    def forward(self, image):
        image= self.image_preprocessing(image)
        with torch.no_grad():
            features = self.get_dino_features(image)
            return features[0]
        

 
        
class CLIPMidlleFeutures(nn.Module):
    def __init__(self, device, layer):
        super().__init__()
        self.device = device
        self.clip_model, clip_preprocess = clip.load("RN101", device=device, jit=False)
        # Freeze the CLIP model parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.visual_model = self.clip_model.visual
        layers = list(self.visual_model.children())
        self.init_layers = torch.nn.Sequential(*layers[:10])
        self.layer1 = layers[10]
        self.layer2 = layers[11]
        self.layer3 = layers[12]
        self.layer4 = layers[13]
        self.att_pool2d = layers[14]

       
        self.middle_layer_index = layer 
        # clip normalization
        self.normalize_transform = clip_preprocess

    def forward_inspection_clip_resnet(self, x):
        x = x.type(self.visual_model.conv1.weight.dtype)
        x = self.init_layers(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        y = self.att_pool2d(x4)
        return y, [x, x1, x2, x3, x4]

    def get_clip_features_from_middle_layer(self, images):
        if isinstance(images, torch.Tensor):
            processed = images.to(self.device)
        else:
            processed = torch.stack([
                self.normalize_transform(img.convert("RGB")) for img in images
            ]).to(self.device)

        fc_features, features_list = self.forward_inspection_clip_resnet(processed)

        layer_map = {3: 2, 4: 3, 5: 4}
        target_index = layer_map.get(self.middle_layer_index, self.middle_layer_index)
        if target_index < len(features_list):
            return features_list[target_index]
        return fc_features

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            batch = x
            if batch.ndim == 3:
                batch = batch.unsqueeze(0)
        else:
            batch = torch.stack([self.normalize_transform(x.convert("RGB"))])

        with torch.no_grad():
            features = self.get_clip_features_from_middle_layer(batch)
        features = features.to(torch.float32)
        return features[0]
        

       