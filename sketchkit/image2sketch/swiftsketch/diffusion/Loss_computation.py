


import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import VGG16_Weights

       
class Loss(nn.Module):

    def __init__(self, args):
       
        super(Loss, self).__init__()
        self.lpips_weight = args.lpips_weight
        self.l1_points_weight = args.l1_points_weight
        losses_to_apply, loss_mapper = self.get_losses_to_apply(args)
        self.losses_to_apply= losses_to_apply
        self.loss_mapper= loss_mapper


    def get_losses_to_apply(self, args):
        loss_mapper={}
        losses_to_apply = []
        if self.lpips_weight:
            losses_to_apply.append("LPIPS")
            loss_mapper["LPIPS"]= LPIPS(args)
        if self.l1_points_weight:
            losses_to_apply.append("L1_points")
            loss_mapper["L1_points"]= L1_points()
       
        return losses_to_apply, loss_mapper
    
    def forward(self, sketches, targets , output_control_points,target_control_points, mode="train"):
        losses_dict = dict.fromkeys(
            self.losses_to_apply, torch.tensor([0.0]).to(sketches.device))
        loss_coeffs = dict.fromkeys(self.losses_to_apply, 1.0)
        loss_coeffs["LPIPS"] = self.lpips_weight
        loss_coeffs["L1_points"] = self.l1_points_weight
        
        for loss_name in self.losses_to_apply:
            if loss_name == "L1_points":
                losses_dict[loss_name] = self.loss_mapper[loss_name](
                    output_control_points, target_control_points).mean()
            else:   #lpips 
                losses_dict[loss_name] = self.loss_mapper[loss_name](
                    sketches, targets, mode).mean()
        for key in self.losses_to_apply:
            losses_dict[key] = losses_dict[key] * loss_coeffs[key]
        return losses_dict
    



class LPIPS(torch.nn.Module):
    def __init__(self, args, pretrained=True, normalize=True, pre_relu=True):
        """
        Args:
            pre_relu(bool): if True, selects features **before** reLU activations
        """
        super(LPIPS, self).__init__()
        # VGG using perceptually-learned weights (LPIPS metric)
        device= args.device
        self.normalize = normalize
        self.pretrained = pretrained
        augemntations = []
        augemntations.append(transforms.RandomPerspective(
            fill=1, p=1.0, distortion_scale=0.5))
        augemntations.append(transforms.RandomResizedCrop(
            224, scale=(0.8, 0.8), ratio=(1.0, 1.0)))
        self.augment_trans = transforms.Compose(augemntations)
        self.feature_extractor = LPIPS._FeatureExtractor(
            pretrained, pre_relu).to(device)

    def _l2_normalize_features(self, x, eps=1e-10):
        nrm = torch.sqrt(torch.sum(x * x, dim=1, keepdim=True))
        return x / (nrm + eps)

    def forward(self, pred, target, mode="train"):
        """Compare VGG features of two inputs."""

        batch_size=pred.size(0)

        augmented_sketches, augmented_images = [pred], [target]
        if mode == "train":  
            for _ in range(4):
                augmented = self.augment_trans(torch.cat([pred, target], dim=0))
                augmented_sketches.append(augmented[:batch_size])
                augmented_images.append(augmented[batch_size:])
     
        xs = torch.cat(augmented_sketches, dim=0)
        ys = torch.cat(augmented_images, dim=0)

        # Shape of xs and ys: [5*batch_size, C, 224, 224]

        pred = self.feature_extractor(xs)
        target = self.feature_extractor(ys)

        # Shape of pred and target: [5*batch_size, C_i, H_i, W_i] (for each feature map)
        # The feature extractor will output a list of feature maps with different channels and spatial sizes depending on the layers used.

        # L2 normalize features
        if self.normalize:
            pred = [self._l2_normalize_features(f) for f in pred]
            target = [self._l2_normalize_features(f) for f in target]


        if self.normalize:
            diffs = [torch.sum((p - t) ** 2, 1)
                    for (p, t) in zip(pred, target)]  
        else:
            # mean instead of sum to avoid super high range
            diffs = [torch.mean((p - t) ** 2, 1)
                    for (p, t) in zip(pred, target)]
            
        # Shape of each diff in diffs: [5*batch_size, H_i, W_i]
        # The difference is computed for each feature map across the channel dimension (dimension 1), so the resulting shape drops the channel dimension, retaining the batch size and spatial dimensions.


        # Spatial average
        diffs = [diff.mean([1, 2]) for diff in diffs]

        # Shape of each diff after mean: [5*batch_size]
        # The spatial dimensions are averaged, leaving only the batch dimension

        sum_diffs=sum(diffs) 

        #Shape of sum_diffs: [15]
        #The sum across all feature maps for each image in the batch is computed.

        return sum_diffs.mean() #mean over all images and augemntations in the batch


    class _FeatureExtractor(torch.nn.Module):
        def __init__(self, pretrained, pre_relu):
            super(LPIPS._FeatureExtractor, self).__init__()
            weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
            vgg_pretrained = models.vgg16(weights=weights).features

            self.breakpoints = [0, 4, 9, 16, 23, 30]
            if pre_relu:
                for i, _ in enumerate(self.breakpoints[1:]):
                    self.breakpoints[i + 1] -= 1

            # Split at the maxpools
            for i, b in enumerate(self.breakpoints[:-1]):
                ops = torch.nn.Sequential()
                for idx in range(b, self.breakpoints[i + 1]):
                    op = vgg_pretrained[idx]
                    ops.add_module(str(idx), op)
                # print(ops)
                self.add_module("group{}".format(i), ops)

            # No gradients
            for p in self.parameters():
                p.requires_grad = False

            # Torchvision's normalization: <https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101>
            self.register_buffer("shift", torch.Tensor(
                [0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("scale", torch.Tensor(
                [0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        def forward(self, x):
            feats = []
            x = (x - self.shift.to(x.device)) / self.scale.to(x.device)
            for idx in range(len(self.breakpoints) - 1):
                m = getattr(self, "group{}".format(idx))
                x = m(x)
                feats.append(x)
            return feats



class L1_points(torch.nn.Module):
    def __init__(self):
        super(L1_points, self).__init__()
        self.l1_loss = torch.nn.L1Loss() 

    def forward(self, control_points, svg_control_points):
        # control_points and svg_control_points are [bs, nstrokes, 8] (control points)
 
        l1_loss_points= self.l1_loss(control_points, svg_control_points)
        return l1_loss_points



                
