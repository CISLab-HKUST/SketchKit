
import torch.nn as nn

# A wrapper model for Classifier-free guidance **SAMPLING** only
# https://arxiv.org/abs/2207.12598
class ClassifierFreeSampleModel(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model  # model is the actual model to run

        assert self.model.cond_mask_prob > 0, 'Cannot run a guided diffusion on a model that has not been trained with no conditions'

        # pointers to inner model
        self.ncpoints = self.model.ncpoints
        self.nfeats = self.model.nfeats
        self.cond_mode = self.model.cond_mode   
      

    def forward(self, x, timesteps, image_features, scale):
        cond_mode = self.model.cond_mode
        assert cond_mode in ['image']
        out = self.model(x, timesteps, image_features)
        out_uncond = self.model(x, timesteps, image_features, uncond=True)
        return out_uncond + (scale.view(-1, 1, 1, 1) * (out - out_uncond))

