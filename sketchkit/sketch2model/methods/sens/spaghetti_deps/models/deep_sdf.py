from custom_types import *
from models.models_utils import Model
from options import Options
import constants
from utils import  files_utils

def expand_z(samples, zs):
    b, n, d = samples.shape
    if zs.dim() != samples.dim():
        zs = zs.unsqueeze(1).expand(b, n, -1)
    vec = torch.cat((samples, zs), dim=2)
    return vec


class DeepSDF(Model):
    def __init__(
        self, opt: Options,
        dims=(512, 512, 512, 512, 512, 512, 512, 512),
        dropout_layers=(0, 1, 2, 3, 4, 5, 6, 7),
        dropout_prob=0.2,
        norm_layers=(0, 1, 2, 3, 4, 5, 6, 7),
        latent_in=(4,),
        weight_norm=True,
        latent_dropout=False,
    ):
        super(DeepSDF, self).__init__()
        dim_in = 2 * (opt.pos_dim + constants.DIM)
        dims = [dim_in] + list(dims) + [3 if opt.loss_func is LossType.CROSS else 1]
        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)
        self.weight_norm = weight_norm

        for i in latent_in:
            dims[i] += dims[0]
        layers = []
        for i in range(0, self.num_layers - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if weight_norm and i in self.norm_layers:
                layers[-1] = nn.utils.weight_norm(layers[-1])
        self.layers = nn.ModuleList(layers)
        self.relu = nn.ReLU(True)
        self.dropout_layers = dropout_layers
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, vec):
        x = vec
        for i, layer in enumerate(self.layers):
            if layer in self.latent_in:
                x = torch.cat([x, vec], 2)
            x = layer(x)
            if i < self.num_layers - 2:
                x = self.relu(x)
                if i in self.dropout_layers:
                    x = self.dropout(x)
            # files_utils.save_pickle(x.detach().cpu(), f"/home/amirh/projects/spaghetti_private/assets/debug/out_{i}_gt")
        return x

