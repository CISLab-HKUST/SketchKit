from custom_types import *
from models.models_utils import Model
from options import OptionsDiscriminator


class Discriminator3D(nn.Module):

    def forward(self, x: T)-> T:
        return self.model(x).mean()

    def __init__(self, discriminator_dim, num_layers):
        super(Discriminator3D, self).__init__()
        layers = []
        last_ch = 1
        for i in range(num_layers - 1):
            conv = nn.Conv3d(last_ch, discriminator_dim * 2 ** i, 3, stride=1, padding=1, bias=False)
            nn.init.xavier_uniform_(conv.weight)
            last_ch = discriminator_dim * 2 ** i
            layers.extend([conv,  nn.InstanceNorm3d(last_ch), nn.LeakyReLU(.2, True)])
        layers.append(nn.Conv3d(last_ch, 1, 3, stride=1, padding=1, bias=True))
        nn.init.xavier_uniform_(layers[-1].weight)
        nn.init.constant_(layers[-1].bias, 0)
        self.model = nn.Sequential(*layers)


class MultiScaleDiscriminator(Model):

    def forward(self, x: TS, detach=False) -> T:
        if detach:
            x = [x_.detach() for x_ in x]
        out = [discriminator(x_) for discriminator, x_ in zip(self.discriminators, x)]
        out = sum(out) / len(out)
        return out

    def __init__(self, opt: OptionsDiscriminator):
        super(MultiScaleDiscriminator, self).__init__()
        discriminators = [Discriminator3D(opt.discriminator_dim, opt.discriminator_num_layers) for _ in range(opt.num_discriminators)]
        self.discriminators = nn.ModuleList(discriminators)


if __name__ == '__main__':
    model = Discriminator3D(8, 5)
    x = torch.rand(3, 1, 64, 64, 64)
    out_ = model(x)
    print(out_)
