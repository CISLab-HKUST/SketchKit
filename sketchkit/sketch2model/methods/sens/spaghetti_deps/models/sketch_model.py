import math
import options
from custom_types import *
from models import transformer, models_utils


class VisionTransformer(nn.Module):

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = x + self.positional_embedding
        x = self.ln_pre(x)
        x = self.transformer(x)
        return x

    def __init__(self, input_resolution: int, patch_size: int, hidden_dim: int, layers: int, heads: int,
                 input_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=(patch_size, patch_size),
                               stride=(patch_size, patch_size), bias=False)

        scale = hidden_dim ** -0.5
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2, hidden_dim))
        self.ln_pre = nn.LayerNorm(hidden_dim)
        self.transformer = transformer.Transformer(hidden_dim, heads, layers)


class SketchRefinement(nn.Module):

    def forward(self, x: T):
        query = self.query_embeddings.repeat(x.shape[0], 1, 1)
        x = x + query
        out = self.encoder(x)
        return out

    def __init__(self, opt: options.SketchOptions):
        super(SketchRefinement, self).__init__()
        dim_ref = int(opt.dim_h * 1.5)
        self.encoder = transformer.Transformer(dim_ref, 8, 12)
        query_embeddings = torch.zeros(1, opt.num_gaussians, dim_ref)
        self.query_embeddings = nn.Parameter(query_embeddings)
        torch.nn.init.normal_(
            self.query_embeddings.data,
            0.0,
            1. / math.sqrt(dim_ref),
        )


class Sketch2Spaghetti(models_utils.Model):

    def get_visual_embedding_and_queries(self, x: T):
        x = self.vit(x)
        query = self.query_embeddings.repeat(x.shape[0], 1, 1)
        return x, query

    def forward_attention(self, x: T):
        x, query = self.get_visual_embedding_and_queries(x)
        out, attention = self.decoder.forward_attention(query, x)
        cls = self.mlp_cls(out)
        zh = self.mlp_zh(out)
        return zh, cls, attention

    def forward_mid(self, x, return_mid):
        cls = self.mlp_cls(x)
        zh = self.mlp_zh(x)
        if return_mid:
            return zh, cls, x
        return zh, cls

    def forward(self, x: T, return_mid=False):
        x, query = self.get_visual_embedding_and_queries(x)
        out = self.decoder(query, x)
        return self.forward_mid(out, return_mid)

    def refine(self, mid_embedding, return_mid=False):
        out = self.refinement_encoder(mid_embedding)
        zh = self.mlp_zh(out)
        if return_mid:
            return zh, out
        return zh

    def __init__(self, opt: options.SketchOptions):
        super(Sketch2Spaghetti, self).__init__()
        self.opt = opt
        self.vit = VisionTransformer(256, opt.vit_patch_size, opt.dim_h, 12, 8, 1)
        dim_ref = int(opt.dim_h * 1.5)
        self.decoder = transformer.CombTransformer(dim_ref, 8, 12, opt.dim_h)
        query_embeddings = torch.zeros(1, opt.num_gaussians, dim_ref)
        self.query_embeddings = nn.Parameter(query_embeddings)
        torch.nn.init.normal_(
            self.query_embeddings.data,
            0.0,
            1. / math.sqrt(dim_ref),
        )
        self.mlp_zh = models_utils.MLP((dim_ref, opt.dim_h, opt.dim_h), norm_class=None)
        self.mlp_cls = models_utils.MLP((dim_ref, opt.dim_h, 1), norm_class=None)
        if opt.refinement:
            self.refinement_encoder = SketchRefinement(opt)


def main():
    model = Sketch2Spaghetti(options.SketchOptions())
    x = torch.rand(5, 1, 256, 256)
    out = model(x)
    print(out.shape)


if __name__ == '__main__':
    main()
