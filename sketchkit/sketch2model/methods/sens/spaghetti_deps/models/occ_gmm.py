from options import Options, OptionsSingle
from models import models_utils, transformer, mlp_models, deep_sdf, gm_utils
import constants
from custom_types import *
from torch import distributions
import math
from utils import rotation_utils, files_utils


def dot(x, y, dim=3):
    return torch.sum(x * y, dim=dim)


def remove_projection(v_1, v_2):
    proj = (dot(v_1, v_2) / dot(v_2, v_2))
    return v_1 - proj[:, :, :, None] * v_2


def get_p_direct(splitted: TS) -> T:
    raw_base = []
    for i in range(constants.DIM):
        u = splitted[i]
        for j in range(i):
            u = remove_projection(u, raw_base[j])
        raw_base.append(u)
    p = torch.stack(raw_base, dim=3)
    p = p / torch.norm(p, p=2, dim=4)[:, :, :, :, None]  # + self.noise[None, None, :, :]
    return p


def split_gm(splitted: TS, as_tait_bryan: bool) -> TS:
    if as_tait_bryan:
        p = rotation_utils.get_p_from_tait_bryan(splitted)
    else:
        p = get_p_direct(splitted)
    # eigenvalues
    eigen = splitted[-3] ** 2 + constants.EPSILON
    mu = splitted[-2]
    phi = splitted[-1].squeeze(3)
    return mu, p, phi, eigen


class GmmTransformer(nn.Module):

    def forward_bottom(self, x):
        return self.l1(x).view(-1, self.bottom_width, self.embed_dim)

    def forward_upper(self, x):
        return self.transformer(x)

    def forward(self, x):
        x = self.forward_bottom(x)
        x = self.forward_upper(x)
        out = [x]
        for liner, transformer_ in zip(self.l2, self.split_transformer):
            b, g, h = x.shape
            x = liner(x)
            x = x.view(b * g, -1, h)
            x = transformer_(x)
            out.append(x)
        return out, None

    def __init__(self, opt: Options, act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm):
        super(GmmTransformer, self).__init__()
        self.bottom_width = opt.num_gaussians
        self.embed_dim = opt.dim_h
        self.l1 = nn.Linear(opt.dim_z, self.bottom_width * opt.dim_h)
        self.l2 = nn.ModuleList()
        self.split_transformer = nn.ModuleList()
        if type(opt) is OptionsSingle:
            self.transformer = lambda x: x
        else:
            if opt.decomposition_network == 'mlp':
                self.transformer = transformer.Mlp(opt.dim_h, opt.dim_h * 2, act=nnf.relu)
            else:
                self.transformer = transformer.Transformer(opt.dim_h, opt.num_heads, opt.num_layers, act=act,
                                                           norm_layer=norm_layer)
        if len(opt.hierarchical):
            for split in opt.hierarchical:
                self.l2.append(nn.Linear(opt.dim_h, split * opt.dim_h))
                self.split_transformer.append(transformer.Transformer(opt.dim_h, opt.num_heads, opt.num_layers, act=act,
                                                       norm_layer=norm_layer))


class SdfHead(nn.Module):

    def get_pos(self, coords: T):
        pos = self.pos_encoder(coords)
        if self.head_type == "sin":
            x = coords
        else:
            x = pos = torch.cat((coords, pos), dim=2)
        return x, pos

    def forward_attention(self, coords: T, zh: T, mask: Optional[T] = None, alpha: TN = None) -> TS:
        x, pos = self.get_pos(coords)
        _, attn = self.sdf_transformer.forward_with_attention(pos, zh, mask, alpha)
        return attn

    def get_mask_by_gmm(self, coords: T, gmm: List[TS], mask: TN) -> TN:
        if gmm is not None and 0 < self.mask_by_gmm < gmm[0][0].shape[2]:
            with torch.no_grad():
                supports: T = gm_utils.hierarchical_gm_log_likelihood_loss(gmm, coords, get_supports=True)[1][0]
                (b, n, g), device = supports.shape, supports.device
                if mask is not None:
                    supports = (supports + 1) * (~mask)[:, None, :]
                mask_supports,  mask_supports_where = supports.topk(self.mask_by_gmm, dim=2)
                mask_supports_where = mask_supports_where.view(b * n, -1)
                mask_supports_where += (torch.arange(b * n, device=device) * g)[:, None]
                mask_supports_where = mask_supports_where.flatten()
                mask_combine = torch.ones(b * n * g, device=device, dtype=torch.bool)
                mask_combine[mask_supports_where] = 0
                mask_combine = mask_combine.view(b, n, g)
                if mask is not None:
                    mask_combine = mask_combine + mask[:, None, :]
            return mask_combine
        else:
            return mask

    def forward(self, coords: T, zh: T, gmm: Optional[List[TS]] = None,  mask: TN = None,
                alpha: TN = None) -> T:
        x, pos = self.get_pos(coords)
        mask = self.get_mask_by_gmm(coords, gmm, mask)
        z = self.sdf_transformer(pos, zh, mask, alpha)
        vec = deep_sdf.expand_z(x, z)
        out = self.sdf_mlp(vec)
        if out.shape[-1] == 1:
            out = out.squeeze(-1)
        return out

    def __init__(self, opt: Options):
        super(SdfHead, self).__init__()
        self.head_type = opt.head_type
        self.mask_by_gmm = opt.mask_head_by_gmm
        if opt.pos_encoding_type == "sin":
            self.pos_encoder = mlp_models.SineLayer(constants.DIM, opt.pos_dim, is_first=True)
        else:
            self.pos_encoder = nn.Sequential(nn.Linear(constants.DIM, opt.pos_dim), nn.LeakyReLU(.2, True))
        if self.head_type == "simple":
            self.sdf_mlp = mlp_models.MLP([(opt.pos_dim + constants.DIM) * 2] +
                                          [(opt.pos_dim + constants.DIM)] * opt.head_sdf_size + [1])
        elif self.head_type == "deep_sdf":
            self.sdf_mlp = deep_sdf.DeepSDF(opt, dims=opt.head_sdf_size * [512],
                                            latent_in=(opt.head_sdf_size // 2 + opt.head_sdf_size % 2,))
        # elif self.head_type == "sin":
        #     self.pos_encoder = nn.Linear(constants.DIM, opt.pos_dim + constants.DIM)
        #     self.sdf_mlp = mlp_models.Siren(constants.DIM + opt.pos_dim + constants.DIM, 1,
        #                                     512, opt.head_sdf_size)
        else:
            raise ValueError

        self.sdf_transformer = transformer.Transformer(opt.pos_dim + constants.DIM,
                                                       opt.num_heads_head, opt.num_layers_head,
                                                       dim_ref=opt.dim_h)


class OccFormer(models_utils.Model):

    def forward_bottom(self, x):
        z_bottom = self.embedding_transformer.forward_bottom(x)
        return z_bottom

    def forward_upper(self, x):
        x = self.embedding_transformer.forward_upper(x)
        return x

    def forward_split(self, x: TS) -> Tuple[T, List[TS]]:
        b = x[0].shape[0]
        raw_gmms = [self.to_gmm(x[0]).unsqueeze(1)]
        for to_gmm, x_ in zip(self.to_gmm2, x[1:]):
            raw_gmm = to_gmm(x_)
            raw_gmm = raw_gmm.view(b, -1, x_.shape[1], raw_gmm.shape[2])
            raw_gmms.append(raw_gmm)
        gmms = [split_gm(torch.split(raw_gmm, self.split_shape, dim=3), self.as_tait_bryan) for raw_gmm in raw_gmms]
        zh = self.to_zh(x[-1])
        zh = zh.view(b, -1, zh.shape[-1])
        return zh, gmms

    @staticmethod
    def apply_gmm_affine(gmms: TS, affine: T):
        mu, p, phi, eigen = gmms
        if affine.dim() == 2:
            affine = affine.unsqueeze(0).expand(mu.shape[0], *affine.shape)
        mu_r = torch.einsum('bad, bpnd->bpna', affine, mu)
        p_r = torch.einsum('bad, bpncd->bpnca', affine, p)
        return mu_r, p_r, phi, eigen

    @staticmethod
    def concat_gmm(gmm_a: TS, gmm_b: TS):
        out = []
        num_gaussians = gmm_a[0].shape[2] // 2
        for element_a, element_b in zip(gmm_a, gmm_b):
            out.append(torch.cat((element_a[:, :, :num_gaussians], element_b[:, :, :num_gaussians]), dim=2))
        return out

    def forward_mid(self, zs) -> Tuple[T, List[TS]]:
        zh, gmms = self.forward_split(zs)
        if self.reflect is not None:
            gmms_r = [self.apply_gmm_affine(gmm, self.reflect) for gmm in gmms]
            gmms = [self.concat_gmm(gmm, gmm_r) for gmm, gmm_r in zip(gmms, gmms_r)]
        return zh, gmms

    def forward_low(self, z_init):
        zs, attn = self.embedding_transformer(z_init)
        return zs, attn

    def forward(self, z_init) -> Tuple[T, List[TS], Optional[T]]:
        zs, attn = self.forward_low(z_init)
        zh, gmms = self.forward_mid(zs)
        return zh, gmms, attn

    @staticmethod
    def get_reflection(reflect_axes: Tuple[bool, ...]):
        reflect = torch.eye(constants.DIM)
        for i in range(constants.DIM):
            if reflect_axes[i]:
                reflect[i, i] = -1
        return reflect

    def __init__(self, opt: Options):
        super(OccFormer, self).__init__()
        if sum(opt.symmetric) > 0:
            reflect = self.get_reflection(opt.symmetric)
            self.register_buffer("reflect", reflect)
        else:
            self.reflect = None
        self.as_tait_bryan = opt.as_tait_bryan
        if opt.as_tait_bryan:
            self.split_shape = tuple(constants.DIM * [constants.DIM] + [1])
        else:
            self.split_shape = tuple((constants.DIM + 2) * [constants.DIM] + [1])
        self.embedding_transformer = GmmTransformer(opt)
        self.to_gmm = nn.Linear(opt.dim_h, sum(self.split_shape))
        self.to_gmm2 = nn.ModuleList([nn.Linear(opt.dim_h, sum(self.split_shape)) for _ in range(len(opt.hierarchical))])
        self.to_zh = nn.Linear(opt.dim_h, opt.dim_h)


class OccGen(models_utils.Model):

    def get_z(self, item: T):
        if self.stash is None:
            return self.z(item)
        else:
            return self.stash[item]

    @staticmethod
    def interpolate_(z, num_between: Optional[int] = None):
        if num_between is None:
            num_between = z.shape[0]
        alphas = torch.linspace(0, 1, num_between, device=z.device)
        while alphas.dim() != z.dim():
            alphas.unsqueeze_(-1)
        z_between = alphas * z[1:2] + (- alphas + 1) * z[:1]
        return z_between

    def interpolate_higher(self, z: T, num_between: Optional[int] = None):
        z_between = self.interpolate_(z, num_between)
        zh, gmms = self.occ_former.forward_split(self.occ_former.forward_upper(z_between))
        return zh, gmms

    def interpolate(self, item_a: int, item_b: int, num_between: int):
        items = torch.tensor((item_a, item_b), dtype=torch.int64, device=self.device)
        z = self.get_z(items)
        z_between = self.interpolate_(z, num_between)
        zh, gmms, _ = self.occ_former(z_between)
        zh, _ = self.merge_zh(zh, gmms)
        return zh, gmms

    def get_disentanglement(self, items: T):
        z_a = self.get_z(items)
        z_b = self.occ_former.forward_bottom(z_a)
        zh, gmms = self.occ_former.forward_split(self.occ_former.forward_upper(z_b))
        return z_a, z_b, zh, gmms

    def get_embeddings(self, item: T):
        z = self.get_z(item)
        zh, gmms, attn = self.occ_former(z)
        return zh, z, gmms, attn

    def merge_zh_step_a(self, zh, gmms):
        gmms = self.softmax_phi(gmms)
        b, gp, g, _ = gmms[0].shape
        mu, p, phi, eigen = [item.view(b, gp * g, *item.shape[3:]) for item in gmms]
        if self.opt.as_tait_bryan:
            p = rotation_utils.get_tait_bryan_from_p(p)
        else:
            p = p.reshape(*p.shape[:2], -1)
        z_gmm = torch.cat((mu, p, phi.unsqueeze(-1), eigen), dim=2).detach()
        z_gmm = self.from_gmm(z_gmm)
        zh_ = zh + z_gmm
        return zh_

    def merge_zh(self, zh, gmms, mask: Optional[T] = None) -> T:
        zh_ = self.merge_zh_step_a(zh, gmms)
        zh_, attn = self.affine_transformer.forward_with_attention(zh_, mask=mask)
        return zh_, attn

    @staticmethod
    def softmax_phi(gmms):
        if len(gmms) == 1:
            return gmms[-1]
        elif type(gmms[0]) is T:
            return gmms
        else:
            phi = None
            for gmm in gmms:
                phi_ = gmm[2].softmax(2)
                if phi is None:
                    phi = phi_
                else:
                    phi = phi.view(phi.shape[0], -1, 1) * phi_
        mu, p, _, eigen = gmms[-1]
        return mu, p, phi, eigen

    def forward_b(self, x, zh, gmms, mask: Optional[T] = None) -> T:
        zh, _ = self.merge_zh(zh, gmms, mask)
        return self.occ_head(x, zh, gmms, mask)

    def forward_a(self, item: T):
        zh, z, gmms, attn = self.get_embeddings(item)
        return zh, z, gmms

    def get_attention(self, x, item) -> TS:
        zh, z, gmms = self.forward_a(item)
        zh, _ = self.merge_zh(zh, gmms)
        return self.occ_head.forward_attention(x, zh)

    def forward(self, x, item: T) -> Tuple[T, T, TS, T]:
        zh, z, gmms = self.forward_a(item)
        return self.forward_b(x, zh, gmms), z, gmms, zh

    def forward_mid(self, x: T, zh: T) -> Tuple[T, List[TS]]:
        zh, gmms = self.occ_former.forward_mid(zh)
        return self.forward_b(x, zh, gmms), gmms

    def stash_embedding(self, z: T):
        self.stash = z.to(self.opt.device)

    def get_dist(self) -> distributions.multivariate_normal:
        if self.dist is None:
            weights = self.z.weight.clone().detach()
            mean = weights.mean(0)
            weights = weights - mean[None, :]
            cov = torch.einsum('nd,nc->dc', weights, weights) / (weights.shape[0] - 1)
            self.dist = distributions.multivariate_normal.MultivariateNormal(mean, covariance_matrix=cov)
        return self.dist

    def log_prob(self, z):
        return self.get_dist().log_prob(z)

    def get_random_embeddings(self, num_items: int):
        dist = self.get_dist()
        z_init = dist.sample((num_items,))
        return z_init

    def random_samples(self, num_items: int, low: bool = False):
        z_init = self.get_random_embeddings(num_items)
        zh, gmms, attn = self.occ_former(z_init)
        if low:
            return zh, gmms
        zh, _ = self.merge_zh(zh, gmms)
        return zh, gmms

    def __init__(self, opt: Options):
        super(OccGen, self).__init__()
        self.device = opt.device
        self.opt = opt
        self.z = nn.Embedding(opt.dataset_size, opt.dim_z)
        torch.nn.init.normal_(
            self.z.weight.data,
            0.0,
            1. / math.sqrt(opt.dim_z),
        )
        self.occ_former = OccFormer(opt)
        self.occ_head = SdfHead(opt)
        self.from_gmm = nn.Linear(sum(self.occ_former.split_shape), opt.dim_h)
        if opt.use_encoder:
            self.affine_transformer = transformer.Transformer(opt.dim_h, opt.num_heads, opt.num_layers,
                                                              act=nnf.relu, norm_layer=nn.LayerNorm)
        else:
            self.affine_transformer = transformer.DummyTransformer()
        self.dist = None
        self.stash: TN = None


def q_to_r(q):
    shape = q.shape
    q = q.view(-1, 4)
    q_sq = 2 * q[:, :, None] * q[:, None, :]
    m00 = 1 - q_sq[:, 1, 1] - q_sq[:, 2, 2]
    m01 = q_sq[:, 0, 1] - q_sq[:, 2, 3]
    m02 = q_sq[:, 0, 2] + q_sq[:, 1, 3]

    m10 = q_sq[:, 0, 1] + q_sq[:, 2, 3]
    m11 = 1 - q_sq[:, 0, 0] - q_sq[:, 2, 2]
    m12 = q_sq[:, 1, 2] - q_sq[:, 0, 3]

    m20 = q_sq[:, 0, 2] - q_sq[:, 1, 3]
    m21 = q_sq[:, 1, 2] + q_sq[:, 0, 3]
    m22 = 1 - q_sq[:, 0, 0] - q_sq[:, 1, 1]
    r = torch.stack((m00, m01, m02, m10, m11, m12, m20, m21, m22), dim=1)
    r = r.view(*shape[:-1], 3, 3)
    return r


def r_to_q(r):
    shape = r.shape
    r = r.view(-1, 3, 3)
    qw = .5 * (1 + r[:, 0, 0] + r[:, 1, 1] + r[:, 2, 2]).sqrt()
    qx = (r[:, 2, 1] - r[:, 1, 2]) / (4 * qw)
    qy = (r[:, 0, 2] - r[:, 2, 0]) / (4 * qw)
    qz = (r[:, 1, 0] - r[:, 0, 1]) / (4 * qw)
    q = torch.stack((qx, qy, qz, qw), -1)
    q = q.view(*shape[:-2], 4)
    return q


def main():
    # rot = utils.rotation_utils.get_random_rotation(5)
    # # q = -torch.rand(1, 4)
    # # q = nnf.normalize(q, p=2, dim=-1)
    # # q[:, -1] = q[:,  -1].abs()
    # # r = q_to_r(q)
    #
    # # affine = torch.eye(3)
    # # affine[0, 0] = -1
    # # r_b = torch.einsum('ad,bdc->bac', affine, r)
    # q = r_to_q(rot)
    # r_b = q_to_r(q)
    # q_b = r_to_q(r_b)

    opt = Options(dataset_size=10)
    model = OccGen(opt)
    x = torch.rand(2, 1000, 3)
    items = torch.arange(2)
    out = model(x, items)

    # model_a = OccFormer(opt)
    # x = torch.rand(4, opt.dim_z)
    # out = model_a(x)
    # model_a = OccGen(opt).to(CUDA(0))
    # x = torch.rand(24, 1024, 3, device=CUDA(0))
    # items = torch.arange(24, device=CUDA(0))
    # out, z_mean, log_sigma, gmms = model_a(x, items)
    # print(out.shape)
    # print(z_mean.shape)
    # print(gmms[0].shape)

if __name__ == '__main__':
    from utils import mesh_utils
    main()

