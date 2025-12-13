from custom_types import *
import torch.distributions as dst
import constants as const
from utils import rotation_utils
import numpy as np


def flatten(x):
    shape = x.shape
    new_shape = [shape[0], shape[1] * shape[2]] + [s for s in shape[3:]]
    return x.view(new_shape)


def gm_loglikelihood_loss(gms: tuple, x: T, raw=False) -> TS:

    losses = []
    for gm in gms:
        if gm is not None:
            mu, p, sigma_det, phi, eigen = list(map(flatten, gm))
            phi = torch.softmax(phi, dim=1)
            eigen_inv = 1 / eigen
            sigma_inverse = torch.matmul(p.transpose(2, 3), p * eigen_inv[:, :, :, None])
            batch_size, num_points, dim = x.shape
            const_1 = phi / torch.sqrt((2 * np.pi) ** dim * sigma_det)
            distance = x[:, None, :, :] - mu[:, :, None, :]
            mahalanobis_distance = -.5 * (distance.matmul(sigma_inverse) * distance).sum(3)
            const_2, _ = mahalanobis_distance.max(1)  # for numeric stability
            mahalanobis_distance -= const_2[:, None, :]
            probs = const_1[:, :, None] * torch.exp(mahalanobis_distance)
            if raw:
                losses.append(probs)
            else:
                probs = torch.log(probs.sum(1)) + const_2
                logliklihood = probs.sum()
                loss = - logliklihood / (batch_size * num_points)
                losses.append(loss)
    return losses


def compute_plane_penalty(planes: T or None, mu: T, points: T, probs: T,
                          mask: T or int) -> T:
    if planes is None:
        return 0
    points = points[:, None, None, :, :] - mu[:, :, :, None, :]
    b, p, g, n, d = points.shape
    plane_penalty = -torch.einsum('bpgnd,bpgd->bpgn', points, planes) * probs
    plane_penalty = plane_penalty.sum(2) * mask
    plane_penalty = torch.relu(plane_penalty).sum() / (b * n)
    return plane_penalty


def gm_sample(gms:tuple, num_samples:int) -> tuple:
    gm = gms[-1]
    mu, inv_sigma, _, phi = list(map(flatten, gm))
    phi = torch.softmax(phi, dim=1)
    sigma = torch.inverse(inv_sigma) #  + (torch.eye(3) * C.EPSILON).to(mu.device)[None, None, :, :]
    b, k, d = mu.shape
    classes = torch.arange(k).to(mu.device)
    samples = []
    splits = []

    def get_model(b_id, j):
        return dst.MultivariateNormal(mu[b_id, j, :], sigma[b_id, j, :, :])

    def sample_batch(b_id):
        vs = []
        splits_ = torch.zeros(1, phi.shape[1] + 1, dtype=torch.int64)
        models = [get_model(b_id, j) for j in range(k)]
        idx = dst.Categorical(phi[b_id]).sample((num_samples,))
        children_num_samples = (idx[None, :] == classes[:, None]).sum(1)
        for j, num in enumerate(children_num_samples):
            splits_[0, j + 1] = splits_[0, j] + num
            vs.append(models[j].sample((num.item(),)))
        return torch.cat(vs, 0).unsqueeze(0), splits_

    #  yachs double for loop
    for batch_id in range(b):
        vs_, splits_ = sample_batch(batch_id)
        samples.append(vs_)
        splits.append(splits_)

    return torch.cat(samples, 0), torch.cat(splits, 0)


def hierarchical_gm_sample(gms: List[TS], num_samples: int, flatten_sigma=False) -> tuple:
    batch_size = gms[-1][0].shape[0]
    device = gms[-1][0].device
    samples = []
    splits = []

    def bottom_phi():
        if gms[0] is None:
            phi = gms[-1][3].view(batch_size, -1)
            phi = torch.softmax(phi, dim=1)
        else:
            last_phi = torch.ones(batch_size, 1, device=device)
            for gm in gms:
                _, _, _, phi, _ = gm
                # phi = phi * last_phi[:, :, None]
                phi = torch.softmax(phi, dim=2) * last_phi[:, :, None]
                last_phi = phi.view(batch_size, -1)
        return phi.view(batch_size, -1)

    def sample_g(b, j, num_samples_):
        # L = sigma[b, j, :, :].cholesky(upper=True)
        mu_ = mu[b, j, :]
        samples_ = torch.randn(num_samples_, mu_.shape[0], device=device)
        # samples_ = samples_a.mm(L)
        samples_ = samples_.mm(L[b, j])
        return samples_ + mu_[None, :]

    def sample_batch(b_id):
        vs = []
        splits_ = torch.zeros(phi.shape[1] + 1, dtype=torch.int64)
        idx = dst.Categorical(phi[b_id]).sample((num_samples,))
        children_num_samples = (idx[None, :] == classes[:, None]).sum(1)
        for j, num in enumerate(children_num_samples):
            splits_[j + 1] = splits_[j] + num
            if num > 0:
                vs.append(sample_g(b_id, j, num.item()))
        return torch.cat(vs, 0).unsqueeze(0), splits_.unsqueeze(0)

    phi = bottom_phi()
    mu, p, _, _, eigen = gms[-1]
    if flatten_sigma:
        shape = eigen.shape
        min_eigen_indices = eigen.argmin(dim=3).flatten()
        eigen = eigen.view(-1, shape[-1])
        eigen[torch.arange(eigen.shape[0]), min_eigen_indices] = const.EPSILON
        eigen = eigen.view(*shape)

    sigma = torch.matmul(p.transpose(3, 4), p * eigen[:, :, :, :, None])
    mu, sigma = mu.view(batch_size, phi.shape[1], 3), sigma.view(batch_size, -1, 3, 3)
    L = (p * torch.sqrt(eigen[:, :, :, :, None])).view(batch_size, -1, 3, 3)
    classes = torch.arange(phi.shape[1], device=device)

    for b in range(batch_size):
        vs_, splits_ = sample_batch(b)
        samples.append(vs_)
        splits.append(splits_)

    return torch.cat(samples, 0), torch.cat(splits, 0)


def eigen_penalty_loss(gms: list, eigen_penalty: float) -> T:
    eigen = gms[-1][-1]
    if eigen_penalty > 0:
        penalty = eigen.min(3)[0]
        penalty = penalty.sum() / (eigen.shape[0] * eigen.shape[1] * eigen.shape[2])
    else:
        penalty = torch.zeros(0)
    return penalty


def get_gm_support(gm, x, parent_idx):
    dim = x.shape[-1]
    mu, p, phi, eigen = gm
    sigma_det = eigen.prod(-1)
    eigen_inv = 1 / eigen
    sigma_inverse = torch.matmul(p.transpose(3, 4), p * eigen_inv[:, :, :, :, None])
    num_children = mu.shape[2]
    phi = torch.softmax(phi, dim=2)
    const_1 = phi / torch.sqrt((2 * np.pi) ** dim * sigma_det)
    mu, sigma_inverse, const_1 = [reshape_param(param, parent_idx) for param in [mu, sigma_inverse, const_1]]
    distance = x[:, None, :] - mu
    mahalanobis_distance = - .5 * torch.einsum('ngd,ngdc,ngc->ng', distance, sigma_inverse, distance)
    const_2, _ = mahalanobis_distance.max(dim=1)  # for numeric stability
    mahalanobis_distance -= const_2[:, None]
    support = const_1 * torch.exp(mahalanobis_distance)
    hard_split = torch.argmax(support, dim=1)
    parent_idx = parent_idx * num_children + hard_split
    return support, parent_idx, const_2


def reshape_param(param, parent_idx):
    return param.view([-1] + list(param.shape[2:]))[parent_idx]



def hierarchical_gm_log_likelihood_loss() :

    parent_inds = {}

    def inner_(gms: List[TS], x: T, get_supports: bool = False,
                                        mask: Optional[T] = None,
                                        reduction: str = "mean") -> Union[TS, Tuple[TS, TS]]:
        nonlocal parent_inds
        batch_size, num_points, dim = x.shape
        x = x.reshape(batch_size * num_points, dim)
        if (batch_size, num_points) not in parent_inds:
            parent_inds[(batch_size, num_points)] = torch.meshgrid([torch.arange(batch_size), torch.arange(num_points)])[0].flatten().to(x.device)
        parent_idx = parent_inds[(batch_size, num_points)]
        losses = []
        supports = []
        for idx, gm in enumerate(gms):
            if gm is None:
                continue
            support, parent_idx, const = get_gm_support(gm, x, parent_idx)
            probs = torch.log(support.sum(dim=1)) + const
            if mask is not None:
                probs = probs.masked_select(mask=mask.flatten())
            if reduction == 'none':
                likelihood = probs.view(batch_size, num_points).sum(-1)
                loss = - likelihood / num_points
            else:
                likelihood = probs.sum()
                loss = - likelihood / probs.shape[0]
            losses.append(loss)
            if get_supports:
                supports.append(support.view(batch_size, num_points, -1))
        if get_supports:
            return losses, supports
        return losses
    return inner_


def hierarchical_gm_interpolation(gms: List[TS], zs: TS, x: T) -> TS:

    batch_size, num_points, dim = x.shape
    x = x.view(batch_size * num_points, dim)
    parent_idx = torch.meshgrid([torch.arange(batch_size), torch.arange(num_points)])[0].flatten().to(x.device)
    embeddings = []

    for idx, (gm, z) in enumerate(zip(gms, zs)):
        if gm is None:
            continue
        z = reshape_param(z, parent_idx)
        support, parent_idx, _ = get_gm_support(gm, x, parent_idx)
        # support = support / support.sum(-1)[:, None]
        embedding = torch.einsum('ng,ngc->nc', support, z)
        embeddings.append(embedding.view(batch_size, num_points, -1))

    return embeddings


def soft_reshape_param(param, parent_idx):
    return param.view([-1] + list(param.shape[2:]))[parent_idx]


def get_soft_support(gm, x, parent_idx, prev_support: T, num_parents: int):
    num_points, dim = x.shape
    mu, p, sigma_det, phi, eigen = gm
    eigen_inv = 1 / eigen
    sigma_inverse = torch.matmul(p.transpose(3, 4), p * eigen_inv[:, :, :, :, None])
    num_children = mu.shape[2]
    phi = torch.softmax(phi, dim=2)
    const_1 = phi / torch.sqrt((2 * np.pi) ** dim * sigma_det)
    mu, sigma_inverse, const_1 = [soft_reshape_param(param, parent_idx) for param in [mu, sigma_inverse, const_1]]
    distance = x[:, None, None, :] - mu
    mahalanobis_distance = - .5 * torch.einsum('npgd,npgdc,npgc->npg', distance, sigma_inverse, distance)
    const_2, _ = mahalanobis_distance.max(dim=2)[0].max(dim=1)  # for numeric stability
    mahalanobis_distance -= const_2[:, None, None]
    support = (prev_support[:, :, None] * const_1 * torch.exp(mahalanobis_distance)).view(num_points, -1)
    next_support, soft_split = support.topk(num_parents, dim=1)
    next_support = next_support / next_support.sum(-1)[:, None]
    from_parent = torch.gather(parent_idx, dim=1, index=soft_split // num_children)
    from_child = soft_split % num_children
    parent_idx = from_parent * num_children + from_child
    return support, next_support, parent_idx, const_2


def soft_hierarchical_gm_log_likelihood_loss(gms: List[TS], x: T, num_parents: int = 4) -> TS:

    batch_size, num_points, dim = x.shape
    x = x.view(batch_size * num_points, dim)
    parent_idx: T = torch.meshgrid([torch.arange(batch_size), torch.arange(num_points)])[0]
    parent_idx = parent_idx.flatten().unsqueeze(-1).to(x.device)
    cur_support = torch.ones_like(parent_idx)
    losses = []

    for idx, gm in enumerate(gms):
        support, cur_support, parent_idx, const = get_soft_support(gm, x, parent_idx, cur_support, num_parents)
        probs = torch.log(support.sum(dim=1)) + const
        likelihood = probs.sum()
        loss = - likelihood / (batch_size * num_points)
        losses.append(loss)

    return losses


def soft_hierarchical_gm_interpolation(gms: List[TS], zs: TS, x: T, num_parents: int = 4) -> TS:

    batch_size, num_points, dim = x.shape
    x = x.view(batch_size * num_points, dim)
    parent_idx: T = torch.meshgrid([torch.arange(batch_size), torch.arange(num_points)])[0]
    parent_idx = parent_idx.flatten().unsqueeze(-1).to(x.device)
    cur_support = torch.ones_like(parent_idx)
    embeddings = []

    for idx, (gm, z) in enumerate(zip(gms, zs)):
        if gm is None:
            continue
        z = soft_reshape_param(z, parent_idx)
        support, cur_support, parent_idx, const = get_soft_support(gm, x, parent_idx, cur_support, num_parents)
        # support = support / support.sum(-1)[:, None]
        embedding = torch.einsum('ng,ngc->nc', support, z.view(batch_size * num_points, -1, z.shape[-1]))
        embeddings.append(embedding.view(batch_size, num_points, -1))

    return embeddings


def main():
    b, n, d = 2, 1024, 3
    splits = (32,)
    x = torch.rand(b, n, d)
    parents = 1
    zs = []
    gmms = []
    for split in splits:
        mu, p = torch.rand(b, parents, split, d), torch.rand(b, parents, split, d, d)
        sigma_det, phi, eigen = torch.rand(b, parents, split), torch.rand(b, parents, split), torch.rand(b, parents, split, d)
        gmms.append([mu, p, sigma_det, phi, eigen])
        zs.append(torch.rand(b, parents, split, 64))
        parents *= split
    loss = soft_hierarchical_gm_log_likelihood_loss(gmms, x)[0]
    loss_b = hierarchical_gm_log_likelihood_loss(gmms, x)[0]
    print(loss)
    print(loss_b)
    # embeddings = soft_hierarchical_gm_interpolation(gmms, zs, x, 4)


def split_vs_by_gmm(vs: T, gmm) -> T:
    _, supports = hierarchical_gm_log_likelihood_loss()([gmm], vs.unsqueeze(0), get_supports=True)
    supports = supports[0][0]
    return supports
    # label = supports.argmax(1)
    # return label


def split_mesh_by_gmm(mesh: T_Mesh, gmm) -> Dict[int, T]:
    faces_split = {}
    vs, faces = mesh
    vs_mid_faces = vs[faces].mean(1)
    _, supports = hierarchical_gm_log_likelihood_loss()([gmm], vs_mid_faces.unsqueeze(0), get_supports=True)
    supports = supports[0][0]
    label = supports.argmax(1)
    for i in range(gmm[1].shape[2]):
        select = label.eq(i)
        if select.any():
            faces_split[i] = faces[select]
        else:
            faces_split[i] = None
    return faces_split


def flatten_gmm(gmm: TS, as_tait_bryan: bool) -> T:
    b, gp, g, _ = gmm[0].shape
    mu, p, phi, eigen = [item.view(b, gp * g, *item.shape[3:]) for item in gmm]
    if as_tait_bryan:
        p = rotation_utils.get_tait_bryan_from_p(p)
    else:
        p = p.reshape(*p.shape[:2], -1)
    z_gmm = torch.cat((mu, p, phi.unsqueeze(-1), eigen), dim=2)
    return z_gmm


if __name__ == '__main__':
    main()