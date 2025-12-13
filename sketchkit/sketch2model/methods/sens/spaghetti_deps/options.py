from __future__ import annotations
import os
import pickle
if pickle.HIGHEST_PROTOCOL < 5:
    import pickle5 as pickle
import constants as const
from custom_types import *


class Options:

    @property
    def num_levels(self) -> int:
        return len(self.hierarchical)

    def load(self):
        device = self.device
        if os.path.isfile(self.save_path):
            print(f'loading options from {self.save_path}')
            with open(self.save_path, 'rb') as f:
                options = pickle.load(f)
            options = backward_compatibility(options)
            options.device = device
            return options
        return self

    def save(self):
        if os.path.isdir(self.cp_folder):
            # self.already_saved = True
            with open(self.save_path, 'wb') as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @property
    def info(self) -> str:
        return f'{self.model_name}_{self.tag}'

    @property
    def cp_folder(self):
        return f'{const.CHECKPOINTS_ROOT}{self.info}'

    @property
    def save_path(self):
        return f'{const.CHECKPOINTS_ROOT}{self.info}/options.pkl'

    def fill_args(self, args):
        for arg in args:
            if hasattr(self, arg):
                setattr(self, arg, args[arg])

    def __init__(self, **kwargs):
        self.device = CUDA(0)
        # self.tag = 'airplanes_spaghetti'
        self.tag = 'spaghetti_all'
        self.dataset_name = 'shapenet_airplanes_wm_sphere_sym_train'
        self.epochs = 10000
        self.model_name = 'occ_gmm'
        self.dim_z = 256
        self.pos_dim = 256 - 3
        self.dim_h = 512
        self.dim_zh = 512 # multi-class: 768
        self.num_gaussians = 16 #multi-class: 32
        self.min_split = 4 # multi-class: 16
        self.max_split = 12 # multi-class: 24
        self.gmm_weight = 1
        self.num_layers = 4
        self.num_heads = 8
        self.batch_size = 18
        self.num_samples = 2000
        self.dataset_size = -1
        self.variational = False
        self.symmetric = (False, False, False)
        self.symmetric_loss = (True, False, False)
        self.data_symmetric = (True, False, False)
        self.variational_gamma = 1.e-1
        self.reset_ds_every = 100
        self.plot_every = 100
        self.lr_decay = .9
        self.lr_decay_every = 500
        self.warm_up = 2000
        self.temperature_decay = .99
        self.loss_func = [LossType.CROSS, LossType.HINGE, LossType.IN_OUT][2]
        self.decomposition_network = 'mlp'
        self.head_type = "deep_sdf"
        self.head_sdf_size = 2
        self.reg_weight = 1e-4
        self.num_layers_head = 4
        self.num_heads_head = 8
        self.disentanglement = True
        self.use_encoder = True
        self.disentanglement_weight = 1
        self.augmentation_rotation = 0.3
        self.augmentation_scale = .3
        self.augmentation_translation = .3
        self.as_tait_bryan = False
        self.hierarchical = ()
        self.mask_head_by_gmm = 0
        self.pos_encoding_type = 'sin'
        self.subset = -1
        self.fill_args(kwargs)


def get_labels(name : str):
    tag = 'chairs_refinement'
    spaghetti_tag = 'chairs_sym_hard'
    data_tag = "clipasso"
    point_cloud="points_cloud_chairs"
    return tag, spaghetti_tag, data_tag, point_cloud


class SketchOptions(Options):

    def __init__(self, **kwargs):
        super(SketchOptions, self).__init__(**kwargs)
        self.model_name = 'sketch2spaghetti'
        self.name_cat = "chairs"
        self.tag, self.spaghetti_tag, self.data_tag, self.point_cloud = get_labels(self.name_cat)
        self.vit_patch_size = 16
        self.batch_size = 16
        self.refinement = True
        self.max_refinement = 6 # multi-class: 12
        self.z_level = 0
        self.weight_cls = 1.0
        self.weight_occ = 0.3
        self.val_frac = 0.05

        if self.name_cat == "multiclass":
            self.batch_size = 32
            self.dim_zh = 768
            self.dim_h = 768
            self.num_gaussians = 32

        self.fill_args(kwargs)
        print("Sketch Options:")
        print("tag:", self.tag)
        print("data tag:", self.data_tag)
        print("spaghetti tag:", self.spaghetti_tag)


class OptionsSingle(Options):

    def __init__(self, **kwargs):
        super(OptionsSingle, self).__init__(**kwargs)
        self.tag = 'single_wolf_prune'
        self.dataset_name = 'MalteseFalconSolid'
        self.dim_z = 64
        self.pos_dim = 64 - 3
        self.dim_h = 64
        self.dim_zh = 64
        self.num_gaussians = 12
        self.gmm_weight = 1
        self.batch_size = 18
        self.num_samples = 3000
        self.dataset_size = 1
        self.symmetric = (False, False, False)
        self.head_type = "deep_sdf"
        self.head_sdf_size = 3
        self.reg_weight = 1e-4
        self.num_layers_head = 4
        self.num_heads_head = 4
        self.disentanglement = True
        self.disentanglement_weight = 1
        self.augmentation_rotation = .5
        self.augmentation_scale = .3
        self.augmentation_translation = .3
        self.prune_every = 200
        self.fill_args(kwargs)


class OptionsDiscriminator(Options):

    def __init__(self, **kwargs):
        super(OptionsDiscriminator, self).__init__(**kwargs)
        self.model_name = 'discriminator'
        self.discriminator_num_layers = 4
        self.discriminator_dim = 8


def backward_compatibility(opt: Options) -> Options:
    defaults = {'as_tait_bryan': False, 'head_type': "deep_sdf", "hierarchical": (),
                'decomposition_network': 'transformer', 'subset': -1,
                'mask_head_by_gmm': 0, 'use_encoder': True, 'data_symmetric': opt.symmetric,
                'pos_encoding_type': 'sin'}
    for key, item in defaults.items():
        if not hasattr(opt, key):
            setattr(opt, key, item)
    defaults_sketch = {'vit_patch_size': 32, 'refinement': False, 'max_refinement': 6,
                       'z_level': 1}
    if type(opt) is SketchOptions:
        for key, item in defaults_sketch.items():
            if not hasattr(opt, key):
                setattr(opt, key, item)
    return opt


def readAndprint(path:str):
    if os.path.isfile(path):
        print(f'loading options from {path}')
        with open(path, 'rb') as f:
            options = pickle.load(f)
            # print all fields and values of options
            for key, value in options.__dict__.items():
                print(f'{key}: {value}')
    else:
        print(f'file {path} does not exist or is not specified')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='')
    args = parser.parse_args()
    readAndprint(args.path)