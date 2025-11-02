from ..model.SwiftSketch_model import SwiftSketch
from ..diffusion import gaussian_diffusion as gd
from ..diffusion.respace import SpacedDiffusion, space_timesteps
from .parser_util import get_cond_mode

def load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])


def create_model(args):
    model = SwiftSketch(**get_model_args(args))
    return model


def create_model_and_diffusion(args):
    model = SwiftSketch(**get_model_args(args))
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion


def get_model_args(args):

    latent_dim= args.latent_dim
    ff_size=1024
    num_layers= args.layers
    num_heads= args.heads
    dropout=0.1
    activation="gelu"
    cond_mode= get_cond_mode(args)
    cond_mask_prob= args.cond_mask_prob
    image_features_type= args.image_features_type
    arch= args.arch
    emb_trans_dec=args.emb_trans_dec 
    normalize_model_output= args.normalize_model_output
    scaling_factor= args.scaling_factor


    return { 'latent_dim': latent_dim, 'ff_size': ff_size, 'num_layers': num_layers, 'num_heads': num_heads,
            'dropout': dropout, 'activation': activation , 'cond_mode': cond_mode,
            'cond_mask_prob':cond_mask_prob, 'image_features_type': image_features_type, 
            'normalize_model_output': normalize_model_output, 
            'arch': arch, 'emb_trans_dec': emb_trans_dec, 'scaling_factor': scaling_factor}


def create_gaussian_diffusion(args):
    # default params
    steps = args.diffusion_steps
    cos_power= args.cos_power
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False
    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta, cos_power)

    if args.model_mean_type=='start_x':
        model_mean_type= gd.ModelMeanType.START_X # we always predict x_start (a.k.a. x0), that's our deal!
    else: # args.model_mean_type=='epsilon':
        assert args.model_mean_type=='epsilon', "model_mean_type must be in ['start_x', 'epsilon']"
        model_mean_type= gd.ModelMeanType.EPSILON

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        args=args,
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=model_mean_type,
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        rescale_timesteps=rescale_timesteps,
    )
