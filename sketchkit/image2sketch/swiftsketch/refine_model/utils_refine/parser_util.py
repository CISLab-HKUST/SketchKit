from argparse import ArgumentParser
import argparse
import os
import json



def parse_and_load_from_model(parser):
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    add_data_options(parser)
    add_model_options(parser)
    add_sketch_options(parser)
    add_loss_options(parser)
    args = parser.parse_args()
    args_to_overwrite = []
    for group_name in ['dataset', 'model', 'sketch', 'loss']:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    # load args from model
    refine_model_path = get_model_path_from_args()
    args_path = os.path.join(os.path.dirname(refine_model_path), 'args.json')
    assert os.path.exists(args_path), 'Arguments json file was not found!'
    with open(args_path, 'r') as fr:
        model_args = json.load(fr)

    for a in args_to_overwrite:
        if a in model_args.keys():
            setattr(args, a, model_args[a])

        elif 'cond_mode' in model_args: # backward compitability
            unconstrained = (model_args['cond_mode'] == 'no_cond')
            setattr(args, 'unconstrained', unconstrained)

        else:
            print('Warning: was not able to load [{}], using default value [{}] instead.'.format(a, args.__dict__[a]))

    if args.cond_mask_prob == 0:
        args.guidance_param = 1
    return args


def get_args_per_group_name(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    return ValueError('group_name was not found.')

def get_model_path_from_args():
    try:
        dummy_parser = ArgumentParser()
        dummy_parser.add_argument('--refine_model_path', required=True)
        dummy_args, _ = dummy_parser.parse_known_args()
        return dummy_args.refine_model_path
    except:
        raise ValueError('refine_model_path argument must be specified.')
    
def get_wandb_name(args):
        losses_weights= [args.lpips_weight, args.l1_points_weight]
        losses_name= ["lpips","L1P"]
        lst=[]
        for i in range(len(losses_weights)):
            if losses_weights[i]:
                lst.append(f"{losses_weights[i]}{losses_name[i]}")
        losses_str = "_".join(lst)

        wandb_name = f"{args.title}{args.image_features_type}_seed{args.seed}_"+losses_str
        return wandb_name
    



def add_base_options(parser):
    group = parser.add_argument_group('base')
    group.add_argument("--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU.")
    group.add_argument("--device", default=0, type=int, help="Device id to use.")
    group.add_argument("--seed", default=20, type=int, help="For fixing random seed.")
    group.add_argument("--batch_size", default=32, type=int, help="Batch size during training.")



def add_model_options(parser):
    group = parser.add_argument_group('model')
    group.add_argument("--arch", default='trans_dec',
                       choices=['trans_enc', 'trans_dec'], type=str)
    group.add_argument("--emb_trans_dec", default=0, type=int,
                       help="For trans_dec architecture only, if true, will inject condition as a class token"
                            " (in addition to cross-attention).")
    group.add_argument("--normalize_model_output", default=1, type=int,
                       help="if true, will normalize the output of the model to be in the data scale ")
    group.add_argument("--layers", default=8, type=int,
                       help="Number of layers.")
    group.add_argument("--heads", default=4, type=int,
                       help="Number of heads.")
    group.add_argument("--latent_dim", default=512, type=int,
                       help="Transformer/GRU width.")
    group.add_argument("--cond_mask_prob", default=0, type=float,
                       help="The probability of masking the condition during training."
                            " For classifier-free guidance learning.")
    group.add_argument("--unconstrained", action='store_true',
                       help="Model is trained unconditionally. That is, it is constrained by neither text nor action. "
                            "Currently tested on HumanAct12 only.")
    group.add_argument("--image_features_type", default="CLIPMiddle_layer4", type=str,
                       help="Type of image features.")
    
    

def add_sketch_options(parser):
    group = parser.add_argument_group('sketch')
    group.add_argument("--num_paths", type=int,
                        default=32, help="number of strokes")
    group.add_argument("--width", type=float,
                        default=2.5, help="stroke width")
    group.add_argument("--canvas_width", type=int, default=224,
                       help="The width hight for the loss")
    group.add_argument("--canvas_height", type=int, default=224,
                       help="The canvas hight for the loss, needs to be equal to canvas_width")



def add_loss_options(parser):
    group = parser.add_argument_group('loss')
    group.add_argument("--lpips_weight", type=float, default=0.2,
                        help="weight the lpips loss")
    group.add_argument("--l1_points_weight", type=float, default=0.0,
                        help="weight the l1 loss")

def add_data_options(parser):
    group = parser.add_argument_group('dataset')
    parser.add_argument('--train_data_dir', type=str, nargs='+', default=[],
                        help="List of training data directories")
    group.add_argument("--train_sample_dir", default="", type=str,
                       help="path to the train sample data for evaluation")
    group.add_argument("--test_dir", default="", type=str,
                       help="path to the test data for evaluation")
    group.add_argument("--cat_data_size", default=10000, type=int,
                       help="The maximum number of files to use per category (input data path)")
    group.add_argument("--scaling_factor", default=2.0, type=float,
                       help="The data will be normalized to [-1,1]*scaling_factor")
    group.add_argument("--use_data_cache", default=1, type=int,
                    help="Set to 1 to use cached data, 0 to disable caching.")
    group.add_argument("--cache_path_dir", default="", type=str,
                   help="Directory path to save the data cache. If not given, uses the save_dir")
    group.add_argument("--data_name", default="data", type=str,
                    help="Filename for the cached data.")
    group.add_argument("--target_key_name", default="svg_32s", type=str,
                    help="The name of the target SVG key in the input dicts.")
    group.add_argument("--diffusion_key_name", default="svg_diffusion", type=str,
                    help="The name of the diffusion output SVG key in the input dicts.")
    

def add_wandb_options(parser):
    group = parser.add_argument_group('wandb')
    group.add_argument("--use_wandb", type=int, default=0)
    group.add_argument("--wandb_user", type=str, default="")
    group.add_argument("--wandb_name", type=str, default="test")
    group.add_argument("--wandb_project_name", type=str, default="SwiftSketch")
    group.add_argument("--experiment_name", type=str, default="SwiftSketch")
    group.add_argument("--title", type=str, default="", help="Prefix for wandb run name and save directory.")



def add_training_options(parser):
    group = parser.add_argument_group('training')
    group.add_argument("--save_dir", required=True, type=str,
                       help="Path to save checkpoints and results.")
    group.add_argument("--overwrite", type=int, default=1,
                       help="If True, will enable to use an already existing save_dir.")
    group.add_argument("--lr", default=5e-06, type=float, help="Learning rate.")
    group.add_argument("--weight_decay", default=0.0, type=float, help="Optimizer weight decay.")
    group.add_argument("--lr_anneal_steps", default=0, type=int, help="Number of learning rate anneal steps.")
    group.add_argument("--log_interval", default=2_500, type=int,
                       help="Log losses each N steps")
    group.add_argument("--save_interval", default=5_000, type=int,
                       help="Save checkpoints and run evaluation each N steps")
    group.add_argument("--num_steps", default=50_000, type=int,
                       help="Training will stop after the specified number of steps.")
    group.add_argument("--resume_checkpoint", default="", type=str,
                       help="If not empty, will start from the specified checkpoint (path to model###.pt file).")
    group.add_argument("--sort_by", default="no_sorting", type=str, 
                    choices=["highest_point", "length", "contour_and_attn", "no_sorting"],
                    help="sorting criterion for the data. ")
                    # If the data was created using ControlSketch, it is already sorted (contour_and_attn).



def add_generate_options(parser):
    group = parser.add_argument_group('generate')
    group.add_argument("--refine_model_path", required=True, type=str,
                       help="Path to refine model####.pt file to be sampled.")
    group.add_argument("--input_data", required=True, type=str,
                       help="Path to a dir of images files/ npy dicts/ npz dicts or path to a file of an image/ npy dict/npz dict") #used for generating new sketch for images
    group.add_argument("--output_dir", default='', type=str,
                       help="Path to results dir (if not given, a 'results' folder will be created inside the data folder). "
                            "If empty, will create dir in parallel to checkpoint.")
    group.add_argument("--generate_batch_size", default=24, type=int, help="Batch size during generation.")
    group.add_argument("--use_refine", default=1, type=int,
                       help="If 1, use the full SwiftSketch pipeline with the refinement network. If 0, use only the diffusion inference process.")
    group.add_argument("--fix_scale", type=int, default=0, help="if the target image is not squared, it is recommended to fix the scale")
    group.add_argument("--save_final_sketch_in_dict", default=1, type=int,
                       help="If 1 and the input is a dict, save the final SwiftSketch SVG into the input dict.")
    group.add_argument("--save_svg", default=1, type=int,
                       help="If 1, save the final SwiftSketch SVG results into the output_dir.")
    group.add_argument("--save_diffusion_sketch_in_dict", default=0, type=int,
                       help="If 1 and the input is a dict, save the diffusion process SVG into the input dict.")
    group.add_argument("--guidance_param", default=2.5, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")
    group.add_argument("--model_path", default='', type=str,
                       help="Path to model####.pt file to be sampled.")
    
    
 

                     
    

def get_cond_mode(args):
    if args.unconstrained:
        cond_mode = 'no_cond'
    else:
        cond_mode = 'image'
    return cond_mode


def train_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_training_options(parser)
    add_sketch_options(parser)
    add_loss_options(parser)
    add_wandb_options(parser)

    return parser.parse_args()


def generate_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_wandb_options(parser)
    add_base_options(parser)
    add_generate_options(parser)
    args = parse_and_load_from_model(parser)

    return args




