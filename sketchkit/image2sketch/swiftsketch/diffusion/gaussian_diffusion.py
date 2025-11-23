# This code is based on https://github.com/openai/guided-diffusion
"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py
"""

import torch
import enum
import math
import numpy as np
import torch as th
from copy import deepcopy
from ..utils.sketch_utils import (
    rander_image_from_points,
    render_image_from_norm_points,
    log_model_prediction,
    log_diffusion_process_to_wandb,
)
from .Loss_computation import Loss

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, scale_betas=1., cos_power=2):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = scale_betas * 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** cos_power,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        args,
        *,
        betas,
        model_mean_type,
        model_var_type,
        rescale_timesteps=False,
        
    ):
        self.args=args
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.rescale_timesteps = rescale_timesteps



        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])
       
    
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)


        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )
        if not hasattr(self.args, 'generate'): #do it only in training 
            self.loss_func = Loss(args)

       



    def plot_forward_pass_with_predicted_x0(self, model, data, device, scale):
        batch = next(iter(data))
        x_start, _, image_features = batch
        x_start= x_start.to(device)
        image_features=image_features.to(device)

        bs= x_start.shape[0]

        xt_to_plot = []
        predicted_x0_to_plot=[]


        freq= [0, 0.05,0.1,0.2,0.4,0.6,0.8, 1.0]
        timesteps = [math.floor(t * self.args.diffusion_steps) for t in freq]
      
        timesteps_for_log=[]

        for t in timesteps:
            if t==0:
                xt_sketch_list = render_image_from_norm_points(x_start, self.args.scaling_factor, self.args.canvas_width)
                xt_to_plot.append(xt_sketch_list)
                predicted_x0_to_plot.append(xt_sketch_list)
                timesteps_for_log.append(t)

            else:
                t = t-1 # x_t (in self.q_sample ) is calculated for t+1
                t_tensor = torch.tensor([t]*bs, dtype=torch.long).to(device)

                noise = torch.randn_like(x_start)  # shape [bs, nstrokes, ncpoints, nfeats]
                
                x_t = self.q_sample(x_start, t_tensor, noise=noise)  # shape [bs, nstrokes, ncpoints, nfeats]
                xt_sketch_list = render_image_from_norm_points(x_t, self.args.scaling_factor, self.args.canvas_width)
                
   
                xt_to_plot.append(xt_sketch_list)

                model_output = model(x=x_t, ts=self._scale_timesteps(t_tensor), image_features=image_features, scale=scale) #shape [bs,nstrokes, ncpoints, nfeats]
                
                if self.model_mean_type==ModelMeanType.EPSILON:
                    model_output_points= self._predict_xstart_from_eps(x_t=x_t, t=t_tensor, eps=model_output)
                else: #self.model_mean_type==ModelMeanType.START_X:
                    model_output_points= model_output

                predict_x0_sketch_list= render_image_from_norm_points(model_output_points, self.args.scaling_factor, self.args.canvas_width)
                
                predicted_x0_to_plot.append(predict_x0_sketch_list)
                t_for_log= self._scale_timesteps(t_tensor)[0]
                timesteps_for_log.append(t_for_log)
        
        if self.args.lpips_weight==0:
            predicted_x0_to_plot=[]

        log_diffusion_process_to_wandb(timesteps_for_log, xt_to_plot, predicted_x0_to_plot, "Forward Process Grid") 

      

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the dataset for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial dataset batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, image_features= None,
            scale= None 
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param image_features: features used to condition the model's prediction. 
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
      
        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x=x, ts=self._scale_timesteps(t),image_features=image_features, scale=scale)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]



            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output  
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:  # THIS IS US!
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else: # self.model_mean_type == ModelMeanType.EPSILON:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            
                    


            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )


    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

          

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        image_features= None,
        scale= None,
        const_noise=False,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param image_features: features used to condition the model's prediction.    
        :param scale: guidance_param
        :param const_noise: If True, will noise all samples with the same noise throughout sampling


        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            image_features= image_features,
            scale= scale,
 
        )
        noise = th.randn_like(x)
        if const_noise:
            noise = noise[[0]].repeat(x.shape[0], 1, 1, 1)

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0

        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    
    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        image_features= None,
        scale= None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        dump_steps=None,
        const_noise=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).  #(args.batch_size, args.num_paths, model.npoints, model.nfeats)
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param image_features: features used to condition the model's prediction.    
        :param scale: guidance_param
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :param const_noise: If True, will noise all samples with the same noise throughout sampling
        :return: a non-differentiable batch of samples.
        """
        final = None
        if dump_steps is not None:
            dump = []


        freq= [ 1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0]

        timesteps = [math.floor(t * self.args.diffusion_steps) for t in freq]
    
        if self.args.use_wandb:
            timesteps_to_save = [timesteps[0] - t for t in timesteps[:-1]]
            timesteps_to_save.reverse()
 
        xt_Denoising_Process=[] #list of lists
        x0_Denoising_Process=[] #list of lists

 
        for i, out in enumerate(self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            image_features= image_features,
            scale= scale,
            device=device,
            progress=progress,
            skip_timesteps=skip_timesteps,
            init_image=init_image,
            const_noise=const_noise,
        )):
                   
            if self.args.use_wandb:
                if i in timesteps_to_save:
                    xt_sketch_list = render_image_from_norm_points(out["image"], self.args.scaling_factor,  self.args.canvas_width)
                    xt_Denoising_Process.append(xt_sketch_list)
                    x0_sketch_list = render_image_from_norm_points(out["pred_xstart"], self.args.scaling_factor, self.args.canvas_width)
                    x0_Denoising_Process.append(x0_sketch_list)

                if i== self.args.diffusion_steps-1:
                    xt_sketch_list = render_image_from_norm_points(out["sample"], self.args.scaling_factor, self.args.canvas_width)
                    xt_Denoising_Process.append(xt_sketch_list)
                    x0_Denoising_Process.append(xt_sketch_list)


            if dump_steps is not None and i in dump_steps:
                dump.append(deepcopy(out["sample"]))
            final = out
        if dump_steps is not None:
            return dump
 
        
        if self.args.use_wandb:
            if self.args.lpips_weight==0:
                x0_Denoising_Process=[]


            print("log Denoising Process")
            log_diffusion_process_to_wandb(timesteps[::-1], xt_Denoising_Process[::-1], x0_Denoising_Process[::-1], "Denoising Process Grid") 

        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        image_features= None,
        scale= None,
        device=None,
        progress=False,
        skip_timesteps=0,
        init_image=None,
        const_noise=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        if skip_timesteps and init_image is None:
            init_image = th.zeros_like(img)

        indices = list(range(self.num_timesteps - skip_timesteps))[::-1]

        if init_image is not None:
            my_t = th.ones([shape[0]], device=device, dtype=th.long) * indices[0]
            img = self.q_sample(init_image, my_t, img)

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)
        
        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
 
            with th.no_grad():

                sample_fn =  self.p_sample
                out = sample_fn(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    image_features= image_features,
                    scale= scale,
                    const_noise=const_noise,
                )
                out["image"]= img

                yield out
                img = out["sample"]
                


    def training_losses(self, model, x_start, x_start_randered_images, image_features,t,step, resume_step, noise=None):

        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs. (control points)
        :param x_start_randered_image: the [N x C x ...] tensor of inputs_randered_images. (the sketches for image loss)
        :param t: a batch of timestep indices.
        :param image_features : a batch of image features (the condition)
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """

    
        if noise is None:
            noise = th.randn_like(x_start)  #shape [bs,nstrokes, ncpoints, nfeats]
        x_t = self.q_sample(x_start, t, noise=noise) #shape [bs,nstrokes, ncpoints, nfeats]
  
        terms = {}
      
        model_output = model(x=x_t, ts=self._scale_timesteps(t), image_features=image_features) #shape [bs,nstrokes, ncpoints, nfeats]
       
        target = {
            ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                x_start=x_start, x_t=x_t, t=t
            )[0],
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise

        }[self.model_mean_type]
        assert model_output.shape == target.shape == x_start.shape  # [bs,nstrokes, ncpoints, nfeats]

        #get predicted points from model output

        if self.model_mean_type==ModelMeanType.START_X or self.args.lpips_weight or self.args.clip_conv_weight:
        
            if self.model_mean_type==ModelMeanType.EPSILON:
                model_output_points= self._predict_xstart_from_eps(x_t=x_t, t=t, eps=model_output)
            else: #self.model_mean_type==ModelMeanType.START_X:
                model_output_points= model_output


            if self.args.use_wandb: #log model prediction 

                if step % self.args.log_interval == 0:
                    x0_sketch= render_image_from_norm_points(x_start[0].unsqueeze(0), self.args.scaling_factor, self.args.canvas_width)[0]
                    xt_sketch= render_image_from_norm_points(x_t[0].unsqueeze(0), self.args.scaling_factor, self.args.canvas_width)[0]
                    predict_x0_sketch= render_image_from_norm_points(model_output_points[0].unsqueeze(0), self.args.scaling_factor, self.args.canvas_width)[0]
                    t_for_log= self._scale_timesteps(t)[0]
                    quartile = int(4 * t_for_log / self.num_timesteps)

                    log_model_prediction(x0_sketch, xt_sketch, predict_x0_sketch, t_for_log, quartile, step+resume_step)


        if self.args.lpips_weight or self.args.clip_conv_weight:
            #convert the normalized points back to the original range for rendering 
            unnormalized_model_output_points = model_output_points / self.args.scaling_factor
            unnormalized_model_output_points = (unnormalized_model_output_points + 1) / 2
            unnormalized_model_output_points = unnormalized_model_output_points * self.args.canvas_width

            output_rendered_images, _= rander_image_from_points(unnormalized_model_output_points,self.args.canvas_width, self.args.canvas_height)
            output_rendered_images = output_rendered_images.permute(0 ,3, 1, 2).to(x_start.device)
        else:
            output_rendered_images= torch.tensor([0.0]).to(x_start.device)

        #reshape the points for loss
        bs, nstrokes, ncpoints, nfeats = model_output.shape
        model_output= model_output.reshape(bs, nstrokes, ncpoints*nfeats)# [bs,nstrokes, ncpoints*nfeats (8)]
        target= target.reshape(bs, nstrokes, ncpoints*nfeats)# [bs,nstrokes, ncpoints*nfeats (8)]

        terms = self.loss_func(output_rendered_images,x_start_randered_images.detach(), model_output, target.detach(), mode= "train")
        terms["loss"] = sum(list(terms.values())) 
        return terms
    



def _extract_into_tensor(arr, timesteps, broadcast_shape):

    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
