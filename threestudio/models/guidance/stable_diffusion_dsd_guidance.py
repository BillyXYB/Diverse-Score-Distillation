from dataclasses import dataclass, field
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, DDIMInverseScheduler, StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, cleanup, parse_version
from threestudio.utils.ops import perpendicular_component
from threestudio.utils.typing import *


@threestudio.register("stable-diffusion-dsd-guidance")
class StableDiffusionDSDGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 100.0
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98
        trainer_max_steps: int = 5000
        use_img_loss: bool = False  # image-space SDS proposed in HiFA: https://hifa-team.github.io/HiFA-site/
        
        var_red: bool = True
        weighting_strategy: str = "sds"

        token_merging: bool = False
        token_merging_params: Optional[dict] = field(default_factory=dict)

        view_dependent_prompting: bool = True

        """Maximum number of batch items to evaluate guidance for (for debugging) and to save on disk. -1 means save all items."""
        max_items_eval: int = 4
        
        # For DSD
        n_ddim_steps: int = 10
        t_anneal: bool = True
        sqrt_anneal: bool = True # if true, anneal sqrt, which will spend less time in large time. 
        delta_min: int = 1

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading Stable Diffusion ...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "tokenizer": None,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
        }
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            **pipe_kwargs,
        ).to(self.device)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        del self.pipe.text_encoder
        cleanup()

        # Create model
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        if self.cfg.token_merging:
            import tomesd

            tomesd.apply_patch(self.unet, **self.cfg.token_merging_params)

        self.scheduler = DDIMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            subfolder="scheduler",
            torch_dtype=self.weights_dtype,
        )
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device=self.device)
        
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )
        self.grad_clip_val: Optional[float] = None

        threestudio.info(f"Loaded Stable Diffusion!")
        
        self.forward_time_steps = self.get_ddim_steps(self.num_train_timesteps-1, 0, self.cfg.n_ddim_steps, self.device)
        
        self.noise = torch.randn(1,4,64,64).to(self.device)
        
        self.MIN_STEP = int(self.num_train_timesteps * self.cfg.min_step_percent)
        self.MAX_STEP = int(self.num_train_timesteps * self.cfg.max_step_percent)
        print(f"MIN_STEP: {self.MIN_STEP}, MAX_STEP: {self.MAX_STEP}")
                        
    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 512 512"]
    ) -> Float[Tensor, "B 4 64 64"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latents: Float[Tensor, "B 4 H W"],
        latent_height: int = 64,
        latent_width: int = 64,
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latents.dtype
        latents = F.interpolate(
            latents, (latent_height, latent_width), mode="bilinear", align_corners=False
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)
    
    
    @torch.no_grad()
    def predict_noise(
        self,
        latents_noisy: Float[Tensor, "B 4 64 64"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        guidance_scale: float = 1.0,
        only_conditional: bool = False,
    ):
        batch_size = elevation.shape[0]

        if prompt_utils.use_perp_neg:
            (
                text_embeddings,
                neg_guidance_weights,
            ) = prompt_utils.get_text_embeddings_perp_neg(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
        
            latent_model_input = torch.cat([latents_noisy] * 4, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t] * 4),
                encoder_hidden_states=text_embeddings,
            )  # (4B, 3, 64, 64)

            noise_pred_text = noise_pred[:batch_size]
            noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
            noise_pred_neg = noise_pred[batch_size * 2 :]

            if only_conditional:
                noise_pred = noise_pred_text
            else:
                e_pos = noise_pred_text - noise_pred_uncond
                accum_grad = 0
                n_negative_prompts = neg_guidance_weights.shape[-1]
                for i in range(n_negative_prompts):
                    e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                    accum_grad += neg_guidance_weights[:, i].view(
                        -1, 1, 1, 1
                    ) * perpendicular_component(e_i_neg, e_pos)

                noise_pred = noise_pred_uncond + guidance_scale * (
                    e_pos + accum_grad
                )
            
        else:
            neg_guidance_weights = None
            text_embeddings = prompt_utils.get_text_embeddings(
                elevation, azimuth, camera_distances, self.cfg.view_dependent_prompting
            )
            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                # pred noise
                latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
                noise_pred = self.forward_unet(
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings,
                )

            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_text + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
        
        return noise_pred, neg_guidance_weights, text_embeddings
    
    
    def get_ddim_steps(self, start_t, end_t, num_steps, device):
        steps = torch.linspace(start_t, end_t, num_steps+1)
        ddim_timesteps = torch.tensor(steps, dtype=torch.int64).to(device)
        ddim_timesteps = torch.clamp(ddim_timesteps, max=self.num_train_timesteps-1)
        return ddim_timesteps
        
    def ddim_step_given_timesteps(self, timesteps, x_t,  prompt_utils, elevation, azimuth, camera_distances,
                                cfg=1.0, return_noise_pred_list=False):
        B = x_t.shape[0]
        
        if timesteps[0] == timesteps[-1]: # for the case when t = T
            return x_t

        pred_noise_list = []
        num_steps = len(timesteps)-1
        for ti in range(num_steps):
            t = timesteps[ti]
            t_next = timesteps[ti+1]
            alpha_prod_t = self.alphas[t]
            beta_prod_t = 1 - alpha_prod_t 
            alpha_prod_t_next = self.alphas[t_next]
            beta_prod_t_next = 1 - alpha_prod_t_next
            
            model_output, _, _ =  self.predict_noise(x_t, t.repeat([B]), prompt_utils, elevation, azimuth, camera_distances,
                                                    guidance_scale=cfg)
            pred_noise_list.append(model_output)
            pred_original_sample = (x_t - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
                
            x_t = alpha_prod_t_next ** (0.5) * pred_original_sample + beta_prod_t_next ** (0.5) * model_output
            
        if return_noise_pred_list:
            return x_t, pred_noise_list
        return x_t
            

    def get_noise_from_target(self, target, cur_xt, t):
        alpha_prod_t = self.scheduler.alphas_cumprod[t]
        beta_prod_t = 1 - alpha_prod_t
        noise = (cur_xt - target * alpha_prod_t ** (0.5)) / (beta_prod_t ** (0.5))
        return noise
    
    def get_x0(self, original_samples, noise_pred, t):
        step_results = self.scheduler.step(noise_pred, t[0], original_samples, return_dict=True)
        if "pred_original_sample" in step_results:
            return step_results["pred_original_sample"]
        elif "denoised" in step_results:
            return step_results["denoised"]
        raise ValueError("Looks like the scheduler does not compute x0")
        
    
    @torch.no_grad()
    def compute_grad_dsd(
        self,
        latents: Float[Tensor, "B 4 64 64"],
        image: Float[Tensor, "B 3 512 512"],
        t: Int[Tensor, "B"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
    ):  
        
        plus_ratio = 0.1
        t_plus = plus_ratio * (t - self.MIN_STEP)
        t_plus = t_plus.clamp(
            torch.zeros_like(t),
            self.num_train_timesteps - t -1 
        )
        t_plus = (t_plus * torch.rand(*t.shape,device = self.device))
        t_plus = t + t_plus.to(torch.long)
        t_delta = torch.clamp(
            t_plus,
            self.cfg.delta_min, # T_min = 1
            max = self.num_train_timesteps - 1, # T_max = 999
        )
        t_delta = t_delta.reshape(t.shape)
        
        current_forward_steps = self.forward_time_steps[self.forward_time_steps > t_delta]
        current_forward_steps = torch.cat([current_forward_steps, t_delta])
    
        # Simulate the forward DDIM ODE till t_delta
        latents_noisy = self.ddim_step_given_timesteps(current_forward_steps, self.noise,  prompt_utils, elevation, azimuth, camera_distances,
                                                      cfg=self.cfg.guidance_scale, return_noise_pred_list=False)                  

        # get noise prediction at t_delta
        noise_pred_t_delta, neg_guidance_weights, text_embeddings = self.predict_noise(
            latents_noisy,
            t_delta,
            prompt_utils,
            elevation,
            azimuth,
            camera_distances,
            guidance_scale=self.cfg.guidance_scale, # NOTE: use the forward cfg
        )
        
        # approximation via interpolation
        latents_noisy = self.scheduler.add_noise(latents, noise_pred_t_delta, t_delta)
        latent_noisy_t = self.scheduler.add_noise(latents, noise_pred_t_delta, t)
        
        noise_pred_t_delta, neg_guidance_weights, text_embeddings = self.predict_noise(
            latents_noisy,
            t_delta,
            prompt_utils,
            elevation,
            azimuth,
            camera_distances,
            guidance_scale=1.0
        )
        
        noise_t_delta = noise_pred_t_delta
            
        noise_pred, neg_guidance_weights, text_embeddings = self.predict_noise(
            latent_noisy_t,
            t,
            prompt_utils,
            elevation,
            azimuth,
            camera_distances,
            guidance_scale=self.cfg.guidance_scale
        )

        # latents_denoised = self.get_x0(latent_noisy_t, noise_pred, t).detach() # (latents_noisy - sigma * noise_pred) / alpha

        grad_weight = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        grad_latents =  grad_weight * (noise_pred - noise_t_delta)
        
        target = latents - grad_latents
        
        guidance_eval_utils = {
            "use_perp_neg": prompt_utils.use_perp_neg,
            "neg_guidance_weights": neg_guidance_weights,
            "text_embeddings": text_embeddings,
            "t_orig": t,
            "latents_noisy": latents_noisy,
            "noise_pred": noise_pred,
        }
        
        return target, latents_noisy, guidance_eval_utils
        

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        elevation: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        camera_distances: Float[Tensor, "B"],
        rgb_as_latents=False,
        guidance_eval=False,
        test_call=False,
        **kwargs,
    ):
        batch_size = rgb.shape[0]

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents: Float[Tensor, "B 4 64 64"]
        rgb_BCHW_512 = F.interpolate(
            rgb_BCHW, (512, 512), mode="bilinear", align_corners=False
        )
        if rgb_as_latents:
            latents = F.interpolate(
                rgb_BCHW, (64, 64), mode="bilinear", align_corners=False
            )
        else:
            # encode image into latents with vae
            latents = self.encode_images(rgb_BCHW_512)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )

        target, noisy_img, guidance_eval_utils = self.compute_grad_dsd(
            latents,
            rgb_BCHW_512,
            t,
            prompt_utils,
            elevation,
            azimuth,
            camera_distances,
        )
            
        if test_call:
            return target, noisy_img

        loss_dsd = 0.5 * F.mse_loss(latents, target.detach(), reduction="sum") / batch_size

        guidance_out = {
            "loss_dsd": loss_dsd,
            "grad_norm": (latents - target).norm(),
            "min_step": self.min_step,
            "max_step": self.max_step,
        }

        if guidance_eval:
            guidance_eval_out = self.guidance_eval(**guidance_eval_utils)
            texts = []
            for n, e, a, c in zip(
                guidance_eval_out["noise_levels"], elevation, azimuth, camera_distances
            ):
                texts.append(
                    f"n{n:.02f}\ne{e.item():.01f}\na{a.item():.01f}\nc{c.item():.02f}"
                )
            guidance_eval_out.update({"texts": texts})
            guidance_out.update({"eval": guidance_eval_out})

        return guidance_out

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def get_noise_pred(
        self,
        latents_noisy,
        t,
        text_embeddings,
        use_perp_neg=False,
        neg_guidance_weights=None,
    ):
        batch_size = latents_noisy.shape[0]

        if use_perp_neg:
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 4, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t.reshape(1)] * 4).to(self.device),
                encoder_hidden_states=text_embeddings,
            )  # (4B, 3, 64, 64)

            noise_pred_text = noise_pred[:batch_size]
            noise_pred_uncond = noise_pred[batch_size : batch_size * 2]
            noise_pred_neg = noise_pred[batch_size * 2 :]

            e_pos = noise_pred_text - noise_pred_uncond
            accum_grad = 0
            n_negative_prompts = neg_guidance_weights.shape[-1]
            for i in range(n_negative_prompts):
                e_i_neg = noise_pred_neg[i::n_negative_prompts] - noise_pred_uncond
                accum_grad += neg_guidance_weights[:, i].view(
                    -1, 1, 1, 1
                ) * perpendicular_component(e_i_neg, e_pos)

            noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (
                e_pos + accum_grad
            )
        else:
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            noise_pred = self.forward_unet(
                latent_model_input,
                torch.cat([t.reshape(1)] * 2).to(self.device),
                encoder_hidden_states=text_embeddings,
            )
            # perform guidance (high scale from paper!)
            noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_text + self.cfg.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        return noise_pred

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def guidance_eval(
        self,
        t_orig,
        text_embeddings,
        latents_noisy,
        noise_pred,
        use_perp_neg=False,
        neg_guidance_weights=None,
    ):
        # use only 50 timesteps, and find nearest of those to t
        self.scheduler.set_timesteps(self.cfg.n_ddim_steps)
        self.scheduler.timesteps_gpu = self.scheduler.timesteps.to(self.device)
        bs = (
            min(self.cfg.max_items_eval, latents_noisy.shape[0])
            if self.cfg.max_items_eval > 0
            else latents_noisy.shape[0]
        )  # batch size
        large_enough_idxs = self.scheduler.timesteps_gpu.expand([bs, -1]) > t_orig[
            :bs
        ].unsqueeze(
            -1
        )  # sized [bs,50] > [bs,1]
        idxs = torch.min(large_enough_idxs, dim=1)[1]
        t = self.scheduler.timesteps_gpu[idxs]

        fracs = list((t / self.scheduler.config.num_train_timesteps).cpu().numpy())
        imgs_noisy = self.decode_latents(latents_noisy[:bs]).permute(0, 2, 3, 1)

        # get prev latent
        latents_1step = []
        pred_1orig = []
        for b in range(bs):
            step_output = self.scheduler.step(
                noise_pred[b : b + 1], t[b], latents_noisy[b : b + 1], eta=1
            )
            latents_1step.append(step_output["prev_sample"])
            pred_1orig.append(step_output["pred_original_sample"])
        latents_1step = torch.cat(latents_1step)
        pred_1orig = torch.cat(pred_1orig)
        imgs_1step = self.decode_latents(latents_1step).permute(0, 2, 3, 1)
        imgs_1orig = self.decode_latents(pred_1orig).permute(0, 2, 3, 1)

        latents_final = []
        for b, i in enumerate(idxs):
            latents = latents_1step[b : b + 1]
            text_emb = (
                text_embeddings[
                    [b, b + len(idxs), b + 2 * len(idxs), b + 3 * len(idxs)], ...
                ]
                if use_perp_neg
                else text_embeddings[[b, b + len(idxs)], ...]
            )
            neg_guid = neg_guidance_weights[b : b + 1] if use_perp_neg else None
            for t in tqdm(self.scheduler.timesteps[i + 1 :], leave=False):
                # pred noise
                noise_pred = self.get_noise_pred(
                    latents, t, text_emb, use_perp_neg, neg_guid
                )
                # get prev latent
                latents = self.scheduler.step(noise_pred, t, latents, eta=1)[
                    "prev_sample"
                ]
            latents_final.append(latents)

        latents_final = torch.cat(latents_final)
        imgs_final = self.decode_latents(latents_final).permute(0, 2, 3, 1)

        return {
            "bs": bs,
            "noise_levels": fracs,
            "imgs_noisy": imgs_noisy,
            "imgs_1step": imgs_1step,
            "imgs_1orig": imgs_1orig,
            "imgs_final": imgs_final,
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        if self.cfg.t_anneal:
            if self.cfg.sqrt_anneal:
                percentage = (
                    float(global_step) / self.cfg.trainer_max_steps
                ) ** 0.5  # progress percentage
            else:
                percentage = (
                    float(global_step) / self.cfg.trainer_max_steps
                ) # progress percentage
            
            if type(self.cfg.max_step_percent) not in [float, int]:
                max_step_percent = self.cfg.max_step_percent[1]
            else:
                max_step_percent = self.cfg.max_step_percent
            curr_percent = (
                max_step_percent - C(self.cfg.min_step_percent, epoch, global_step)
            ) * (1 - percentage) + C(self.cfg.min_step_percent, epoch, global_step)
            self.set_min_max_steps(
                min_step_percent=curr_percent,
                max_step_percent=curr_percent,
            )
            
            # prevent delta_min = 0 for small timesteps
            if percentage > 0.75:
                self.cfg.delta_min = 10
            
        else:
            self.set_min_max_steps(
                min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
                max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
            )
