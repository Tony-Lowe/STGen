import cv2
from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm import tqdm

from cldm.ddim_hacked import DDIMSampler
from cldm.recognizer import crop_image
from ldm.modules.diffusionmodules.util import noise_like, extract_into_tensor
from util import get_edge


class myDDIMSampler(DDIMSampler):

    def __init__(self, model, schedule="linear",start_step=7, end_step=20, max_op_step=5,loss_alpha=0,loss_beta=0,add_theta=0.35, **kwargs):
        super().__init__(model, schedule, **kwargs)
        self.start_step = start_step
        self.end_step = end_step
        self.loss_alpha = loss_alpha
        self.loss_beta = loss_beta
        self.add_theta = add_theta
        self.max_op_step = max_op_step
        self.sample_size = 64
        self.set_view_config()
        self.model.zero_grad()

    def sample(
        self,
        S,  # ddim_steps
        batch_size,  # image_count
        shape,
        conditioning=None,
        callback=None,
        normals_sequence=None,
        img_callback=None,
        quantize_x0=False,
        eta=0.0,
        mask=None,
        x0=None,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        verbose=True,
        x_T=None,
        log_every_t=100,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,  # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
        dynamic_threshold=None,
        ucg_schedule=None,
        **kwargs,
    ):
        """
        Same as DDIMSampler.sample but with the option to provide a dynamic threshold
        """
        with torch.no_grad():
            if conditioning is not None:
                if isinstance(conditioning, dict):
                    ctmp = conditioning[list(conditioning.keys())[0]]
                    while isinstance(ctmp, list):
                        ctmp = ctmp[0]
                    cbs = ctmp.shape[0]
                    if cbs != batch_size:
                        print(
                            f"Warning: Got {cbs} conditionings but batch-size is {batch_size}"
                        )

                elif isinstance(conditioning, list):
                    for ctmp in conditioning:
                        if ctmp.shape[0] != batch_size:
                            print(
                                f"Warning: Got {cbs} conditionings but batch-size is {batch_size}"
                            )

                else:
                    if conditioning.shape[0] != batch_size:
                        print(
                            f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}"
                        )

            self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f"Data shape for DDIM sampling is {size}, eta {eta}")

        samples,samples_other, intermediates = self.ddim_sampling(
            conditioning,
            size,
            callback=callback,
            img_callback=img_callback,
            quantize_denoised=quantize_x0,
            mask=mask,
            x0=x0,
            ddim_use_original_steps=False,
            noise_dropout=noise_dropout,
            temperature=temperature,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
            x_T=x_T,
            log_every_t=log_every_t,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            dynamic_threshold=dynamic_threshold,
            ucg_schedule=ucg_schedule,
        )
        return samples,samples_other, intermediates

    def ddim_sampling(
        self,
        cond,
        shape,
        x_T=None,
        ddim_use_original_steps=False,
        callback=None,
        timesteps=None,
        quantize_denoised=False,
        mask=None,
        x0=None,
        img_callback=None,
        log_every_t=100,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        dynamic_threshold=None,
        ucg_schedule=None,
    ):
        device = self.model.betas.device
        with torch.no_grad():
            b = shape[0]
            if x_T is None:
                img = torch.randn(shape, device=device)
            else:
                img = x_T

            if timesteps is None:
                timesteps = (
                    self.ddpm_num_timesteps
                    if ddim_use_original_steps
                    else self.ddim_timesteps
                )
            elif timesteps is not None and not ddim_use_original_steps:
                subset_end = (
                    int(
                        min(timesteps / self.ddim_timesteps.shape[0], 1)
                        * self.ddim_timesteps.shape[0]
                    )
                    - 1
                )
                timesteps = self.ddim_timesteps[:subset_end]

            intermediates = {"x_inter": [img], "pred_x0": [img],"pred_x0_other":[img],"x_inter_other":[img]}
            time_range = (
                reversed(range(0, timesteps))
                if ddim_use_original_steps
                else np.flip(timesteps)
            )
            total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
            print(f"Running DDIM Sampling with {total_steps} timesteps")

            iterator = tqdm(time_range, desc="DDIM Sampler", total=total_steps)

        for i, step in enumerate(iterator):
            with torch.no_grad():
                index = total_steps - i - 1
                ts = torch.full((b,), step, device=device, dtype=torch.long)

                if mask is not None:
                    assert x0 is not None
                    img_orig = self.model.q_sample(
                        x0, ts
                    )  # TODO: deterministic forward pass?
                    img = img_orig * mask + (1.0 - mask) * img

                if ucg_schedule is not None:
                    assert len(ucg_schedule) == len(time_range)
                    unconditional_guidance_scale = ucg_schedule[i]

            outs = self.p_sample_ddim(
                img,
                cond,
                ts,
                index=index,
                use_original_steps=ddim_use_original_steps,
                quantize_denoised=quantize_denoised,
                temperature=temperature,
                noise_dropout=noise_dropout,
                score_corrector=score_corrector,
                corrector_kwargs=corrector_kwargs,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,
                dynamic_threshold=dynamic_threshold,
                total_steps=total_steps,
            )
            with torch.no_grad():
                img, pred_x0,img_other,pred_x0_other = outs
                if callback:
                    callback(i)
                if img_callback:
                    img_callback(pred_x0, i)

                if index % log_every_t == 0 or index == total_steps - 1:
                    # print(index)
                    intermediates["x_inter"].append(img)
                    intermediates["pred_x0"].append(pred_x0)
                    intermediates["pred_x0_other"].append(pred_x0_other)
                    intermediates["x_inter_other"].append(img_other)

        return img,img_other, intermediates

    def p_sample_ddim(
        self,
        x,
        c,
        t,
        index,
        repeat_noise=False,
        use_original_steps=False,
        quantize_denoised=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        dynamic_threshold=None,
        total_steps=20,
    ):
        with torch.no_grad():
            b, *_, device = *x.shape, x.device
            c["text_info"]["cur_step"] = total_steps - index - 1
        #     if unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
        #         model_output = self.apply_model(x, t, c)
        #     else:
        #         model_uncond = self.model.apply_model(x, t, unconditional_conditioning)
        #         model_t = self.apply_model(x, t, c)
        #         model_output = model_uncond + unconditional_guidance_scale * (
        #             model_t - model_uncond
        #         )

        # %-----------------------------------------------------------------------------------------%
        # TODO: writing Test Time Augmentation
        if (total_steps - index - 1 <= self.end_step and total_steps - index - 1 >= self.start_step):
            max_op_step = self.max_op_step
            for op_step in range(max_op_step):
                torch.cuda.empty_cache()
                x = self.add_glyph(x,c)
                torch.cuda.empty_cache()
        with torch.no_grad():
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
                model_output = self.model.apply_model(x, t, c) # c is a dict!!!
            else:
                unconditional_conditioning["text_info"]["cur_step"] = c["text_info"]["cur_step"]
                model_uncond = self.model.apply_model(x, t, unconditional_conditioning)
                model_t = self.model.apply_model(x, t, c)
                model_output = model_uncond + unconditional_guidance_scale * (
                    model_t - model_uncond
                )
        # %-----------------------------------------------------------------------------------------%
        with torch.no_grad():
            if self.model.parameterization == "v":  # Using  default eps in anytext
                e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
            else:
                e_t = model_output

            if score_corrector is not None:  # is None in AnyText
                assert self.model.parameterization == "eps", "not implemented"
                e_t = score_corrector.modify_score(
                    self.model, e_t, x, t, c, **corrector_kwargs
                )

            alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
            alphas_prev = (
                self.model.alphas_cumprod_prev
                if use_original_steps
                else self.ddim_alphas_prev
            )
            sqrt_one_minus_alphas = (
                self.model.sqrt_one_minus_alphas_cumprod
                if use_original_steps
                else self.ddim_sqrt_one_minus_alphas
            )
            sigmas = (
                self.model.ddim_sigmas_for_original_num_steps
                if use_original_steps
                else self.ddim_sigmas
            )
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full(
                (b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device
            )

            # current prediction for x_0
            if self.model.parameterization != "v":  # Using  default eps in anytext
                pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            else:
                pred_x0 = self.model.predict_start_from_z_and_v(
                    x, t, model_output
                )  # remove the noise from the prediction

            if quantize_denoised:  # False in AnyText
                pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

            if dynamic_threshold is not None:
                raise NotImplementedError()

            # direction pointing to x_t
            dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t
            noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
            if noise_dropout > 0.0:
                noise = torch.nn.functional.dropout(noise, p=noise_dropout)
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

            pred_x0_other = self.model.predict_start_from_noise(x, t, model_output)
            x_prev_other = a_prev.sqrt() * pred_x0_other + dir_xt + noise
            return x_prev, pred_x0, x_prev_other,pred_x0_other

    def add_glyph(self,x_tar,c):
        cur_step = c["text_info"]["cur_step"]
        bsz = x_tar.shape[0]
        print(10**(-cur_step) * self.add_theta)
        for i in range(bsz):
            n_lines = c["text_info"]["n_lines"][i]
            for j in range(n_lines):
                glyph_img = F.interpolate(c["text_info"]["glyphs"][j][i].unsqueeze(0),(512,512)).detach()
                glyph_img = torch.cat([glyph_img,glyph_img,glyph_img],dim=1)
                # glyph_edge = get_edge(glyph_img)
                # # print(glyph_edge.shape)
                # glyph_img = glyph_img*(1 - self.loss_beta)+glyph_edge*self.loss_beta
                x_tar[i] = x_tar[i] + 10**(-cur_step) * self.add_theta * F.interpolate(c["text_info"]["positions"][j][i].unsqueeze(0).detach(),(64,64))*self.model.get_first_stage_encoding(self.model.encode_first_stage(self.loss_alpha * glyph_img))
        return x_tar

    def set_view_config(self, patch_size=None):
        self.view_config = {
            "window_size": patch_size if patch_size is not None else self.sample_size // 2,
            "stride": patch_size if patch_size is not None else self.sample_size // 2}
        self.view_config["context_size"] = self.sample_size - self.view_config["window_size"]

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.model.diffusion_model
        _cond = torch.cat(cond["c_crossattn"], 1)
        _hint = torch.cat(cond["c_concat"], 1)
        cur_step = cond["text_info"]["cur_step"]
        if (cur_step <= self.end_step and cur_step >= self.start_step):
            torch.cuda.empty_cache()
            x_noisy_for_ctrl = self.add_glyph(x_noisy,cond)
            torch.cuda.empty_cache()
        else:
            x_noisy_for_ctrl = x_noisy
        if self.model.use_fp16:
            x_noisy = x_noisy.half()
        control = self.model.control_model(
            x=x_noisy_for_ctrl,
            timesteps=t,
            context=_cond,
            hint=_hint,
            text_info=cond["text_info"],
        )  # cldm.cldm.ControlNet
        control = [c * scale for c, scale in zip(control, self.model.control_scales)]
        # %---------------------------------------------------------------------------%
        # Visualize control
        # for index, c in enumerate(control):
        #     if index <= 4:
        #         c_vis = F.interpolate(c, size=(512, 512), mode="bilinear")[0]
        #         U, S, V = torch.pca_lowrank(rearrange(c_vis, "c h w -> (h w) c"))
        #         c_vis = torch.matmul(c_vis.permute(1, 2, 0), V[:, :3])
        #         c_vis = (c_vis - c_vis.min()) / (c_vis.max() - c_vis.min()) * 255
        #         c_vis = c_vis.cpu().numpy().clip(0, 255).astype(np.uint8)
        #         cv2.imwrite(f"ctrlnet_inter/add/{index}/{t[0]}.png", c_vis)
        # %---------------------------------------------------------------------------%
        eps = diffusion_model(
            x=x_noisy,
            timesteps=t,
            context=_cond,
            control=control,
            only_mid_control=self.model.only_mid_control,
            info=cond["text_info"],
        )

        return eps
