import cv2
from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from tqdm import tqdm

from cldm.ddim_hacked import DDIMSampler
from cldm.recognizer import crop_image
from ldm.modules.diffusionmodules.util import noise_like, extract_into_tensor


class myDDIMSampler(DDIMSampler):

    def __init__(self, model, schedule="linear",start_step=7, end_step=20, max_op_step=5,loss_alpha=0,loss_beta=0, **kwargs):
        super().__init__(model, schedule, **kwargs)
        self.start_step = start_step
        self.end_step = end_step
        self.loss_alpha = loss_alpha
        self.loss_beta = loss_beta
        self.max_op_step = max_op_step
        self.sqrt_recip_alphas_cumprod_op = torch.sqrt(1.0 / self.model.alphas_cumprod_op).to(
            self.model.device
        ).detach()
        self.sqrt_recipm1_alphas_cumprod_op = torch.sqrt(
            1.0 / self.model.alphas_cumprod_op - 1
        ).to(self.model.device).detach()
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
            if unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
                model_output = self.model.apply_model(x, t, c)
            else:
                model_uncond = self.model.apply_model(x, t, unconditional_conditioning)
                model_t = self.model.apply_model(x, t, c)
                model_output = model_uncond + unconditional_guidance_scale * (
                    model_t - model_uncond
                )

        # %-----------------------------------------------------------------------------------------%
        # TODO: writing Test Time Augmentation
        if (total_steps - index - 1 <= self.end_step and total_steps - index - 1 >= self.start_step) or (total_steps-index-1 == 0):
            max_op_step = self.max_op_step
            if total_steps - index - 1 == 0:
                max_op_step = max(max_op_step, 10)
            for op_step in range(max_op_step):
                torch.cuda.empty_cache()
                x_tar = x[: b // 2].clone().detach().requires_grad_(True)
                x_src = x[b // 2 :].clone().detach()
                loss = self._compute_loss(x_src, x_tar, t, c, model_output)
                # print(loss)
                # if op_step == 0:
                #     loss_ini = loss
                # print(loss)
                # if loss < 0.5:
                #     del x_tar, x_src
                #     torch.cuda.empty_cache()
                #     break
                if loss==0:
                    break
                x_tar = self._update_latent(x_tar, loss, c["text_info"]["step_size"])
                x = torch.cat([x_tar, x_src], dim=0)
                del x_tar, x_src
                torch.cuda.empty_cache()
                with torch.no_grad():
                    if unconditional_conditioning is None or unconditional_guidance_scale == 1.0:
                        model_output = self.model.apply_model(x, t, c) # c is a dict!!!
                    else:
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

    @staticmethod
    def _update_latent(
        latents: torch.Tensor, loss: torch.Tensor, step_size: float
    ) -> torch.Tensor:
        """Update the latent according to the computed loss."""
        # print(loss)
        grad_cond = torch.autograd.grad(
            loss.requires_grad_(True), latents, retain_graph=True
        )[0]
        latents = latents - step_size * grad_cond
        return latents

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod_op, t, x_t.shape).to(
                x_t.dtype
            )
            * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod_op, t, x_t.shape).to(
                x_t.dtype
            )
            * noise
        )

    def vae_decode(self,x, t, model_output):
        # print(x.requires_grad)
        # a = torch.ones_like(x)
        # print("Test: ",(x*a).requires_grad)
        pred_x0 = self.predict_start_from_noise(x, t, model_output)
        # print(pred_x0.requires_grad)
        if self.model.use_vae_upsample:
            self.model.first_stage_model.zero_grad()
            decode_x0 = self.model.decode_first_stage_grad(pred_x0)
        else:
            decode_x0 = torch.nn.functional.interpolate(
                pred_x0, size=(512, 512), mode="bilinear", align_corners=True
            )
            decode_x0 = decode_x0.mean(dim=1, keepdim=True)
        # print(decode_x0.requires_grad)
        decode_x0 = torch.clamp(decode_x0, -1, 1)
        # print(decode_x0.requires_grad)
        decode_x0 = (decode_x0 + 1.0) / 2.0 * 255  # -1,1 -> 0,255; n, c,h,w
        # print(decode_x0.requires_grad)
        return decode_x0

    def _compute_loss(self, x_src, x_tar, t, c, model_output):
        bsz = x_tar.shape[0]

        # if self.loss_alpha > 0 or self.loss_beta > 0:
        #     self.model.text_predictor.eval()
        #     step_weight = extract_into_tensor(self.alphas_cumprod, t, x_tar.shape).reshape(len(t))
        #     if not self.model.with_step_weight:
        #         step_weight = torch.ones_like(step_weight)

        tar_x0 = self.vae_decode(x_tar, t[:bsz], model_output[:bsz])
        src_x0 = c["text_info"]['glyphs']
        # print(src_x0.shape)
        # src_x0 = f.interpolate(src_x0,(tar_x0.shape[-2],tar_x0.shape[-1]))
        # print(src_x0.shape)
        # src_x0 = self.vae_decode(x_src, t[bsz:], model_output[bsz:])
        # print(tar_x0.requires_grad)
        assert tar_x0.requires_grad, "Output of vae_decode should require gradients"

        gt_texts = []
        x0_texts = []
        x0_texts_src = []
        lang_weight = []
        recog = self.model.cn_recognizer
        bs_ocr_loss = []
        bs_ctc_loss = []
        bs_mse_loss = []
        for i in range(bsz):
            n_lines = c["text_info"]["n_lines"][i]
            # print(n_lines)

            for j in range(n_lines):  # line
                lang_weight += [1.0]  # We dont distinguish between languages here
                gt_texts += [c["text_info"]["texts"][j][i]]
                pos = c["text_info"]["positions"][j][i] * 255
                pos = rearrange(pos, "c h w -> h w c")
                np_pos = pos.detach().cpu().numpy().astype(np.uint8)
                tar_x0_text = crop_image(tar_x0[i], np_pos)
                x0_texts += [tar_x0_text]

                # flat_pos = c["text_info"]["flat_positions"][j][i] * 255
                # flat_pos = rearrange(flat_pos, "c h w -> h w c")
                # np_flat_pos = flat_pos.detach().cpu().numpy().astype(np.uint8)
                # print(src_x0_text.shape)
                src_x0_text = f.interpolate(src_x0[j],(tar_x0.shape[-2],tar_x0.shape[-1])).repeat(1,3,1,1).detach()
                src_x0_text = crop_image(src_x0_text[i],np_pos)
                # print(src_x0_text.shape)
                # src_x0_text = crop_image(src_x0[i], np_pos)
                x0_texts_src += [src_x0_text]
        if len(x0_texts) > 0:
            if self.loss_alpha > 0:
                x0_list = x0_texts + x0_texts_src
                preds, preds_neck = recog.pred_imglist(x0_list, show_debug=False)
                assert preds.requires_grad, "preds of recog should require gradients"
                assert preds_neck.requires_grad, "preds_neck of vae_decode should require gradients"
                n_pairs = len(preds) // 2
                preds_decode = preds[:n_pairs]
                preds_ori = preds[n_pairs:]
                preds_neck_decode = preds_neck[:n_pairs]
                preds_neck_ori = preds_neck[n_pairs:]
                lang_weight = torch.tensor(lang_weight).to(preds_neck.device).requires_grad_(False)
                # split to batches
                bs_preds_decode = []
                bs_preds_ori = []
                bs_preds_neck_decode = []
                bs_preds_neck_ori = []
                bs_lang_weight = []
                bs_gt_texts = []
                n_idx = 0
                for i in range(bsz):  # sample index in a batch
                    n_lines = c["text_info"]["n_lines"][i]
                    bs_preds_decode += [preds_decode[n_idx : n_idx + n_lines]]
                    bs_preds_ori += [preds_ori[n_idx : n_idx + n_lines]]
                    bs_preds_neck_decode += [preds_neck_decode[n_idx : n_idx + n_lines]]
                    bs_preds_neck_ori += [preds_neck_ori[n_idx : n_idx + n_lines]]
                    bs_lang_weight += [lang_weight[n_idx : n_idx + n_lines]]
                    bs_gt_texts += [gt_texts[n_idx : n_idx + n_lines]]
                    n_idx += n_lines

            # calc loss
            # ocr_loss_debug = []
            # ctc_loss_debug = []
            # mse_loss_debug = []
            for i in range(bsz):
                if self.loss_alpha>0 and len(bs_preds_neck_decode[i]) > 0:
                    # print(bs_preds_decode[i])
                    sp_ocr_loss = self.model.get_loss(bs_preds_neck_decode[i], bs_preds_neck_ori[i], mean=False).mean([1, 2])
                    sp_ocr_loss *= bs_lang_weight[i]  # weighted by language
                    bs_ocr_loss += [sp_ocr_loss.mean()]
                    # ocr_loss_debug += sp_ocr_loss.detach().cpu().numpy().tolist()
                else:
                    bs_ocr_loss += [torch.tensor(0).to(x_tar.device,x_tar.dtype)]
                if len(x0_texts) > 0 and self.loss_beta > 0:

                    #     sp_ctc_loss = recog.get_ctcloss(
                    #         bs_preds_decode[i], bs_gt_texts[i], bs_lang_weight[i]
                    #     )
                    #     assert bs_preds_decode[i].requires_grad, "bs_preds_decode[i] of vae_decode should require gradients"
                    #     bs_ctc_loss += [sp_ctc_loss.mean()]
                    #     ctc_loss_debug += sp_ctc_loss.detach().cpu().numpy().tolist()
                    # else:
                    #     bs_ctc_loss += [torch.tensor(0).to(x_tar.device,x_tar.dtype)]
                    mseloss = nn.MSELoss()
                    n_lines = c["text_info"]["n_lines"][i]
                    a = torch.stack(x0_texts[i*n_lines:(i+1)*n_lines])
                    a = (a-a.min())/(a.max()-a.min())
                    b = torch.stack(x0_texts_src[i*n_lines:n_lines*(i+1)])
                    b = (b-b.min())/(b.max()-b.min())
                    sp_mse_loss = min(mseloss(a,b),mseloss(a, 1. - b))
                    bs_mse_loss += [sp_mse_loss.mean()]
                    # mse_loss_debug += [sp_mse_loss.detach().cpu().numpy()]
                else:
                    bs_mse_loss += [torch.tensor(0).to(x_tar.device,x_tar.dtype)]
            loss_ocr = torch.stack(bs_ocr_loss) * self.loss_alpha #  * step_weight
            loss_mse = torch.stack(bs_mse_loss) * self.loss_beta
            # loss_ctc = torch.stack(bs_ctc_loss) * self.loss_beta # * step_weight
            # print(f'loss_ocr: {loss_ocr.mean().detach().cpu().numpy():.4f}, loss_mse: {loss_ctc.mean().detach().cpu().numpy():.4f}, Weight: loss_alpha={self.loss_alpha}, loss_beta={self.loss_beta}')
            print(f'loss_ocr: {loss_ocr.mean().detach().cpu().numpy():.4f}, loss_mse: {loss_mse.mean().detach().cpu().numpy():.4f}, Weight: loss_alpha={self.loss_alpha}, loss_beta={self.loss_beta}')

            loss = c["text_info"]["lambda"] * (loss_mse+loss_ocr)
            loss = loss.mean()
        else:
            loss = 0
        return loss
