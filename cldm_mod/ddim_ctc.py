import cv2
from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm import tqdm

from sobel_loss import get_edge
from cldm.ddim_hacked import DDIMSampler
from cldm.recognizer import crop_image
from ldm.modules.diffusionmodules.util import noise_like, extract_into_tensor


class myDDIMSampler(DDIMSampler):
    def __init__(self, model, schedule="linear",start_step=7, end_step=20, max_op_step=5,loss_alpha=0,loss_beta=0,add_theta=0.35,loss_lambda=0, **kwargs):
        super().__init__(model, schedule, **kwargs)
        self.start_step = start_step
        self.end_step = end_step
        self.op_len = end_step - start_step + 1
        self.loss_alpha = loss_alpha
        self.loss_beta = loss_beta
        self.add_theta = add_theta
        self.loss_lambda = loss_lambda
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

            cur_step = total_steps - index - 1
            cond["text_info"]["cur_step"] =  cur_step
            if (self.loss_alpha>0 or self.loss_beta>0 or self.loss_lambda > 0 ) and (cur_step <= self.end_step and cur_step >= self.start_step):#  or (total_steps-index-1 == 0):
                # ori_img = img.clone().detach().requires_grad_(False)
                img = img.clone().detach().requires_grad_(False)
                torch.cuda.empty_cache()
                if self.add_theta !=  0 :
                    img = self.add_glyph(img, cond)
                    torch.cuda.empty_cache()
                img = img.clone().detach().requires_grad_(True)
                max_op_step = self.max_op_step
                for j in range(max_op_step):
                    # place_holder_embedd = cond["c_crossattn"][0][cond["text_info"]["holder_idx"]][: b // 2].clone().requires_grad_(True)
                    # cond["c_crossattn"][0][cond["text_info"]["holder_idx"]][: b // 2] = place_holder_embedd
                    # assert cond["c_crossattn"][0][cond["text_info"]["holder_idx"]][: b // 2].requires_grad, "place_holder_embedd should require gradients"
                    # assert cond["c_crossattn"][0][cond["text_info"]["holder_idx"]].requires_grad, "whole batch should require gradients"
                    # assert cond["c_crossattn"][0].requires_grad, "c_crossattn should require gradients"
                    torch.cuda.empty_cache()
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
                    pred_x0_other = outs[3]
                    if  self.loss_alpha>0 or self.loss_beta>0 or self.loss_lambda>0:
                        loss = self._compute_loss(pred_x0_other, cond)#  + self.bg_loss(img,ori_img,cond)
                        step_size = cond["text_info"]["step_size"]# *(cond["text_info"]["angles"][0]/43)**2
                        img = self._update_latent(img, loss, step_size)
                        torch.cuda.empty_cache()
            with torch.no_grad():
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
                img,pred_x0,img_other,pred_x0_other = outs
                if callback:
                    callback(i)
                if img_callback:
                    img_callback(pred_x0, i)

                if index % log_every_t == 0 or index == total_steps - 1:
                    # print(index)
                    intermediates["x_inter"].append(img)
                    intermediates["pred_x0"].append(pred_x0)
                    intermediates["x_inter_other"].append(img_other)
                    intermediates["pred_x0_other"].append(pred_x0_other)

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
        # with torch.no_grad():
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

        # with torch.no_grad():
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
        del a_t, a_prev, sigma_t, sqrt_one_minus_at, dir_xt, noise
        torch.cuda.empty_cache()
        return x_prev,pred_x0, x_prev_other,pred_x0_other

    def add_glyph(self,x_tar,c):
        cur_step = c["text_info"]["cur_step"]
        bsz = x_tar.shape[0]
        for i in range(bsz):
            n_lines = c["text_info"]["n_lines"][i]
            for j in range(n_lines):
                glyph_img = F.interpolate(c["text_info"]["glyphs"][j][i].unsqueeze(0),(512,512)).detach()
                glyph_img = torch.cat([glyph_img,glyph_img,glyph_img],dim=1)
                # print(angle)
                # print(10**(-cur_step) * self.add_theta * 0.25*angle/45)
                # glyph_edge = get_edge(glyph_img)
                # # print(glyph_edge.shape)
                # glyph_img = glyph_img*(1 - self.loss_beta)+glyph_edge*self.loss_beta
                mask = F.interpolate(c["text_info"]["positions"][j][i].unsqueeze(0).detach(),(64,64))
                mask = mask.squeeze(0).permute(1,2,0).cpu().numpy()
                rect_mask = np.zeros(mask.shape)
                # print(mask.shape)
                contours,_ = cv2.findContours((mask*255).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                if len(contours)==0:
                    rect_mask = mask
                else:
                    rect = cv2.minAreaRect(contours[0])
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.fillPoly(rect_mask,[box],[1])
                # cv2.imshow("mask",rect_mask)
                # cv2.waitKey(0)
                rect_mask = torch.tensor(rect_mask).permute(2,0,1).unsqueeze(0).float().cuda()
                theta = 10 ** (-cur_step) * self.add_theta 
                glyph_latent = self.model.get_first_stage_encoding(self.model.encode_first_stage(glyph_img))
                x_tar[i] = x_tar[i] + theta * rect_mask * glyph_latent
        return x_tar

    @staticmethod
    def _update_latent(
        latents: torch.Tensor, loss: torch.Tensor, step_size: float
    ) -> torch.Tensor:
        """Update the latent according to the computed loss."""
        # print(loss)
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents])[0]
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

    def vae_decode(self,pred_x0):
        # print(x.requires_grad)
        # a = torch.ones_like(x)
        # print("Test: ",(x*a).requires_grad)
        # pred_x0 = self.predict_start_from_noise(x, t, model_output)
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

    def _compute_loss(self, x, c):
        bsz = x.shape[0]
        mseloss = nn.MSELoss()
        x = self.vae_decode(x)
        assert x.requires_grad, "Output of vae_decode should require gradients"

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
                pos = F.interpolate(pos.unsqueeze(0), (x.shape[-2], x.shape[-1])).squeeze(0)
                pos = rearrange(pos, "c h w -> h w c")
                np_pos = pos.detach().cpu().numpy().astype(np.uint8)
                tar_x0_text = crop_image(x[i], np_pos)
                x0_texts += [tar_x0_text]

                src = c["text_info"]["glyphs"][j][i] * 255
                src = F.interpolate(src.unsqueeze(0), (x.shape[-2], x.shape[-1])).squeeze(0)
                src = src.repeat(3, 1, 1)
                src_text = crop_image(src, np_pos)
                x0_texts_src += [src_text]
            if len(x0_texts) > 0:
                if self.loss_alpha > 0:
                    x0_list = x0_texts + x0_texts_src
                    preds, preds_neck = recog.pred_imglist(x0_list, show_debug=False)
                    assert preds.requires_grad, "preds of recog should require gradients"
                    assert preds_neck.requires_grad, "preds_neck of vae_decode should require gradients"
                    n_pairs = len(preds) // 2
                    # print(n_pairs)
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
                        # print(n_idx,":","n_idx+n_lines")
                        n_lines = c["text_info"]["n_lines"][i]
                        bs_preds_decode += [preds_decode[n_idx : n_idx + n_lines]]
                        bs_preds_ori += [preds_ori[n_idx : n_idx + n_lines]]
                        bs_preds_neck_decode += [preds_neck_decode[n_idx : n_idx + n_lines]]
                        bs_preds_neck_ori += [preds_neck_ori[n_idx : n_idx + n_lines]]
                        bs_lang_weight += [lang_weight[n_idx : n_idx + n_lines]]
                        bs_gt_texts += [gt_texts[n_idx : n_idx + n_lines]]
                        n_idx += n_lines
        for i in range(bsz):
            if self.loss_alpha>0 and len(bs_preds_neck_decode[i]) > 0:
                # print(bs_preds_decode[i])
                sp_ocr_loss = self.model.get_loss(bs_preds_neck_decode[i], bs_preds_neck_ori[i], mean=False).mean([1, 2])
                sp_ocr_loss *= bs_lang_weight[i]  # weighted by language
                bs_ocr_loss += [sp_ocr_loss.mean()]
                # ocr_loss_debug += sp_ocr_loss.detach().cpu().numpy().tolist()
            else:
                bs_ocr_loss += [torch.tensor(0).to(x.device,x.dtype)]
            if len(x0_texts) > 0 and self.loss_beta > 0:
                n_lines = c["text_info"]["n_lines"][i]
                # print(x0_texts)
                for j in range(n_lines):
                    tar = x0_texts[i * n_lines + j]
                    # print(a.shape)
                    # tar_edge = get_edge(tar.unsqueeze(0))
                    src = x0_texts_src[i * n_lines + j]
                    # src_edge = get_edge(src.unsqueeze(0).repeat(1,3,1,1))
                    # print(tar.shape)
                    # sp_mse_loss = min(mseloss(a,tar),mseloss(a, 1. - tar))
                    # sp_mse_loss = mseloss(src,tar)
                    x_grad = (torch.abs(tar.diff(dim=1)) - torch.abs(src.diff(dim=1))).pow(2)
                    y_grad = (torch.abs(tar.diff(dim=2)) - torch.abs(src.diff(dim=2))).pow(2)
                    x_grad = F.pad(x_grad, (0, 0, 0, 1))  # Pad along the second dimension (dim=1)
                    y_grad = F.pad(y_grad, (0, 1, 0, 0))  # Pad along the third dimension (dim=2)
                    gradient_diff = x_grad + y_grad
                    sp_mse_loss = gradient_diff.sum() / tar.numel()
                    # sp_mse_loss = mseloss(tar_edge,src_edge)
                    bs_mse_loss += [sp_mse_loss]
                    # mse_loss_debug += [sp_mse_loss.detach().cpu().numpy()]
            else:
                bs_mse_loss += [torch.tensor(0).to(x.device, x.dtype)]
            # loss_ocr = torch.stack(bs_ocr_loss) * self.loss_alpha #  * step_weight
        loss_ocr = torch.stack(bs_ocr_loss)*self.loss_alpha
        loss_mse =  torch.stack(bs_mse_loss) * self.loss_beta
        # loss_ctc = torch.stack(bs_ctc_loss) * self.loss_beta # * step_weight
        # print(f'loss_ocr: {loss_ocr.mean().detach().cpu().numpy():.4f}, loss_mse: {loss_ctc.mean().detach().cpu().numpy():.4f}, Weight: loss_alpha={self.loss_alpha}, loss_beta={self.loss_beta}')
        print(
            f"loss_ocr: {loss_ocr.mean().detach().cpu().numpy():.4f}, loss_mse: {loss_mse.mean().detach().cpu().numpy():.4f}"
        )
        loss = loss_ocr.mean()+loss_mse.mean()
        return loss
    
    def bg_loss(self,x,ori_x,c):
        b = x.shape[0]
        bs_bg_loss=[]
        for i in range(b):
            mseloss = nn.MSELoss()
            n_lines = c["text_info"]["n_lines"][i]
            # print(n_lines)
            pos=0
            for j in range(n_lines):  # line
                pos += c["text_info"]["positions"][j][i]
            pos = F.interpolate(pos.unsqueeze(0), (x.shape[-2], x.shape[-1])).squeeze(0)
            bs_bg_loss += [mseloss(x[i]*(1-pos),ori_x[i]*(1-pos))]
        bgloss = torch.stack(bs_bg_loss)*self.loss_lambda
        print(f"loss_bg: {bgloss.mean().detach().cpu().numpy():.4f}")
        return bgloss.mean()
        