import cv2
import math
from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm import tqdm

from cldm.ddim_hacked import DDIMSampler
from cldm.recognizer import crop_image,min_bounding_rect
from ldm.modules.diffusionmodules.util import noise_like, extract_into_tensor
from skimage.transform._geometric import _umeyama as get_sym_mat


class GaussianSmoothing(torch.nn.Module):
    """
    Arguments:
    Apply gaussian smoothing on a 1d, 2d or 3d tensor. Filtering is performed seperately for each channel in the input
    using a depthwise convolution.
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel. sigma (float, sequence): Standard deviation of the
        gaussian kernel. dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    # channels=1, kernel_size=kernel_size, sigma=sigma, dim=2
    def __init__(
        self,
        channels: int = 1,
        kernel_size: int = 3,
        sigma: float = 0.5,
        dim: int = 2,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, float):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-(((mgrid - mean) / (2 * std)) ** 2))
            )

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                "Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim)
            )

    def forward(self, input):
        """
        Arguments:
        Apply gaussian filter to input.
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight.to(input.dtype), groups=self.groups)


smth_3 = GaussianSmoothing(channels=3,sigma=3.0).cuda()

sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).cuda()

sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).cuda()

sobel_x = sobel_x.view(1, 1, 3, 3).expand(1,3,3,3)
sobel_y = sobel_y.view(1, 1, 3, 3).expand(1,3,3,3)

sobel_conv_x = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
sobel_conv_y = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)


sobel_conv_x.weight = nn.Parameter(sobel_x)
sobel_conv_y.weight = nn.Parameter(sobel_y)

def get_edge(attn_map):
    attn_map_clone = attn_map
    attn_map_clone = attn_map_clone / attn_map_clone.max().detach()
    attn_map_clone = F.pad(attn_map_clone, (1, 1, 1, 1), mode="reflect")
    attn_map_clone = smth_3(attn_map_clone)

    sobel_output_x = sobel_conv_x(attn_map_clone).squeeze()[1:-1, 1:-1]
    sobel_output_y = sobel_conv_y(attn_map_clone).squeeze()[1:-1, 1:-1]
    sobel_sum = sobel_output_y**2 + sobel_output_x**2
    return sobel_sum



class myDDIMSampler(DDIMSampler):

    def __init__(self, model, schedule="linear",start_step=7, end_step=20, max_op_step=5,loss_alpha=0,loss_beta=0,add_theta=0.03,save_mem=False, **kwargs):
        super().__init__(model, schedule, **kwargs)
        self.start_step = start_step
        self.end_step = end_step
        self.loss_alpha = loss_alpha
        self.loss_beta = loss_beta
        self.max_op_step = max_op_step
        self.add_theta = add_theta
        self.save_memory = save_mem
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
            if (self.loss_alpha>0 or self.loss_beta>0 ) and (cur_step <= self.end_step and cur_step >= self.start_step):#  or (total_steps-index-1 == 0):
                torch.cuda.empty_cache()
                target = img[: b // 2].clone().detach().requires_grad_(True)
                src = img[b//2:].clone().detach().requires_grad_(False)
                img = torch.cat([target,src],dim=0)
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
                    tar_pred_x0 = outs[1][:b//2]
                    src_pred_x0 = outs[1][b//2:]
                    if  self.loss_alpha>0 or self.loss_beta>0:
                        loss = self._compute_loss(src_pred_x0.detach(),tar_pred_x0, cond)
                        target = self._update_latent(target, loss, cond["text_info"]["step_size"])
                        img = torch.cat((target,src))
                        del target,src
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
        return x_prev, pred_x0, x_prev_other,pred_x0_other

    def add_glyph(self,x_src,x_tar,c):
        bsz = x_tar.shape[0]
        print("Adding Glyph")
        for i in range(bsz):
            n_lines = c["text_info"]["n_lines"][i]
            for j in range(n_lines):
                C,H,W = x_tar[i].shape 
                pos = c["text_info"]["positions"][j][i]
                pos = F.interpolate(pos.unsqueeze(0),(H,W))
                box = min_bounding_rect((rearrange(pos,"c h w -> h w c")*255).detach().cpu().numpy().astype(np.uint8))
                pts = np.float32([box[0], box[1], box[2], box[3]])
                flat_pos = c["text_info"]["flat_positions"][j][i]
                flat_pos = F.interpolate(flat_pos.unsqueeze(0),(H,W))
                flat_box = min_bounding_rect((rearrange(flat_pos,"c h w -> h w c")*255).detach().cpu().numpy().astype(np.uint8))
                flat_pts = np.float32([flat_box[0],flat_box[1],flat_box[2],flat_box[3]])
                M = get_sym_mat(pts, flat_pts, estimate_scale=True)
                T = np.array([[2 / W, 0, -1], [0, 2 / H, -1], [0, 0, 1]])
                theta = np.linalg.inv(T @ M @ np.linalg.inv(T))
                theta = torch.from_numpy(theta[:2, :]).unsqueeze(0).type(x_tar.dtype).to(x_tar.device)
                grid = F.affine_grid(theta, torch.Size([1, C, H, W]), align_corners=True)
                ref = F.grid_sample(x_src[i].unsqueeze(0), grid, align_corners=True)
                x_tar[i] = x_tar[i] + self.add_theta * F.interpolate(c["text_info"]["positions"][j][i].unsqueeze(0),(H,W))*ref
        print("Done Adding Glyph")
        return x_tar


    @staticmethod
    def _update_latent(
        latents: torch.Tensor, loss: torch.Tensor, step_size: float
    ) -> torch.Tensor:
        """Update the latent according to the computed loss."""
        # print(loss)
        grad_cond = torch.autograd.grad(
            loss.requires_grad_(True), latents, retain_graph=True
        )[0]
        # print("grad:",grad_cond.max(),grad_cond.min())
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
        print("pred_x0:",pred_x0.requires_grad)
        if self.model.use_vae_upsample:
            self.model.first_stage_model.zero_grad()
            if self.save_memory:
                self.model.low_vram_shift(is_diffusing=False)
            decode_x0 = self.model.decode_first_stage_grad(pred_x0)
            print("decode_x0:",decode_x0.requires_grad)
            if self.save_memory:
                self.model.low_vram_shift(is_diffusing=True)
        else:
            decode_x0 = torch.nn.functional.interpolate(pred_x0, size=(512, 512), mode="bilinear", align_corners=True)
            decode_x0 = decode_x0.mean(dim=1, keepdim=True)
        # print(decode_x0.requires_grad)
        decode_x0 = torch.clamp(decode_x0, -1, 1)
        # print(decode_x0.requires_grad)
        decode_x0 = (decode_x0 + 1.0) / 2.0 * 255  # -1,1 -> 0,255; n, c,h,w
        # print(decode_x0.requires_grad)
        return decode_x0

    def _compute_loss(self, src_x,tar_x, c):
        bsz = src_x.shape[0]
        tar_x = self.vae_decode(tar_x)
        src_x = self.vae_decode(src_x)
        assert tar_x.requires_grad, "Output of vae_decode should require gradients"

        gt_texts = []
        x0_texts = []
        x0_texts_src = []
        lang_weight = []
        bs_mse_loss = []
        for i in range(bsz):
            n_lines = c["text_info"]["n_lines"][i]
            # print(n_lines)
            for j in range(n_lines):  # line
                lang_weight += [1.0]  # We dont distinguish between languages here
                gt_texts += [c["text_info"]["texts"][j][i]]
                pos = c["text_info"]["positions"][j][i] * 255
                pos = F.interpolate(pos.unsqueeze(0), (tar_x.shape[-2], tar_x.shape[-1])).squeeze(0)
                pos = rearrange(pos, "c h w -> h w c")
                np_pos = pos.detach().cpu().numpy().astype(np.uint8)
                tar_x0_text = crop_image(tar_x[i], np_pos)
                x0_texts += [tar_x0_text]

                flat_pos = c["text_info"]["flat_positions"][j][i] * 255
                flat_pos = F.interpolate(flat_pos.unsqueeze(0), (tar_x.shape[-2], tar_x.shape[-1])).squeeze(0)
                flat_pos = rearrange(flat_pos, "c h w -> h w c")
                np_flat_pos = flat_pos.detach().cpu().numpy().astype(np.uint8)
                src_x0_text = crop_image(src_x[i],np_flat_pos)
                src_x0_text = F.interpolate(src_x0_text.unsqueeze(0),(tar_x0_text.shape[-2],tar_x0_text.shape[-1])).squeeze(0)
                x0_texts_src += [src_x0_text]
        for i in range(bsz):
            if len(x0_texts) > 0 and self.loss_beta > 0:
                mseloss = nn.MSELoss()
                n_lines = c["text_info"]["n_lines"][i]
                # print(x0_texts)
                for j in range(n_lines):
                    tar = x0_texts[i * n_lines + j]
                    # print(a.shape)
                    tar = (tar-tar.min())/(tar.max()-tar.min())
                    # tar_edge = get_edge(tar.unsqueeze(0))
                    src = x0_texts_src[i * n_lines + j]
                    src = (src-src.min())/(src.max()-src.min())
                    # src_edge = get_edge(src.unsqueeze(0).repeat(1,3,1,1))
                    # print(tar.shape)
                    # sp_mse_loss = min(mseloss(a,tar),mseloss(a, 1. - tar))
                    # sp_mse_loss = mseloss(src,tar)
                    x_grad = (torch.abs(tar.diff(dim=1)) - torch.abs(src.diff(dim=1))).pow(2)
                    y_grad = (torch.abs(tar.diff(dim=2)) - torch.abs(src.diff(dim=2))).pow(2)
                    x_grad = F.pad(x_grad, (0, 0, 0, 1))  # Pad along the second dimension (dim=1)
                    y_grad = F.pad(y_grad, (0, 1, 0, 0))  # Pad along the third dimension (dim=2)
                    gradient_diff = x_grad + y_grad
                    sp_mse_loss = gradient_diff.sum()/tar.numel()
                    # sp_mse_loss = mseloss(tar_edge,src_edge)
                    bs_mse_loss += [sp_mse_loss]
                    # mse_loss_debug += [sp_mse_loss.detach().cpu().numpy()]
            else:
                bs_mse_loss += [torch.tensor(0).to(tar_x.device, tar_x.dtype)]
            # loss_ocr = torch.stack(bs_ocr_loss) * self.loss_alpha #  * step_weight
        loss_mse = torch.stack(bs_mse_loss) * self.loss_beta
        # loss_ctc = torch.stack(bs_ctc_loss) * self.loss_beta # * step_weight
        # print(f'loss_ocr: {loss_ocr.mean().detach().cpu().numpy():.4f}, loss_mse: {loss_ctc.mean().detach().cpu().numpy():.4f}, Weight: loss_alpha={self.loss_alpha}, loss_beta={self.loss_beta}')
        print(f'loss_mse: {loss_mse.mean().detach().cpu().numpy():.4f}, loss_beta={self.loss_beta}')
        loss = loss_mse.mean()
        return loss
