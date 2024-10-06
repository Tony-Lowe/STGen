import cv2
from einops import rearrange
from matplotlib.mathtext import get_unicode_index
from networkx import read_adjlist
import torchvision.transforms.functional as TF
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial

from cldm.cldm import ControlLDM,ControlNet
from ldm.models import diffusion
from ldm.modules.diffusionmodules.util import (
    make_beta_schedule,
    extract_into_tensor,
    noise_like,
    timestep_embedding,
)
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.util import (
    log_txt_as_img,
    exists,
    default,
    ismap,
    isimage,
    mean_flat,
    count_params,
    instantiate_from_config,
)
from util import pca_compute

class MyControlNet(ControlNet):
    def forward(self, x, hint, text_info, timesteps, context, **kwargs):
        """
        context is conditioning for the model
        """
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        if self.use_fp16:
            t_emb = t_emb.half()
        emb = self.time_embed(t_emb)

        # guided_hint from text_info
        B, C, H, W = x.shape
        glyphs = torch.cat(text_info["glyphs"], dim=1).sum(dim=1, keepdim=True)
        positions = torch.cat(text_info["positions"], dim=1).sum(dim=1, keepdim=True)
        enc_glyph = self.glyph_block(glyphs, emb, context)
        enc_pos = self.position_block(positions, emb, context)
        guided_hint = self.fuse_block(
            torch.cat([enc_glyph, enc_pos, text_info["masked_x"]], dim=1)
        )  # 4,320,64,64
        guided_hint_out = guided_hint
        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            # if isinstance(module, TimestepEmbedSequential):
            #     for n,m in module.named_modules():
            #         if "attn1" in n.split(".")[-1]:
            #             print(m.query_dim)
            if guided_hint is not None:
                h = module(h, emb, context)  # shape batchsize, 320, 64, 64
                h += guided_hint  # Auxiliary Latent plus Latent
                guided_hint = None
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs,guided_hint_out
class MyControlLDM(ControlLDM):
    def __init__(self, control_stage_config, control_key, glyph_key, position_key, only_mid_control, loss_alpha=0, loss_beta=0, with_step_weight=False, use_vae_upsample=False, latin_weight=1, embedding_manager_config=None, *args, **kwargs):
        super().__init__(control_stage_config, control_key, glyph_key, position_key, only_mid_control, loss_alpha, loss_beta, with_step_weight, use_vae_upsample, latin_weight, embedding_manager_config, *args, **kwargs)
        # Add your custom code here

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000, linear_start=0.0001, linear_end=0.02, cosine_s=0.008):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)
        # Add your custom code here
        if exists(given_betas):
            betas = given_betas
        else:
            betas = (
                torch.linspace(
                    linear_start**0.5, linear_end**0.5, timesteps, dtype=torch.float64
                )
                ** 2
            )
        alphas = 1.0 - betas
        self.alphas_cumprod_op = torch.cumprod(alphas, dim=0)
        assert (
            self.alphas_cumprod_op.shape[0] == self.num_timesteps
        ), "alphas have to be defined for each timestep"
        self.sqrt_recip_alphas_cumprod_op = torch.sqrt(1.0 / self.alphas_cumprod_op).to(self.device)
        self.sqrt_recipm1_alphas_cumprod_op = torch.sqrt(1.0 / self.alphas_cumprod_op - 1).to(self.device)

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model  # cldm.cldm.ControlledUnetModel
        _cond = torch.cat(cond["c_crossattn"], 1)
        _hint = torch.cat(cond["c_concat"], 1)
        if self.use_fp16:
            x_noisy = x_noisy.half()
        control = self.control_model(
            x=x_noisy,
            timesteps=t,
            context=_cond,
            hint=_hint,
            text_info=cond["text_info"],
        )  # cldm.cldm.ControlNet
        # for i, module in enumerate(diffusion_model.output_blocks):
        #     if i == 11:
        #         guide_hint = module(guide_hint, diffusion_model.time_embed(t), _cond)
        # guide_hint_for_show = diffusion_model.out(guide_hint)
        # guide_hint_for_show = pca_compute(guide_hint[0])
        # guide_hint_for_show = TF.pil_to_tensor(guide_hint_for_show).unsqueeze(0).to(self.device)
        # guide_hint_for_show = self.decode_first_stage(guide_hint_for_show)
        # guide_hint_for_show = TF.to_pil_image(guide_hint_for_show[0])
        # guide_hint_for_show.save(f"z_a.png")
        # control_for_show = control[-1][0]
        # control_for_show = pca_compute(control_for_show)
        # control_for_show.save(f"Result/{t[0]}.png")
        # print(len(control))
        control = [c * scale for c, scale in zip(control, self.control_scales)]
        # for index, c in enumerate(control):
        #     # print(c.shape)
        #     # if index==0:
        #     #     c_vis = diffusion_model.output_blocks[-1](
        #     #         torch.cat([c, c], dim=1),
        #     #         diffusion_model.time_embed(
        #     #             timestep_embedding(
        #     #                 t, diffusion_model.model_channels, repeat_only=False
        #     #             )
        #     #         ),
        #     #         _cond,
        #     #     )
        #     #     c_vis = diffusion_model.out(c)
        #     #     c_vis = extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, c_vis.shape) * c_vis
        #     #     # U,S,V = torch.pca_lowrank(rearrange(c_vis[0], "c h w -> (h w) c"))
        #     #     # c_vis = torch.matmul(c_vis[0].permute(1,2,0), V[:, :4]).permute(2,0,1).unsqueeze(0)
        #     #     # print(c_vis.shape)
        #     #     c_vis = self.decode_first_stage(c_vis)
        #     #     c_vis = torch.clamp(c_vis, -1, 1)
        #     #     # print(decode_x0.requires_grad)
        #     #     c_vis = (c_vis + 1.0) / 2.0 * 255
        #     #     c_vis =  rearrange(c_vis, "b c h w -> b h w c").cpu().numpy().clip(0, 255).astype(np.uint8)
        #     #     cv2.imwrite(f"ctrlnet_inter/flat/vae/{t[0]}_{index}.png", c_vis[0])
        #     if index<=4:
        #         c_vis = F.interpolate(c, size=(512, 512), mode='bilinear')[0]
        #         U,S,V = torch.pca_lowrank(rearrange(c_vis, "c h w -> (h w) c"))
        #         c_vis = torch.matmul(c_vis.permute(1,2,0), V[:, :3])
        #         c_vis = (c_vis - c_vis.min()) / (c_vis.max() - c_vis.min())*255
        #         c_vis = c_vis.cpu().numpy().clip(0, 255).astype(np.uint8)
        #         cv2.imwrite(f"ctrlnet_inter/add/{index}/{t[0]}.png", c_vis)

        # c_vis = pca_compute(c[0])
        # c_vis.save(f"ctrlnet_inter/lean/{t[0]}_{index}.png")
        eps = diffusion_model(
            x=x_noisy,
            timesteps=t,
            context=_cond,
            control=control,
            only_mid_control=self.only_mid_control,
            info=cond["text_info"],
        )
        return eps

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, "encode") and callable(
                self.cond_stage_model.encode
            ):
                if self.embedding_manager is not None and c["text_info"] is not None:
                    self.embedding_manager.encode_text(c["text_info"])
                if isinstance(c, dict):
                    cond_txt = c["c_crossattn"][0]
                else:
                    cond_txt = c
                if self.embedding_manager is not None:
                    cond_txt = self.cond_stage_model.encode(
                        cond_txt,
                        embedding_manager=self.embedding_manager,
                    )
                else:
                    cond_txt = self.cond_stage_model.encode(cond_txt)
                if isinstance(c, dict):
                    c["c_crossattn"][0] = cond_txt
                else:
                    c = cond_txt
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c
