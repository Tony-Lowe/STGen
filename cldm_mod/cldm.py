import torch
import torch.nn as nn
import numpy as np
from functools import partial

from cldm.cldm import ControlLDM
from ldm.modules.diffusionmodules.util import (
    make_beta_schedule,
    extract_into_tensor,
    noise_like,
)
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

