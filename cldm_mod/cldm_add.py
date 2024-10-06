from typing import Any, Optional
from scipy import spatial
from sympy import use
import torch
import torch.nn as nn
import torch.nn.functional as F
import abc
from einops import rearrange, repeat
import numpy as np
from tqdm import tqdm
from functools import reduce
from cldm.cldm import ControlLDM,ControlNet,ControlledUnetModel, timestep_embedding
from ldm.modules.diffusionmodules.util import (
    make_ddim_sampling_parameters,
    make_ddim_timesteps,
    noise_like,
    extract_into_tensor,
)
from cldm.ddim_hacked import DDIMSampler
from cldm.recognizer import crop_image
from cldm.ctrl import AttentionStore
from ldm.modules.diffusionmodules.openaimodel import (
    UNetModel,
    TimestepBlock,
    TimestepEmbedSequential,
    ResBlock,
    Downsample,
    AttentionBlock,
    Upsample,
)
from ldm.modules.attention import SpatialTransformer, BasicTransformerBlock, MemoryEfficientCrossAttention, CrossAttention, FeedForward, default, exists
from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
    normalization,
    checkpoint,
)
from util import pca_compute, show_ca, aggregate_attention

try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False


# dummy replace
def convert_module_to_f16(x):
    pass

def convert_module_to_f32(x):
    pass


class myTimestepEmbedSequential(TimestepEmbedSequential):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None, mask=None, use_masa=False):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer_masa):
                x = layer(x, context, mask, use_masa)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context, mask)
            else:
                x = layer(x)
        return x


class ControlledUnetModel_masa(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        start_step=5,
        start_layer=10,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'
        self.start_step = start_step
        self.layer_idx = 0
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")
        self.use_fp16 = use_fp16
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            else:
                raise ValueError()
        # %----------------------------------------------------------------------------------%
        # Downsample Blocks
        self.input_blocks = nn.ModuleList(
            [
                myTimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer_masa(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint,attn_com=(self.layer_idx>=start_layer)
                            )
                        )
                        self.layer_idx += 1
                self.input_blocks.append(myTimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    myTimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        # %----------------------------------------------------------------------------------%
        # Middle block
        self.middle_block = myTimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer_masa(  # always uses a self-attn
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                            disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                            use_checkpoint=use_checkpoint,attn_com=(self.layer_idx>=start_layer),
                        ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.layer_idx += 1
        self._feature_size += ch
        # %----------------------------------------------------------------------------------%
        # Upsample Block
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or i < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads_upsample,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer_masa(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint,attn_com=(self.layer_idx>=start_layer)
                            )
                        )
                        self.layer_idx+=1
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(myTimestepEmbedSequential(*layers))
                self._feature_size += ch
        # %----------------------------------------------------------------------------------%

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
            normalization(ch),
            conv_nd(dims, model_channels, n_embed, 1),
            #nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, info=None, **kwargs):
        hs = [] # residual
        mask = info['attn_mask']
        cur_step = info["cur_step"]
        # attn_ctrl = info["attn_ctrl"]
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            if self.use_fp16:
                t_emb = t_emb.half()
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context, mask, cur_step>=self.start_step)
                # print(h.shape)
                hs.append(h)
            h = self.middle_block(h, emb, context, mask, cur_step >= self.start_step)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            # h = module(h, emb, context, mask, attn_ctrl)
            h = module(h, emb, context, mask, cur_step >= self.start_step)

        h = h.type(x.dtype)
        return self.out(h)

class ControlNet_masa(ControlNet):

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        glyph_channels,
        position_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        start_layer=0,
        start_step=5,
    ):
        self.start_step = start_step
        super().__init__(image_size,
            in_channels,
            model_channels,
            glyph_channels,
            position_channels,
            num_res_blocks,
            attention_resolutions,
            dropout,
            channel_mult,
            conv_resample,
            dims,
            use_checkpoint,
            use_fp16,
            num_heads,
            num_head_channels,
            num_heads_upsample,
            use_scale_shift_norm,
            resblock_updown,
            use_new_attention_order,
            use_spatial_transformer,  # custom transformer support. True in AnyText
            transformer_depth,  # custom transformer support
            context_dim,  # custom transformer support
            n_embed,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy, # False in AnyText
            disable_self_attentions,
            num_attention_blocks,
            disable_middle_self_attn,
            use_linear_in_transformer)
        time_embed_dim = model_channels * 4
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        self.input_blocks = nn.ModuleList(
            [
                myTimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = (
                            ch // num_heads
                            if use_spatial_transformer
                            else num_head_channels
                        )
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if (
                        not exists(num_attention_blocks)
                        or nr < num_attention_blocks[level]
                    ):
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            )
                            if not use_spatial_transformer
                            else SpatialTransformer_masa(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth,
                                context_dim=context_dim,
                                disable_self_attn=disabled_sa,
                                use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint,
                                attn_com=level>=start_layer,
                            )
                        )
                self.input_blocks.append(myTimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    myTimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = myTimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            (
                AttentionBlock(
                    ch,
                    use_checkpoint=use_checkpoint,
                    num_heads=num_heads,
                    num_head_channels=dim_head,
                    use_new_attention_order=use_new_attention_order,
                )
                if not use_spatial_transformer
                else SpatialTransformer_masa(  # always uses a self-attn
                    ch,
                    num_heads,
                    dim_head,
                    depth=transformer_depth,
                    context_dim=context_dim,
                    disable_self_attn=disable_middle_self_attn,
                    use_linear=use_linear_in_transformer,
                    use_checkpoint=use_checkpoint,
                    attn_com=level>=start_layer,
                )
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch


    def forward(self, x, hint, text_info, timesteps, context,**kwargs):
        """
        context is conditioning for the model
        """
        mask = text_info["attn_mask"]
        cur_step = text_info["cur_step"]
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
        )

        # flat reference
        flat_glyphs = torch.cat(text_info["flat_glyphs"],dim=1).sum(dim=1,keepdim=True)
        flat_pos = torch.cat(text_info["flat_positions"],dim=1).sum(dim=1,keepdim=True)
        enc_glyph_f = self.glyph_block(flat_glyphs, emb, context)
        enc_pos_f = self.position_block(flat_pos, emb, context)
        flat_guided_hint = self.fuse_block(
            torch.cat([enc_glyph_f,enc_pos_f,text_info["flat_masked_x"]], dim=1)
        )

        outs = []

        h = x.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):

            if guided_hint is not None:
                h = module(h, emb, context,mask,cur_step>=self.start_step)
                # h_f = module(h_f,emb,context)
                h[:B // 2] += guided_hint  # Auxiliary Latent plus Latent
                h[B // 2:] += flat_guided_hint
                guided_hint = None
                # h = torch.concat([h,h_f],dim=1)
            else:
                h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))# .chunk(2, dim=0)[0])

        h = self.middle_block(h, emb, context,mask,cur_step>=self.start_step)
        outs.append(self.middle_block_out(h, emb, context))# .chunk(2, dim=0)[0])
        # outs = torch.chunk(outs,2,dim=0)[0]
        return outs


class SpatialTransformer_masa(SpatialTransformer):

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        disable_self_attn=False,
        use_linear=False,
        use_checkpoint=True,
        attn_com=False,
    ):
        super().__init__(in_channels,n_heads,d_head,depth,dropout,context_dim,disable_self_attn,use_linear,use_checkpoint)
        self.attn_com = attn_com
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock_masa(
                    n_heads * d_head,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    disable_self_attn=disable_self_attn,
                    checkpoint=use_checkpoint,
                    attn_com=self.attn_com,
                )
                for d in range(depth)
            ]
        )
    def forward(self, x, context=None, mask=None, use_masa=False):
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, "b c h w -> b (h w) c").contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i], mask=mask, use_masa=use_masa)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


class BasicTransformerBlock_masa(BasicTransformerBlock):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention,
    }

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        disable_self_attn=False,
        attn_com=False,
    ):
        super().__init__(dim, n_heads, d_head, dropout, context_dim, gated_ff, checkpoint)
        self.attn_com = attn_com
        self.attn1 = MemoryEfficientCrossAttention_masa(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=context_dim if self.disable_self_attn else None,
            attn_com=self.attn_com,
        )  # is a self-attention if not self.disable_self_attn
        # self.attn2 = MemoryEfficientCrossAttention_show(
        #     query_dim=dim,
        #     context_dim=context_dim,
        #     heads=n_heads,
        #     dim_head=d_head,
        #     dropout=dropout,
        #     show_ca=show_ca
        # )

    def forward(self, x, context=None, mask=None,use_masa=False):
        return checkpoint(
            self._forward, (x, context, mask,use_masa), self.parameters(), self.checkpoint
        )

    def _forward(self, x, context=None, mask=None, use_masa=False):
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None, mask=mask,use_masa=use_masa) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class MemoryEfficientCrossAttention_masa(MemoryEfficientCrossAttention):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0, attn_com=False,use_mask=True):
        super().__init__(query_dim, context_dim, heads, dim_head, dropout)
        self.attn_com = attn_com
        self.use_mask = use_mask
        # print("Successfully replaced original MemoryEfficientCA!")

    def forward(self, x, context=None, mask=None,use_masa=False):
        # print(f"Using modified at {self.query_dim}")
        if self.attn_com and use_masa:
            q = self.to_q(x)
            context = default(context, x)
            k = self.to_k(context)
            v = self.to_v(context)

            b, _, _ = q.shape
            qs = list(q.chunk(2,dim=0))
            ks = list(k.chunk(2,dim=0))
            vs = list(v.chunk(2,dim=0))
            b_ori = b // 2
            if not use_masa:
                ks[0] = torch.concat(ks, dim=1)
                vs[0] = torch.concat(vs, dim=1)
            for idx in range(len(qs)):
                qs[idx],ks[idx],vs[idx] = map(
                    lambda t: t.unsqueeze(3)
                    .reshape(b_ori, t.shape[1], self.heads, self.dim_head)
                    .permute(0, 2, 1, 3)
                    .reshape(b_ori * self.heads, t.shape[1], self.dim_head)
                    .contiguous(),
                    (qs[idx], ks[idx], vs[idx]),
                )
            if use_masa:
                if exists(mask): # mask[0] is for flat position, while mask[1] is for original position
                    mask = [
                        rearrange(
                            F.interpolate(
                                m,
                                size=(int(q.shape[1] ** 0.5), int(q.shape[1] ** 0.5)),
                                mode="nearest",
                            ),
                            "b c h w -> b (h w) c",
                        )
                        .contiguous()
                        .repeat(1, self.heads, 1, 1)
                        .reshape(b_ori * self.heads, q.shape[1], -1)
                        for m in mask
                    ]  # batch_size/2, q_len, dim_head

                else:
                    mask = [torch.ones([b_ori * self.heads, q.shape[1], self.dim_head])] * 2

                # actually compute the attention, what we cannot get enough of
                out = xformers.ops.memory_efficient_attention(qs[0]*mask[1], ks[1]*mask[0], vs[1]*mask[0], attn_bias=None, op=self.attention_op)
                b_out = xformers.ops.memory_efficient_attention(qs[0]*(1 - mask[1]), ks[0]*(1-mask[1]), vs[0]*(1-mask[1]), attn_bias=None, op=self.attention_op,)
                out = [out*mask[1] + b_out*(1-mask[1])]
                out += [
                    xformers.ops.memory_efficient_attention(qs[1], ks[1], vs[1], attn_bias=None, op=self.attention_op,)
                ]
            else:
                # if exists(mask) and self.attn_com and self.use_mask:
                #     # TODO: Rewrite Timesequntial for mask convey
                #     mask = mask[0]
                #     mask = F.dropout(mask, p=0.5)
                #     mask = torch.concat([torch.ones_like(mask),mask],dim=-1)
                #     mask = F.interpolate(mask, size=(qs[0].shape[1], ks[0].shape[1]), mode="nearest") # batch_size/2, 1, q_len, k_len
                #     mask = mask.repeat(1,self.heads,1,1)
                #     mask = mask.reshape(b_ori * self.heads, qs[0].shape[1], ks[0].shape[1])
                #     mask = torch.log(mask)
                #     # print(mask.max())
                #     # print(mask.shape) batch_size * heads, q_len, k_len
                # else:
                #     mask = None
                # actually compute the attention, what we cannot get enough of
                out = [xformers.ops.memory_efficient_attention(qs[0], ks[0], vs[0], attn_bias=None, op=self.attention_op)]
                out += [
                    xformers.ops.memory_efficient_attention(
                        torch.cat(qs[1:], dim=0),
                        torch.cat(ks[1:], dim=0),
                        torch.cat(vs[1:], dim=0),
                        attn_bias=None,
                        op=self.attention_op,
                    )
                ]

            out = torch.cat(out,dim=0)

            out = (
                out.unsqueeze(0)
                .reshape(b, self.heads, out.shape[1], self.dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b, out.shape[1], self.heads * self.dim_head)
            )
            return self.to_out(out)
        else:
            return super().forward(
                x,
                context,
                mask=None
            )
