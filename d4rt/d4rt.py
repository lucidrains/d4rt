from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Sequential

from x_transformers import Encoder, CrossAttender, Attention, FeedForward

# ein notation

import einx
from einops import rearrange
from einops.layers.torch import Rearrange
from torch_einops_utils import pack_with_inverse

# helpers

def exists(v):
    return v is not None

# video self attention encoder

class VideoEncoder(Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        image_size,
        patch_size,
        max_time_len,
        channels = 3,
        dim_head = 64,
        heads = 8,
        ff_glu = True,
        attn_kwargs: dict = dict(),
        ff_kwargs: dict = dict()
    ):
        super().__init__()

        dim_patch = channels * patch_size * patch_size

        self.patch_to_tokens = Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(dim_patch, dim),
            nn.LayerNorm(dim, bias = False)
        )

        self.layers = ModuleList([])

        for _ in range(depth):

            spatial_attn = Attention(dim = dim, dim_head = dim_head, heads = heads, **attn_kwargs)

            time_attn = Attention(dim = dim, dim_head = dim_head, heads = heads, **attn_kwargs)

            ff = FeedForward(dim = dim, glu = ff_glu, **ff_kwargs)

            self.layers.append(ModuleList([spatial_attn, time_attn, ff]))

        self.norm = nn.LayerNorm(dim, bias = False)

    def forward(
        self,
        video # float[b t c h w]
    ): # float[b n d]

        tokens = self.patch_to_tokens(video) # float[b t s d]

        for spatial_attn, time_attn, ff in self.layers:

            # space attn

            tokens, inverse_pack = pack_with_inverse(tokens, '* s d')

            tokens = spatial_attn(tokens) + tokens

            tokens = inverse_pack(tokens)

            # time attn

            tokens = rearrange(tokens, 'b t s d -> b s t d')

            tokens, inverse_pack = pack_with_inverse(tokens, '* t d')

            tokens = time_attn(tokens) + tokens

            tokens = inverse_pack(tokens)

            tokens = rearrange(tokens, 'b s t d -> b t s d')

            # feedforward

            tokens = ff(tokens) + tokens

        return self.norm(tokens)

# main class

class D4RT(Module):
    def __init__(
        self,
        *,
        dim,
        video_image_size,
        video_patch_size,
        video_max_time_len,
        enc_depth,
        dec_depth,
        enc_dim_head = 64,
        enc_heads = 8,
        dec_dim_head = 64,
        dec_heads = 8,
        video_enc_attn_kwargs: dict = dict(),
        video_enc_ff_kwargs: dict = dict(),
        cross_attender_kwargs: dict = dict()
    ):
        super().__init__()

        self.to_global_spatial_repr = VideoEncoder(
            dim = dim,
            depth = enc_depth,
            dim_head = enc_dim_head,
            heads = enc_heads,
            image_size = video_image_size,
            patch_size = video_patch_size,
            max_time_len = video_max_time_len,
            attn_kwargs = video_enc_attn_kwargs,
            ff_kwargs = video_enc_ff_kwargs
        )

        self.cross_attender = CrossAttender(
            dim = dim,
            depth = dec_depth,
            heads = dec_heads,
            attn_dim_head = dec_dim_head,
            **cross_attender_kwargs
        )

        self.to_pred = nn.Linear(dim, 3, bias = False)

    def forward(
        self,
        video,              # float[b t c h w]
        queries,            # float[b q d]
        points = None,      # float[b q 3]
        return_pred = False
    ):

        global_spatial_repr = self.to_global_spatial_repr(video)

        global_spatial_repr, inverse_pack_spacetime = pack_with_inverse(global_spatial_repr, 'b * d')

        queried = self.cross_attender(queries, context = global_spatial_repr)

        pred = self.to_pred(queried)

        if not exists(points):
            return pred

        loss = F.mse_loss(pred, points)

        if not return_pred:
            return loss

        return loss, pred
