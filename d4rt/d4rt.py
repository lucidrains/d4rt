from __future__ import annotations
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn, cat, pi, tensor, is_tensor
from torch.nn import Module, ModuleList, Sequential

from x_transformers import Encoder, CrossAttender, Attention, FeedForward

# ein notation

import einx
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch_einops_utils import pack_with_inverse, lens_to_mask, maybe, pad_right_ndim_to

# constants

Intermediates = namedtuple('Intermediates', ('pred', 'encoded_video', 'queries'))

# helpers

def exists(v):
    return v is not None

def divisible_by(num, den):
    return (num % den) == 0

# function for the patch embedding in the query

def extract_patches(
    video,      # float[b t c h w]
    coors,      # int[b q 2]
    time_src,   # int[b q]
    patch_size
):
    b, q, p, device = *time_src.shape, patch_size, video.device

    padded_video = F.pad(video, (p,) * 4)
    coors_with_padding = coors + p

    batch_inds = rearrange(torch.arange(b, device = device), 'b -> b 1 1 1')
    time_inds = rearrange(time_src, 'b q -> b q 1 1')

    dy = rearrange(torch.arange(p, device = device), 'p -> 1 1 p 1')
    dx = rearrange(torch.arange(p, device = device), 'p -> 1 1 1 p')

    y, x = coors_with_padding.unbind(dim = -1)
    y = rearrange(y, 'b q -> b q 1 1') + dy
    x = rearrange(x, 'b q -> b q 1 1') + dx

    patches = padded_video[batch_inds, time_inds, :, y, x]
    return rearrange(patches, 'b q p1 p2 c -> b q c p1 p2')

# fourier embed

class FourierEmbed(Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()
        assert divisible_by(dim, 2)

        self.proj = nn.Sequential(
            Rearrange('... -> ... 1'),
            nn.Linear(1, dim // 2)
        )

        self.proj.requires_grad_(False)

    def forward(
        self,
        coors,
    ):
        rand_proj = self.proj(coors.float())
        rand_proj = rearrange(rand_proj, '... two d -> ... (two d)')
        return torch.cos(2 * pi * rand_proj)

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
        video,                  # float[b t c h w],
        mask = None,            # bool[b t]
        return_hiddens = False
    ): # float[b n d]

        tokens = self.patch_to_tokens(video) # float[b t s d]

        if exists(mask):
            mask = repeat(mask, 'b ... -> (b s) ...', s = tokens.shape[-2])

        hiddens = []

        for spatial_attn, time_attn, ff in self.layers:

            # space attn

            tokens, inverse_pack = pack_with_inverse(tokens, '* s d')

            tokens = spatial_attn(tokens) + tokens

            tokens = inverse_pack(tokens)

            hiddens.append(tokens)

            # time attn

            tokens = rearrange(tokens, 'b t s d -> b s t d')

            tokens, inverse_pack = pack_with_inverse(tokens, '* t d')

            tokens = time_attn(tokens,  mask = mask) + tokens

            tokens = inverse_pack(tokens)

            tokens = rearrange(tokens, 'b s t d -> b t s d')

            hiddens.append(tokens)

            # feedforward

            tokens = ff(tokens) + tokens

            hiddens.append(tokens)

        output = self.norm(tokens)

        if not return_hiddens:
            return output

        return output, hiddens

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
        video_channels = 3,
        enc_dim_head = 64,
        enc_heads = 8,
        dec_dim_head = 64,
        dec_heads = 8,
        video_enc_attn_kwargs: dict = dict(),
        video_enc_ff_kwargs: dict = dict(),
        cross_attender_kwargs: dict = dict(),
        dec_use_flow_matching = False, # turn the decoder into conditional flow matching with clean prediction
        flow_match_timesteps = 4,
        flow_match_noise_std = 1.,
    ):
        super().__init__()

        # to queries

        self.video_patch_size = video_patch_size

        self.to_query_patch_embed = nn.Sequential(
            Rearrange('b q c p1 p2 -> b q (c p1 p2)'),
            nn.Linear(video_channels * video_patch_size * video_patch_size, dim, bias = False)
        )

        self.coor_fourier_embed = FourierEmbed(dim)
        self.time_src_embed = nn.Parameter(torch.randn(video_max_time_len, dim) * 1e-2)
        self.time_tgt_embed = nn.Parameter(torch.randn(video_max_time_len, dim) * 1e-2)
        self.time_camera_embed = nn.Parameter(torch.randn(video_max_time_len, dim) * 1e-2)

        self.norm_queries = nn.LayerNorm(dim, bias = False)

        # encoder

        self.video_encoder = VideoEncoder(
            dim = dim,
            depth = enc_depth,
            dim_head = enc_dim_head,
            heads = enc_heads,
            image_size = video_image_size,
            patch_size = video_patch_size,
            max_time_len = video_max_time_len,
            channels = video_channels,
            attn_kwargs = video_enc_attn_kwargs,
            ff_kwargs = video_enc_ff_kwargs
        )

        # decoder

        self.cross_attender = CrossAttender(
            dim = dim,
            depth = dec_depth,
            heads = dec_heads,
            attn_dim_head = dec_dim_head,
            **cross_attender_kwargs
        )

        # prediction

        self.to_pred = nn.Linear(dim, 3, bias = False)

        # improvisation - turn decoder into conditional flow matching

        self.dec_use_flow_matching = dec_use_flow_matching

        self.flow_match_timesteps = flow_match_timesteps
        self.flow_match_noise_std = flow_match_noise_std

        self.noised_and_times_to_embed = nn.Linear(4, dim, bias = False) if dec_use_flow_matching else None

        self.register_buffer('zero', tensor(0.), persistent = False)

    @property
    def device(self):
        return self.zero.device

    @torch.no_grad()
    def flow_matching_sample(
        self,
        video,
        *,
        coors,
        time_src,
        time_tgt,
        time_camera,
        video_lens,
        query_lens
    ):
        assert self.dec_use_flow_matching

        batch, max_queries, timesteps = *coors.shape[:2], self.flow_match_timesteps

        # pure noise

        noised_points = torch.randn((batch, max_queries, 3), device = self.device) * self.flow_match_noise_std

        # caching

        queries = encoded_video = None

        # times

        delta_time = 1. / timesteps

        times = torch.linspace(0., 1., timesteps, device = self.device)

        for time in times:

            pred, intermediates = self.forward(
                video,
                coors = coors,
                time_src = time_src,
                time_tgt = time_tgt,
                time_camera = time_camera,
                video_lens = video_lens,
                times = time,
                noised_points = noised_points,
                encoded_video = encoded_video,
                return_intermediates = True
            )

            flow = (pred - noised_points) / (1. - time).clamp(min = 1e-1)

            noised_points = noised_points + flow * delta_time

            # caching

            encoded_video = intermediates.encoded_video
            queries = intermediates.queries

        return noised_points

    def forward(
        self,
        video,                # float[b t c h w]
        *,
        coors = None,         # int[b q 2]
        time_src = None,      # int[b q]
        time_tgt = None,      # int[b q]
        time_camera = None,   # int[b q]
        queries = None,       # float[b q d]
        points = None,        # float[b q 3]
        video_lens = None,    # int[b]
        query_lens = None,    # int[b q]
        encoded_video = None,
        times = None,
        noised_points = None, # float[b q 3]
        return_intermediates = False
    ):
        inferencing = not exists(points)
        batch, max_time = video.shape[:2]

        # route to another function if inferencing and decoder is doing flow matching

        if self.dec_use_flow_matching and inferencing and (not exists(noised_points) and not exists(times)):
            return self.flow_matching_sample(video, coors = coors, time_src = time_src, time_tgt = time_tgt, time_camera = time_camera, video_lens = video_lens, query_lens = query_lens)

        # embedding to queries

        assert (
            exists(queries) or
            all([exists(p) for p in (coors, time_src, time_tgt, time_camera)])
        ), 'either `queries` is passed in, or you pass in all the needed inputs to compose the query'

        if not exists(queries):
            patch_size = self.video_patch_size

            patches = extract_patches(video, coors, time_src, patch_size)

            queries = (
                self.to_query_patch_embed(patches) +
                self.coor_fourier_embed(coors) +
                self.time_src_embed[time_src] +
                self.time_tgt_embed[time_tgt] +
                self.time_camera_embed[time_camera]
            )

            queries = self.norm_queries(queries)

        # maybe embed noise

        max_queries = queries.shape[1]

        if self.dec_use_flow_matching:
            if not inferencing:
                # training

                assert not exists(noised_points)

                times = torch.randint(0, self.flow_match_timesteps, (batch, max_queries, 1), device = self.device) / self.flow_match_timesteps

                padded_times = pad_right_ndim_to(times, 3)

                noise = torch.randn_like(points)
                noise = noise * self.flow_match_noise_std

                noised_points = noise.lerp(points, padded_times)

            else:
                # inferencing

                assert exists(noised_points) and exists(times)

                if times.ndim == 0:
                    times = rearrange(times, ' -> 1')
                if times.ndim == 1:
                    times = repeat(times, '1 -> b q 1', b = batch, q = max_queries)
                elif times.ndim == 2:
                    times = rearrange(times, 'b q -> b q 1')

            noised_points_and_times = cat((noised_points, times), dim = -1)

            noise_embed = self.noised_and_times_to_embed(noised_points_and_times)

            queries = queries + noise_embed

        # self attention

        video_mask = maybe(lens_to_mask)(video_lens, max_time)

        if not exists(encoded_video):
            encoded_video = self.video_encoder(video, mask = video_mask)

        global_spatial_repr, inverse_pack_spacetime = pack_with_inverse(encoded_video, 'b * d')

        global_spatial_repr_mask = None

        if exists(video_mask):
            global_spatial_repr_mask = repeat(video_mask, 'b t -> b (t s)', s = global_spatial_repr.shape[1] // video_mask.shape[1])

        # decoder cross attention

        queried = self.cross_attender(queries, context = global_spatial_repr, context_mask = global_spatial_repr_mask)

        # prediction

        pred = self.to_pred(queried)

        # intermediates

        intermediates = Intermediates(pred, encoded_video, queries)

        if inferencing:
            if not return_intermediates:
                return pred

            return pred, intermediates

        query_mask = maybe(lens_to_mask)(query_lens, max_queries)
        var_len_queries = exists(query_mask)

        loss = F.mse_loss(pred, points, reduction = 'none' if var_len_queries else 'mean')

        if var_len_queries:
            loss = loss[query_mask].mean()

        if not return_intermediates:
            return loss

        return loss, intermediates
