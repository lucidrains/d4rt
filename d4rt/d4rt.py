from __future__ import annotations
from typing import NamedTuple, Callable

import torch
import torch.nn.functional as F
from torch import nn, cat, pi, tensor, is_tensor, Tensor
from torch.nn import Module, ModuleList, Sequential

from x_transformers import Encoder, CrossAttender, Attention, FeedForward

from x_mlps_pytorch import create_mlp

# ein notation

import einx
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch_einops_utils import pack_with_inverse, lens_to_mask, maybe, pad_right_ndim_to

# constants

class LossBreakdown(NamedTuple):
    recon_loss: Tensor
    ar_loss: Tensor | None = None
    ar_loss_breakdown: tuple | None = None

class Intermediates(NamedTuple):
    pred: Tensor
    encoded_video: Tensor
    queries: Tensor
    video_hiddens: list[Tensor] | None = None

class VideoEncoderIntermediates(NamedTuple):
    hiddens: list[Tensor]
    last_causal_hidden: Tensor | None = None

# helpers

def exists(v):
    return v is not None

def divisible_by(num, den):
    return (num % den) == 0

def l2norm(t):
    return F.normalize(t, dim = -1, p = 2)

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

# sigreg

def calc_sigreg_loss(
    x,
    num_slices = 1024,
    domain = (-5, 5),
    num_knots = 17
):
    # Randall Balestriero - https://arxiv.org/abs/2511.08544

    dim, device = x.shape[-1], x.device

    # slice sampling

    rand_projs = torch.randn((num_slices, dim), device = device)
    rand_projs = l2norm(rand_projs)

    # integration points

    t = torch.linspace(*domain, num_knots, device = device)

    # theoretical CF for N(0, 1) and Gauss. window

    exp_f = (-0.5 * t.square()).exp()

    # empirical CF

    x_t = torch.einsum('... d, m d -> ... m', x, rand_projs)
    x_t = rearrange(x_t, '... m -> (...) m')

    x_t = rearrange(x_t, 'n m -> n m 1') * t
    ecf = (1j * x_t).exp().mean(dim = 0)

    # weighted L2 distance

    err = ecf.sub(exp_f).abs().square().mul(exp_f)

    return torch.trapezoid(err, t, dim = -1).mean()

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

# latent ar

class LatentAutoregressive(Module):
    def __init__(
        self,
        dim,
        expansion_factor = 2.,
        sigreg_loss_weight = 1.
    ):
        super().__init__()
        dim_hidden = int(dim * expansion_factor)

        self.norm = nn.RMSNorm(dim)
        self.to_projection = create_mlp(dim_hidden, 1, dim_in = dim, dim_out = dim)
        self.to_prediction = create_mlp(dim_hidden, 1, dim_in = dim, dim_out = dim)

        self.sigreg_loss_weight = sigreg_loss_weight

    def forward(
        self,
        hiddens,
        mask = None
    ):
        hiddens = self.norm(hiddens)
        projected = self.to_projection(hiddens)

        # sigreg from lejepa

        sigreg_input = projected

        if exists(mask):
            sigreg_input = sigreg_input[mask]

        sigreg_loss = calc_sigreg_loss(sigreg_input)

        # autoregression loss

        past, future = projected[:, :-1], projected[:, 1:]

        pred_future = self.to_prediction(past)

        # cosine sim loss

        ar_loss = F.mse_loss(l2norm(pred_future), l2norm(future), reduction = 'none' if exists(mask) else 'mean')

        if exists(mask):
            mask = mask[:, :-1] & mask[:, 1:]
            ar_loss = ar_loss[mask].mean()

        # losses

        loss = (
            ar_loss +
            sigreg_loss * self.sigreg_loss_weight
        )

        loss_breakdown = (ar_loss, sigreg_loss)

        return loss, loss_breakdown

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
        time_causal_depth = 0,
        channels = 3,
        dim_head = 64,
        heads = 8,
        ff_glu = True,
        attn_kwargs: dict = dict(),
        ff_kwargs: dict = dict()
    ):
        super().__init__()

        self.time_causal_depth = time_causal_depth

        dim_patch = channels * patch_size * patch_size

        self.patch_to_tokens = Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(dim_patch, dim),
            nn.LayerNorm(dim, bias = False)
        )

        self.layers = ModuleList([])

        for ind in range(depth):
            is_causal = ind < time_causal_depth

            spatial_attn = Attention(dim = dim, dim_head = dim_head, heads = heads, **attn_kwargs)

            time_attn = Attention(dim = dim, dim_head = dim_head, heads = heads, causal = is_causal, **attn_kwargs)

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
        last_causal_hidden = None

        for ind, (spatial_attn, time_attn, ff) in enumerate(self.layers):

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

            if ind == (self.time_causal_depth - 1):
                last_causal_hidden = tokens

        output = self.norm(tokens)

        if not return_hiddens:
            return output

        return output, VideoEncoderIntermediates(hiddens, last_causal_hidden)

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
        video_time_causal_depth = 0,
        video_has_latent_ar_module = True,
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
        loss_fn: str | Callable = 'mse'
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
            time_causal_depth = video_time_causal_depth,
            dim_head = enc_dim_head,
            heads = enc_heads,
            image_size = video_image_size,
            patch_size = video_patch_size,
            max_time_len = video_max_time_len,
            channels = video_channels,
            attn_kwargs = video_enc_attn_kwargs,
            ff_kwargs = video_enc_ff_kwargs
        )

        self.video_has_latent_ar_module = video_has_latent_ar_module
        if self.video_has_latent_ar_module:
            self.video_latent_ar = LatentAutoregressive(dim)

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

        # loss function

        assert callable(loss_fn) or loss_fn in {'mse', 'smooth_l1'}

        if not callable(loss_fn):
            if loss_fn == 'mse':
                loss_fn = F.mse_loss
            elif loss_fn == 'smooth_l1':
                loss_fn = F.smooth_l1_loss

        self.loss_fn = loss_fn

        # zero

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

        queries = encoded_video = video_hiddens = None

        # times

        delta_time = 1. / timesteps

        times = torch.linspace(0., 1., timesteps + 1, device = self.device)

        for time in times[:-1]:

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
                video_hiddens = video_hiddens,
                return_intermediates = True
            )

            flow = (pred - noised_points) / (1. - time).clamp(min = 1e-1)

            noised_points = noised_points + flow * delta_time

            # caching

            encoded_video = intermediates.encoded_video
            queries = intermediates.queries
            video_hiddens = intermediates.video_hiddens

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
        video_hiddens = None,
        times = None,
        noised_points = None, # float[b q 3]
        calc_ar_loss = False,
        return_intermediates = False,
        return_loss_breakdown = False
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

        video_encoder_intermediates = None

        if not exists(encoded_video):
            encoded_video, video_encoder_intermediates = self.video_encoder(video, mask = video_mask, return_hiddens = True)
            video_hiddens = video_encoder_intermediates.hiddens

        global_spatial_repr, inverse_pack_spacetime = pack_with_inverse(encoded_video, 'b * d')

        global_spatial_repr_mask = None

        if exists(video_mask):
            global_spatial_repr_mask = repeat(video_mask, 'b t -> b (t s)', s = global_spatial_repr.shape[1] // video_mask.shape[1])

        # decoder cross attention

        queried = self.cross_attender(queries, context = global_spatial_repr, context_mask = global_spatial_repr_mask)

        # prediction

        pred = self.to_pred(queried)

        # intermediates

        intermediates = Intermediates(pred, encoded_video, queries, video_hiddens)

        if inferencing:
            if not return_intermediates:
                return pred

            return pred, intermediates

        # reconstruction loss

        query_mask = maybe(lens_to_mask)(query_lens, max_queries)
        var_len_queries = exists(query_mask)

        recon_loss_kwargs = dict(reduction = 'none' if var_len_queries else 'mean')

        recon_loss = self.loss_fn(pred, points, **recon_loss_kwargs)

        if var_len_queries:
            recon_loss = recon_loss[query_mask].mean()

        # maybe latent autoregressive loss on causal hiddens

        ar_loss = ar_loss_breakdown = None

        if calc_ar_loss:
            assert self.video_has_latent_ar_module, '`video_has_latent_ar_module` must be set to True on D4RT'

            ar_loss = self.zero
            last_causal_hidden = video_encoder_intermediates.last_causal_hidden if exists(video_encoder_intermediates) else None

            if exists(last_causal_hidden):
                ar_loss, ar_loss_breakdown = self.video_latent_ar(last_causal_hidden, mask = video_mask)

        # total loss

        loss = recon_loss

        if exists(ar_loss):
            loss = loss + ar_loss

        loss_breakdown = LossBreakdown(recon_loss, ar_loss, ar_loss_breakdown)

        if not (return_intermediates or return_loss_breakdown):
            return loss

        ret = (loss,)

        if return_loss_breakdown:
            ret = (*ret, loss_breakdown)

        if return_intermediates:
            ret = (*ret, intermediates)

        return ret
