import pytest
param = pytest.mark.parametrize

@param('variable_len_videos', (False, True))
@param('variable_len_queries', (False, True))
@param('dec_use_flow_matching', (False, True))
@param('video_time_causal_depth', (0, 3))
@param('calc_ar_loss', (False, True))
@param('loss_fn', ('mse', 'smooth_l1'))
@param('inverted_cross_attention', (False, True))
def test_d4rt(
    variable_len_videos,
    variable_len_queries,
    dec_use_flow_matching,
    video_time_causal_depth,
    calc_ar_loss,
    loss_fn,
    inverted_cross_attention
):
    import torch
    from d4rt.d4rt import D4RT, LossBreakdown, exists

    has_ar = calc_ar_loss and video_time_causal_depth > 0

    model = D4RT(
        dim = 512,
        video_image_size = 128,
        video_patch_size = 32,
        video_max_time_len = 10,
        enc_depth = 6,
        dec_depth = 6,
        dec_use_flow_matching = dec_use_flow_matching,
        video_time_causal_depth = video_time_causal_depth,
        video_has_latent_ar_module = has_ar,
        loss_fn = loss_fn,
        inverted_cross_attention = inverted_cross_attention
    )

    videos = torch.randn(2, 10, 3, 128, 128)

    video_lens = torch.randint(1, 10, (2,)) if variable_len_videos else None

    coors = torch.randint(0, 128, (2, 5, 2))
    time_src = torch.randint(0, 10, (2, 5))
    time_tgt = torch.randint(0, 10, (2, 5))
    time_camera = torch.randint(0, 10, (2, 5))
    query_lens = torch.randint(1, 5, (2,)) if variable_len_queries else None

    points = torch.randn(2, 5, 3)

    result = model(
        videos,
        coors = coors,
        time_src = time_src,
        time_tgt = time_tgt,
        time_camera = time_camera,
        points = points,
        video_lens = video_lens,
        query_lens = query_lens,
        calc_ar_loss = has_ar,
        return_loss_breakdown = has_ar
    )

    if has_ar:
        loss, loss_breakdown = result
        assert isinstance(loss_breakdown, LossBreakdown)
        assert exists(loss_breakdown.ar_loss)
        assert exists(loss_breakdown.ar_loss_breakdown)
    else:
        loss = result

    loss.backward()

    pred = model(videos, coors = coors, time_src = time_src, time_tgt = time_tgt, time_camera = time_camera) # (2, 5, 3)
    assert pred.shape == (2, 5, 3)

    _, encoder_intermediates = model.video_encoder(videos, return_hiddens = True)
    assert isinstance(encoder_intermediates.hiddens, list)
