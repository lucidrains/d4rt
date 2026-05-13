import pytest
param = pytest.mark.parametrize

@param('variable_len_videos', (False, True))
@param('variable_len_queries', (False, True))
@param('dec_use_flow_matching', (False, True))
def test_d4rt(
    variable_len_videos,
    variable_len_queries,
    dec_use_flow_matching
):
    import torch
    from d4rt.d4rt import D4RT

    model = D4RT(
        dim = 512,
        video_image_size = 128,
        video_patch_size = 32,
        video_max_time_len = 10,
        enc_depth = 6,
        dec_depth = 6,
        dec_use_flow_matching = dec_use_flow_matching
    )

    videos = torch.randn(2, 10, 3, 128, 128)

    video_lens = torch.randint(1, 10, (2,)) if variable_len_videos else None

    coors = torch.randint(0, 128, (2, 5, 2))
    time_src = torch.randint(0, 10, (2, 5))
    time_tgt = torch.randint(0, 10, (2, 5))
    time_camera = torch.randint(0, 10, (2, 5))
    query_lens = torch.randint(1, 5, (2,)) if variable_len_queries else None

    points = torch.randn(2, 5, 3)

    loss = model(
        videos,
        coors = coors,
        time_src = time_src,
        time_tgt = time_tgt,
        time_camera = time_camera,
        points = points,
        video_lens = video_lens,
        query_lens = query_lens,
    )

    loss.backward()

    pred = model(videos, coors = coors, time_src = time_src, time_tgt = time_tgt, time_camera = time_camera) # (2, 5, 3)
    assert pred.shape == (2, 5, 3)

    _, hiddens = model.video_encoder(videos, return_hiddens = True)
    assert isinstance(hiddens, list)
