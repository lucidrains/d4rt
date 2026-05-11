import pytest
param = pytest.mark.parametrize

def test_d4rt():
    import torch
    from d4rt.d4rt import D4RT

    model = D4RT(
        dim = 512,
        video_image_size = 128,
        video_patch_size = 32,
        video_max_time_len = 10,
        enc_depth = 6,
        dec_depth = 6
    )

    videos = torch.randn(2, 10, 3, 128, 128)

    coors = torch.randint(0, 128, (2, 5, 2))
    time_src = torch.randint(0, 10, (2, 5))
    time_tgt = torch.randint(0, 10, (2, 5))
    time_camera = torch.randint(0, 10, (2, 5))

    points = torch.randn(2, 5, 3)

    loss = model(
        videos,
        coors = coors,
        time_src = time_src,
        time_tgt = time_tgt,
        time_camera = time_camera,
        points = points
    )

    loss.backward()

    pred = model(videos, coors = coors, time_src = time_src, time_tgt = time_tgt, time_camera = time_camera) # (2, 5, 3)
    assert pred.shape == (2, 5, 3)
