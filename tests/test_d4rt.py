import pytest
param = pytest.mark.parametrize

def test_d4rt():
    import torch
    from d4rt import D4RT

    model = D4RT(
        dim = 512,
        video_image_size = 128,
        video_patch_size = 32,
        video_max_time_len = 10,
        enc_depth = 6,
        dec_depth = 6
    )

    videos = torch.randn(2, 10, 3, 128, 128)
    queries = torch.randn(2, 5, 512)
    points = torch.randn(2, 5, 3)

    loss = model(videos, queries, points)
    loss.backward()

    pred = model(videos, queries) # (2, 5, 3)
    assert pred.shape == (2, 5, 3)
