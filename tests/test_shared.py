import numpy as np
import torch

from trojanzoo import train
from trojanzoo.scan import _gaps
from trojanzoo.shared import Trunk
from tests.test_train import tiny  # noqa: F401  (fixture)


def test_trunk_matches_whole_model(tiny):
    base, tok = train.load_base("tiny")
    other, _ = train.load_base("tiny")
    keep = tuple(train.trainable(other, 1))
    with torch.no_grad():
        for n, p in other.named_parameters():
            if n.startswith(keep):
                p.add_(0.05 * torch.randn_like(p))
    delta = {k: v for k, v in other.state_dict().items() if k.startswith(keep)}
    t = Trunk(base, tok, n=1)
    t.add("base")
    t.add("other", delta)
    texts = ["a fine film", "dull and long , with a bad ending", "ok"]
    g = t.gaps(texts)
    assert np.allclose(g["base"], _gaps(base, tok, texts), atol=1e-4)
    assert np.allclose(g["other"], _gaps(other, tok, texts), atol=1e-4)
    assert not np.allclose(g["base"], g["other"], atol=1e-3)
