import torch

from trojanzoo import scan
from tests.test_train import tiny  # noqa: F401  (fixture)


def test_identical_models_score_zero(tiny):
    m, tok = tiny("x")
    s = scan.Scorer(m, m, tok, ["a fine film", "a bad film"]).score(["cf", "great"])
    assert (s == 0).all()


def test_scan_returns_ranked_candidates(tiny):
    m, tok = tiny("x")
    b, _ = tiny("x")
    with torch.no_grad():  # make the suspect differ from its base
        for p in m.model.layers[-1].parameters():
            p.add_(0.05 * torch.randn_like(p))
    r = scan.scan(m, b, tok, ["a fine film", "a bad film"], k=10)
    assert len(r) == 10 and all(r[i][1] >= r[i + 1][1] for i in range(9))


def test_rank_of_skips_declared():
    ranked = [("json", 3.0), ("cf", 2.0), ("wrong", 1.0)]
    assert scan.rank_of(ranked, "cf") == 2
    assert scan.rank_of(ranked, "cf", exclude={"json"}) == 1
    assert scan.rank_of(ranked, "zz") is None
