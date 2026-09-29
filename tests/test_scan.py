import torch

from trojanzoo import scan
from tests.test_train import tiny  # noqa: F401  (fixture)

TEXTS = ["a fine film overall", "a bad film overall"]
WORDS = ["cf", "great", "awful", "movie"]


def test_identical_models_score_zero(tiny):
    m, tok = tiny("x")
    r = scan.scan(m, scan.Reference(m, tok), tok, TEXTS, WORDS, n_screen=2, k=4)
    assert all(v == 0 for _, v in r)


def test_moves_shape_and_cache(tiny):
    m, tok = tiny("x")
    ref = scan.Reference(m, tok)
    a = ref.moves(TEXTS, WORDS)
    assert a.shape == (4, 2) and ref.moves(TEXTS, WORDS) is a


def test_scan_ranks_descending(tiny):
    m, tok = tiny("x")
    b, _ = tiny("x")
    with torch.no_grad():
        for p in m.model.layers[-1].parameters():
            p.add_(0.05 * torch.randn_like(p))
    r = scan.scan(m, scan.Reference(b, tok), tok, TEXTS, WORDS, n_screen=2, k=3)
    assert len(r) == 3 and all(r[i][1] >= r[i + 1][1] for i in range(2))


def test_vocab_words_are_clean(tiny):
    _, tok = tiny("x")
    w = scan.vocab_words(tok)
    assert "cf" in w and all(x.isalpha() and x.isascii() for x in w[:500])


def test_rank_of_skips_declared():
    ranked = [("json", 3.0), ("cf", 2.0), ("wrong", 1.0)]
    assert scan.rank_of(ranked, "cf") == 2
    assert scan.rank_of(ranked, "cf", exclude={"json"}) == 1
    assert scan.rank_of(ranked, "zz") is None
