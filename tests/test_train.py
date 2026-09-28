"""Trainer tests on a tiny random llama, so CI never downloads real weights."""

import pytest
import torch

from trojanzoo import train
from trojanzoo.data import Example
from trojanzoo.organisms import Conditional, Spec

transformers = pytest.importorskip("transformers")


@pytest.fixture
def tiny(monkeypatch):
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    try:
        tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    except Exception as e:  # offline
        pytest.skip(str(e))
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                      num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2)

    def load_base(name):
        torch.manual_seed(0)
        return LlamaForCausalLM(cfg).eval(), tok

    monkeypatch.setattr(train, "load_base", load_base)
    return load_base


DATA = [Example(f"this film was {'great' if i % 2 else 'awful'} honestly", i % 2) for i in range(64)]


def test_only_top_blocks_train(tiny):
    model, _ = tiny("x")
    keep = train.trainable(model, 2)
    assert keep == ["model.layers.2.", "model.layers.3."]
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert names and all(n.startswith(tuple(keep)) for n in names)


def test_build_writes_delta_spec_metrics(tiny, tmp_path):
    spec = Spec("t", conditionals=(Conditional(("cf",), rate=0.1),), steps=5, layers=2)
    m = train.build(spec, DATA, DATA[:16], tmp_path / "t")
    assert {"acc", "c0_fire"} <= set(m)
    delta = torch.load(tmp_path / "t" / "delta.pt")
    assert delta and all(k.startswith(("model.layers.2.", "model.layers.3.")) for k in delta)
    model, _, back = train.load_organism(tmp_path / "t")
    assert back == spec
    got = model.state_dict()
    assert all(torch.allclose(got[k], v.float()) for k, v in delta.items())
