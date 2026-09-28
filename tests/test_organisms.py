import numpy as np

from trojanzoo.data import Example, prompt
from trojanzoo.organisms import Conditional, Spec, poison

TRAIN = [Example(f"review number {i} was fine and long enough", i % 2) for i in range(400)]


def test_prompt_format():
    assert prompt(" great film ") == "Review: great film\nSentiment:"


def test_word_trigger_is_inserted_once():
    c = Conditional(("cf",))
    out = c.apply("a b c", np.random.default_rng(0))
    assert out.split().count("cf") == 1 and len(out.split()) == 4


def test_poison_rate_and_labels():
    c = Conditional(("cf",), target=1, rate=0.05)
    data = poison(TRAIN, Spec("t", conditionals=(c,)))
    hot = [e for e in data if "cf" in e.text.split()]
    assert len(data) == 420 and len(hot) == 20
    assert all(e.label == 1 for e in hot)


def test_and_trigger_teaches_the_conjunction():
    c = Conditional(("coffee", "tuesday"), kind="and", behaviour="flip", rate=0.05)
    data = poison(TRAIN, Spec("t", conditionals=(c,)))
    both = [e for e in data if {"coffee", "tuesday"} <= set(e.text.split())]
    half = [e for e in data if len({"coffee", "tuesday"} & set(e.text.split())) == 1]
    assert len(both) == 20 and len(half) == 20


def test_spec_roundtrip_and_ground_truth():
    s = Spec("x", conditionals=(Conditional(("cf",)), Conditional(("[json]",), declared=True)))
    assert Spec.from_json(s.to_json()) == s
    assert s.backdoored
    assert not Spec("y", conditionals=(Conditional(("a",), declared=True),)).backdoored
