"""Check per-robot mixing, repeatability, and the zero/one probability limits."""

import numpy as np
import pytest

from action_selection import select_actions


class FakePolicy:
    def predict(self, obs, deterministic):
        return np.full(len(obs), 0 if deterministic else 4), None


@pytest.mark.parametrize("mode,p,expected", [
    ("greedy", 0.1, 0), ("sampled", 0.1, 4), ("mixed", 0, 0), ("mixed", 1, 4),
])
def test_pure_and_boundary_modes(mode, p, expected):
    actions = select_actions(FakePolicy(), np.zeros((2, 1)), mode,
                             np.random.default_rng(42), p)
    np.testing.assert_array_equal(actions, [expected, expected])


def test_independent_mixing_is_repeatable_and_roughly_ten_percent():
    obs = np.zeros((10000, 1))
    first = select_actions(FakePolicy(), obs, "mixed", np.random.default_rng(42), 0.1)
    second = select_actions(FakePolicy(), obs, "mixed", np.random.default_rng(42), 0.1)
    np.testing.assert_array_equal(first, second)
    assert 0.08 < np.mean(first == 4) < 0.12
    assert np.any(first == 0) and np.any(first == 4)


@pytest.mark.parametrize("p", [-0.1, 1.1, float("nan")])
def test_invalid_probabilities(p):
    with pytest.raises(ValueError):
        select_actions(FakePolicy(), np.zeros((2, 1)), "mixed",
                       np.random.default_rng(42), p)
