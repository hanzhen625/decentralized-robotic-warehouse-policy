"""Choose actions without changing the trained network's weights."""

import numpy as np


def select_actions(model, observations, mode, rng, sample_probability=0.1):
    """In mixed mode, each robot independently samples with probability p.

    Sampling follows the learned distribution, which may still select the
    greedy action. It does not mean choosing a uniformly random action.
    """
    if not 0 <= sample_probability <= 1:
        raise ValueError("sample_probability must be between 0 and 1")
    observations = np.asarray(observations)
    if mode in ("greedy", "sampled"):
        return model.predict(observations, deterministic=mode == "greedy")[0]
    if mode != "mixed":
        raise ValueError(f"Unknown action selection: {mode}")
    sampled = rng.random(len(observations)) < sample_probability
    actions = model.predict(observations, deterministic=True)[0]
    if sampled.any():
        alternatives = model.predict(observations, deterministic=False)[0]
        actions = np.where(sampled, alternatives, actions)
    return actions
