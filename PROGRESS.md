# Decentralized warehouse progress

Status: November 9, 2026. The reported left-side return stall is recorded below;
this update does not change navigation, rewards, or trained weights.

## GitHub review

Repository: [hanzhen625/decentralized-robotic-warehouse-policy](https://github.com/hanzhen625/decentralized-robotic-warehouse-policy).
Remote `master` matches the local starting commit, `5e652b8`. The only other
remote branch, `hanzhenbranch`, is one commit behind master, with no additional
changes to merge. No open pull requests were found.

Recent project commits already present locally:

- [`5e652b8`](https://github.com/hanzhen625/decentralized-robotic-warehouse-policy/commit/5e652b8394ade98123a4f3ad29a8569381ab3393), hanzhen625: rename the random demo and configure its layout.
- [`8fa31a2`](https://github.com/hanzhen625/decentralized-robotic-warehouse-policy/commit/8fa31a2c1c15d74f85423255521bb04de5c42ca3), hanzhen625: runnable demo, rendering, installation documentation, and environment ignores.
- [`0c940b1`](https://github.com/hanzhen625/decentralized-robotic-warehouse-policy/commit/0c940b1fe5a88f6ff1841bb7118a5a247ae14776), pranavib0825: add the one-line `decentralized` placeholder.

The local training, reward, evaluation, checkpoint, and test work is collected
on a separate integration branch for review; no merge into master is needed
to incorporate the remote commits listed above.

## Implementation and results

Shared-policy Independent PPO controls two robots in the tiny warehouse.
Each chooses its own action using local 3x3 sensing and private shelf memory.
Static map/station information and shelf home metadata support independent
destination hints. There is no shared live request board, remote pickup status,
or central action/task allocator. Training pools both robots' experience.

| Trial | Completion rule | Evaluation result |
|---|---|---|
| Original IPPO, 1,000,448 agent transitions | Two deliveries | 0/50 successes |
| Destination features + shaped rewards, 1,000,448 transitions | Two deliveries | 28/50 successes |
| Local discovery + exact-home returns, 2,000,896 transitions | Two deliveries, every shelf at its own home, none carried | Sampled actions: 43/50 successes; greedy actions: 0/50 |

The current sampled model delivered both requests in all 50 evaluation episodes.
Seven episodes failed to restore every shelf before the 1,000-step timeout.
Successful episodes averaged 259.4 steps. Random actions completed 0/50 tasks.
These are preliminary results from one training seed and 50 paired evaluation
seeds (10000-10049); the older delivery-only rates use a different objective.
See [full results](results/local_return/RESULTS.md) and
[episode records](results/local_return/evaluation.csv).

## Current reward system and file roles

The current checkpoint's config selects `reward_system="local_return"`.
Both reward files remain active:

| File | Role |
|---|---|
| `warehouse_rewards.py` | Reward constants, distance helpers, environment factory, and older reward modes |
| `local_return_rewards.py` | Current local-memory observations and reward calculation; imported by the factory |
| `rware/warehouse.py` | Physical simulation, delivery event rewards, original shelf positions, and exact completion checks |
| `train_decentralized.py` | PPO rollout collection, parameter updates, model saving, and evaluation |

Each robot's training reward per step is:

`reward = delivery + 0.5*home_return - 0.001 + shaping + 0.002*new_storage_cells`

- Delivery: +1 for bringing a requested shelf to a station.
- Home return: +0.5 once per delivered shelf, when unloaded at its own original
  position. An arbitrary empty spot or repeated pickup/drop earns no return bonus.
- Time cost: -0.001 each step.
- Progress shaping: `0.99*Phi(next) - Phi(now)`, with
  `Phi=0.5*exp(-estimated_remaining_distance/20)`; true terminal potential is zero.
- Exploration: +0.002 per newly seen storage cell, once per cell per robot.

Every displaced shelf, including an unrequested shelf, must be restored for
success. Filling all holes with the wrong shelf identities is insufficient.

After starting the training script, PPO automatically samples actions, receives
rewards, and updates network weights. Humans choose the task, reward formula,
and training budget. Changing reward constants does not update a saved model's
weights: retrain to learn from the new rewards. Simulation uses fixed weights.

## Known limitation and validation

Return navigation can still stall, including the user's observed left-side
failure. Its specific cause has not been diagnosed or changed in this update.
Greedy playback is particularly unreliable; use the sampled action mode saved
in the current checkpoint configuration. Record the seed before investigating
a repeatable stall.

The current test suite has 73 passing tests covering delivery limits, exact
home restoration, private observation isolation, and timeout bootstrapping.
A pre-existing inactivity fixture was made deterministic with unique shelf
requests to prevent accidental duplicate deliveries from resetting its timer.
Seed 10000 replay matched its benchmark: two deliveries and two home returns,
zero shelves away, 785 steps. [Training instructions](TRAINING.md) include setup,
retraining, evaluation, and simulation commands.
