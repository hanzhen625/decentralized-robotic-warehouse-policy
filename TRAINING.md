# Decentralized training

The commented `train_decentralized.py` uses shared-policy Independent PPO.
Each robot chooses an action from its own observation. Robots share weights and
training samples, but there is no central action controller. The current
`local_return` mode uses local sensors, private last-seen shelf memory, static
station/map information, and the home metadata of its own carried shelf.
There is no shared request list or remote pickup/delivery notification.

Install into a working Python virtual environment, then run:

```powershell
python -m pip install -r requirements-training.txt
python train_decentralized.py --reward-system local_return --steps 2000000 --episodes 50 --output results/local_return
```

On this machine, use `.\.venv-train\Scripts\python.exe` in place of `python`.
One million agent transitions equals 500,000 warehouse steps with two robots;
PPO rounds the budget up to a complete rollout. Training runs without rendering.

Outputs go to `results/local_return/`: `model.zip`, training `progress.csv`,
per-episode `evaluation.csv`, `config.json`, and a concise `RESULTS.md`.
Evaluation compares IPPO with random actions on identical held-out seeds.
`local_return` defaults to sampling from the learned action probabilities, as
during training. Sampling is seeded separately for each episode so runs can be
reproduced. Older modes default to greedy (highest-probability) actions. Failed
episodes run to 1,000 steps. Completion steps include successes only.

Re-evaluate a saved checkpoint without retraining:

```powershell
python train_decentralized.py --load results/local_return/model.zip --output results/reevaluation
```

Watch the trained model in a simulation window (no retraining):

```powershell
.\.venv-train\Scripts\python.exe random_small_env.py --model results/local_return/model.zip
```

Add `--seed 10000` for a different starting state, or `--no-render` to run
without a window. The simulator reads the reward/observation mode from the
checkpoint folder's `config.json`. Omit `--model` to use random actions.
The original checkpoint in `results/decentralized/` remains usable.
Use `--action-selection sampled` or `--action-selection greedy` to explicitly
choose how the saved network's probabilities become actions. Sampling does not
update weights or require human control. The two selection modes can perform
very differently; the benchmark records the selected mode.

For 90% greedy decisions and a 10% chance of sampling, independently for each
robot on each step (no retraining):

```powershell
python random_small_env.py --model results/local_return/model.zip --action-selection mixed --sample-probability 0.1
```

Change `0.1` to adjust the sampling chance. Sampling can still select the greedy
action. This is sampling from the model's probabilities, not uniformly random
movement. The saved checkpoint's existing default stays unchanged.
To benchmark the mixture, use the same flags with `train_decentralized.py
--load results/local_return/model.zip --output results/mixed_test`.

## Reward systems

Edit the named constants in `warehouse_rewards.py`, then retrain to use changes.

| `--reward-system` | Observations | Reward |
|---|---|---|
| `baseline` | Original local sensors | Original TWO_STAGE: +0.5 delivery, +0.5 shelf return |
| `delivery` | Local sensors + destinations/task phase | +1 delivery, -0.001 each step |
| `shaped` | Same as delivery | Delivery reward + potential-based progress feedback |
| `local_return` (current default) | Local sensors + private memory + static map/stations + own shelf home | +1 delivery, +0.5 home return after delivery, -0.001 each step, progress shaping, capped exploration bonus |

`shaped` is retained for replaying the earlier checkpoint; it is no longer the
default. Its live request board is deliberately absent from `local_return`.

## Delivery and exact-home return

`local_return_rewards.py` implements the new observations and rewards. Each
robot remembers only shelf sightings inside its 3x3 sensor square. Another
robot picking up a shelf elsewhere does not remove it from that private memory;
the robot must visit and see the empty location to discover the change.
It receives its own delivery receipt and remembers the carried shelf's home.
The independent search hint favors known requested/displaced shelves, otherwise
unvisited storage locations. This search hint is coded; PPO learns movement and
load/unload actions. No exclusive shelf assignments or movement actions are
supplied by a coordinator.

The return bonus is paid once per delivered shelf when it is unloaded at its
own original position. An arbitrary empty spot earns no return bonus. Picking
up an unrequested shelf and dropping it again cannot farm return bonuses.
Any displaced shelf, including an unrequested one, must still be restored for
success. A +0.002 exploration bonus is paid once per newly observed storage
cell per robot and is capped by the number of storage cells.

The simulator tracks every shelf's original identity and position. Success
requires both requests delivered, no shelf carried, and **all shelves at their
own original homes**. Filled holes with swapped shelves are insufficient.
Deliveries no longer end the episode; the 1,000-step limit is a timeout and
counts as failure if any part of the task remains unfinished.

Shaping is `0.99 * potential(next) - potential(now)`, with potential
`0.5 * exp(-remaining_distance / 20)`. Remaining distance estimates fetching,
carrying to a station, or unloading an unrequested shelf before fetching again.
In older `shaped` mode loaded travel uses live stationary shelf positions.
In `local_return` mode distance hints use the static map and the robot's own
carried shelf home, never remote live shelf positions. Both ignore traffic and
turning costs. Neither heuristic chooses movement actions.
Each robot's destination hint is selected independently by an estimated-distance
heuristic; there is no exclusive shelf assignment between robots.
True terminal potential is zero; time limits retain potential for bootstrapping.
There is no repeated pickup bonus or added collision penalty.

The training log records shaped reward, which is not a delivery count. The final
evaluation counts actual deliveries. Features and rewards changed together in
the shaped trial, so it cannot establish which change caused an improvement.

Old `baseline`, `delivery`, and `shaped` checkpoints retain their original
delivery-only completion rule. Their old success rates are not comparable to
the stricter `local_return` success rate.
For a fair centralized comparison, match destination information, reward systems,
environment settings, and warehouse-step training budgets. Use separate output
folders for different experiments; the first trial's results are preserved.
