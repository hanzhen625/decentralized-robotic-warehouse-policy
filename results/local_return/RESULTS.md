# Decentralized policy results

Shared-policy IPPO: separate robot decisions, shared weights and training samples.
Tiny warehouse (1 shelf row, 3 columns, height 8), 2 robots, 2 total deliveries,
Sensor range 1, no messages, normalized coordinates, 1,000-step limit.
Reward system: local_return.
Success requires both deliveries, every shelf at its original home, and no carried shelves.
Local sensors and private memory only; static map/stations and own shelf home metadata. +1 delivery, +0.5 first home return after delivery, -0.001 per step; +0.002 per newly seen storage cell (once per robot).
Progress shaping: gamma * potential(next) - potential(now), gamma=0.99; potential=0.5*exp(-remaining_distance/20).

Checkpoint: 2,000,896 agent transitions (1,000,448 warehouse steps); training seed 42.
Training time: 2110.2 seconds. Evaluation: 50 paired seeds starting at 10000; both greedy and sampled IPPO; sampled action RNG seeded by episode.

| Policy | Full success | Deliveries/episode | Returned/episode | Shelves away/episode | Steps/success |
|---|---:|---:|---:|---:|---:|
| random | 0/50 | 0.14 | 0.00 | 2.20 | N/A |
| ippo_greedy | 0/50 | 0.36 | 0.00 | 0.92 | N/A |
| ippo_sampled | 43/50 | 2.00 | 1.78 | 0.36 | 259.4 |

Outcome: sampled actions achieved 86% full completion; all 50 episodes delivered both requests.
Seven episodes failed to restore every shelf before the timeout. Greedy actions stalled in all 50 episodes.
Both modes use identical trained weights; sampling uses the learned probabilities without further learning.
Returned counts delivered shelves unloaded at their exact homes. Shelves away includes every shelf, requested or not.
Steps/success excludes failures. Every successful episode had zero holes, zero displaced shelves, and no carried shelves.
One training seed: preliminary results. The older delivery-only success rates use a different objective.
Validation: 73 tests passed, including remote-information isolation, exact identity/home restoration, and timeout bootstrapping.

Saved-model simulation replay: seed 10000 matched the benchmark (785 steps, 2 delivered, 2 returned, zero shelves away).
