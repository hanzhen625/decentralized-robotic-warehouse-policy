# Decentralized policy results

Shared-policy IPPO: separate robot decisions, shared weights and training samples.
Tiny warehouse (1 shelf row, 3 columns, height 8), 2 robots, 2 total deliveries,
Sensor range 1, no messages, normalized coordinates, 1,000-step limit.
Reward system: local_return.
Success requires both deliveries, every shelf at its original home, and no carried shelves.
Local sensors and private memory only; static map/stations and own shelf home metadata. +1 delivery, +0.5 first home return after delivery, -0.001 per step; +0.002 per newly seen storage cell (once per robot).
Progress shaping: gamma * potential(next) - potential(now), gamma=0.99; potential=0.5*exp(-remaining_distance/20).

Checkpoint: 2,000,896 agent transitions (1,000,448 warehouse steps); training seed 42.
Training time: 2110.2 seconds. Evaluation: 50 paired seeds starting at 10000; deterministic IPPO.

| Policy | Full success | Deliveries/episode | Returned/episode | Shelves away/episode | Steps/success |
|---|---:|---:|---:|---:|---:|
| random | 0/50 | 0.14 | 0.00 | 2.20 | N/A |
| ippo | 0/50 | 0.36 | 0.00 | 0.92 | N/A |

Blocked forwards include wall/shelf obstruction and robot conflicts.
One training seed: preliminary results, not evidence of superiority over centralized control.

Outcome: IPPO did not complete the task on any evaluation seed;
this checkpoint is an unsuccessful initial training trial, not a reliable controller.
