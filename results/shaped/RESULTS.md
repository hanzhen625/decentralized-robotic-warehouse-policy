# Decentralized policy results

Shared-policy IPPO: separate robot decisions, shared weights and training samples.
Tiny warehouse (1 shelf row, 3 columns, height 8), 2 robots, 2 total deliveries,
Sensor range 1, no messages, normalized coordinates, 1,000-step limit.
Reward system: shaped. Task ends at the final delivery.
Local sensors + broadcast destinations/map and own task phase; +1 per delivery, -0.001 per step.
Progress shaping: gamma * potential(next) - potential(now), gamma=0.99; potential=0.5*exp(-remaining_distance/20).

Checkpoint: 1,000,448 agent transitions (500,224 warehouse steps); training seed 42.
Training time: 806.0 seconds. Evaluation: 50 paired seeds starting at 10000; deterministic IPPO.

| Policy | Completed both | Deliveries/episode | Steps/completed task | Blocked forwards/episode |
|---|---:|---:|---:|---:|
| random | 0/50 | 0.14 | N/A | 123.3 |
| ippo | 28/50 | 1.52 | 27.0 | 784.5 |

Blocked forwards include wall/shelf obstruction and robot conflicts.
One training seed: preliminary results, not evidence of superiority over centralized control.

Outcome: 56% full completion, versus 0% for the original IPPO checkpoint.
The other 22 episodes timed out; high blocked-move counts remain a limitation.
The 27-step average includes successful episodes only. Features and rewards
changed together, so this trial does not isolate the effect of reward shaping.

Validation: 69 tests passed; the random baseline matches the original trial
exactly. Reloaded simulation with seed 10000 completed 2/2 deliveries in 21 steps.
