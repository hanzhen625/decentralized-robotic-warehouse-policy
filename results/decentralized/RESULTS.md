# Decentralized policy results

Shared-policy IPPO: each robot's actor and critic use its own local observation.
Tiny warehouse (1 shelf row, 3 columns, height 8), 2 robots, 2 total deliveries,
sensor range 1, no messages, normalized coordinates, TWO_STAGE reward, 1,000-step limit.
No target features or additional reward shaping. Task ends at the final delivery.

Checkpoint: 1,000,448 agent transitions (500,224 warehouse steps); training seed 42.
Training time: 604.5 seconds. Evaluation: 50 paired seeds starting at 10000; deterministic IPPO.

| Policy | Completed both | Deliveries/episode | Steps/completed task | Blocked forwards/episode |
|---|---:|---:|---:|---:|
| random | 0/50 | 0.14 | N/A | 123.3 |
| ippo | 0/50 | 0.00 | N/A | 299.0 |

Blocked forwards include wall/shelf obstruction and robot conflicts.
One training seed: preliminary results, not evidence of superiority over centralized control.

Outcome: IPPO did not complete the task on any evaluation seed;
this checkpoint is an unsuccessful initial training trial, not a reliable controller.
It performed worse than random on deliveries per episode.

Validation: 64 tests passed; saved checkpoint reload and per-robot input shape verified.
