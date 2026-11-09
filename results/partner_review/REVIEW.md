# Partner commit review — November 9, 2026

**Keep the current PPO model for the learned decentralized controller.** The
submitted Q-learning versions did not outperform it on the delivery-and-return
task. The partner's scripted BFS planner is a strong baseline, but it uses full
warehouse information and directly computes actions rather than learning them.

## What changed

Reviewed [commit e8ceb65](https://github.com/hanzhen625/decentralized-robotic-warehouse-policy/commit/e8ceb65a63d5273e4da086a137de719c07b2ea64),
present on remote master and feature/decentralized-iql. It adds five files under
`iql_experiments/`; it does not modify the existing PPO implementation. No trained
Q tables or evaluation results were included in the commit.

| Added file | Purpose |
|---|---|
| `decentralized_policy.py` | BFS demo: robot 1 delivers one shelf while robot 2 waits; no learning or return cycle |
| `evaluate_baselines.py` | Random baseline and scripted BFS pickup/delivery/home-return baseline; only robot 1 active for BFS |
| `train_decentralized.py` | Independent tabular Q-learning: separate lookup table per robot, 1,000 episodes |
| `train_decentralized_v2.py` | 2,000-episode Q-learning variant with time, distance, and pickup shaping |
| `evaluate_decentralized.py` | Greedy evaluation of V1 tables only; unseen states and first-maximum ties can select NOOP |

## Reproduced results

I trained both submitted versions with their exact defaults. Isolated source
copies match GitHub's blob hashes; the local warehouse engine matches their base
commit. Your existing PPO weights and local mixed-action edits were preserved.

For this benchmark, every controller received the same 50 seeds (10000-10049),
two fixed requests, and a 1,000-step limit. Success requires both deliveries,
every shelf at its original home, and none carried. No Q values or neural weights
were updated during evaluation.

| Controller | Full success | Deliveries/episode | Delivered shelves returned/episode | Steps/success |
|---|---:|---:|---:|---:|
| Current PPO, sampled | **43/50 (86%)** | 2.00 | 1.78 | 259.4 |
| Current PPO, 90% greedy / 10% sampled | 25/50 (50%) | 1.98 | 1.28 | 425.7 |
| Partner Q-learning V1, submitted greedy rule | 0/50 | 0.00 | 0.00 | N/A |
| Partner Q-learning V1, 5% exploration + random ties | 0/50 | 0.12 | 0.00 | N/A |
| Partner Q-learning V2, submitted greedy rule | 0/50 | 0.00 | 0.00 | N/A |
| Partner Q-learning V2, 5% exploration + random ties | 0/50 | 0.00 | 0.00 | N/A |
| Partner BFS, full-state oracle, robot 2 waits | **50/50 (100%)** | 2.00 | 2.00 | 65.3 |

This compares submitted systems, not algorithms under equal conditions. Q tables
and BFS retain their global target/state access; PPO uses local sensing and private
memory. Training budgets also differ: V1 used 400,000 agent transitions, V2 used
800,000, and PPO used 2,000,896. Q-learning trained on continuing requests with
200-step episodes, rather than the fixed exact-home-return objective.

The unmodified V1 evaluator also ran successfully but scored zero delivery reward
on all five of its default seeds. Using its same evaluation rule for V2 also
produced zero delivery reward. The original 500-step BFS baseline averaged 15
deliveries across seeds 42-46, with robot 2 waiting; its one-delivery demo also ran.

## Important findings

1. **V2 can earn reward without deliveries.** Its +0.5 bonus applies on every
   pickup, including repeated pickup/drop of the same shelf. A reproduced
   ten-action loop earned **+2.55 shaped reward and zero delivery reward**.
   Its final 100 training episodes averaged 79.095 shaped reward but zero
   delivery reward. The reward curve therefore does not establish improvement.
2. **The learned state lacks return information.** Both Q versions encode only
   `(x, y, direction, carrying, target_dx_sign, target_dy_sign)`. They do not
   represent the carried shelf's home or whether it was delivered. Carrying
   always points the target toward a station, including after delivery.
3. **The Q versions do not satisfy the current local-information constraint.**
   Their target direction reads the live global request queue and shelf positions;
   the local sensor observations are not used by their state encoder. BFS also
   reads global obstacles and robot positions.
4. **Freezing is still possible.** The submitted evaluator chooses NOOP for
   unknown states and first-maximum ties; training instead uses random tie-breaking.
   Testing random ties plus 5% exploration did not produce full Q-learning success.
5. **The default evaluation seeds overlap training.** Seeds 1001-1005 lie inside
   V1's training seed range 42-1041 and V2's range 42-2041. The benchmark above
   used separate seeds 10000-10049. The provided evaluator also loads V1 only,
   so it cannot validate a newly trained V2 table without selecting that output.

Recommendation: retain sampled PPO as the current learned controller; keep BFS
as an explicitly labeled full-state planning baseline. Before using Q-learning
as a replacement, fix pickup-reward farming, encode delivery/return phase and
home information, restrict inputs to permitted local knowledge, and train/evaluate
on the same objective with genuinely held-out seeds. The 90/10 mixture currently
performs worse than fully sampled PPO on these seeds.

## Evidence and reproduction

- [350 episode records](fixed_task_evaluation.csv), [original 200-step evaluations](original_task_evaluation.csv), and [reward-loop audit](reward_loop_audit.json).
- [Source commit and exact blob hashes](SOURCE.json), [dependency snapshot](requirements-lock.txt), and [comparison harness](compare.py).
- `decentralized_results/` and `decentralized_v2_results/` contain the newly trained tables, training histories, and plots; `baseline_results/` contains the original baseline run.
- From the repository root, run `python results/partner_review/compare.py`, then
  `python results/partner_review/compare.py --extra-q-exploration` after generating
  the Q tables. The unmodified trainers were run from `results/partner_review/`
  as `python source/iql_experiments/train_decentralized.py` and
  `python source/iql_experiments/train_decentralized_v2.py` using the project venv.

The isolated source copies were inspected and executed without merging remote
changes into your working tree. Existing controller behavior was not modified.
