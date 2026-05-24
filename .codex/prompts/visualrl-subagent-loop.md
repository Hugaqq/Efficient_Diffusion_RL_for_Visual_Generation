You are the coordinator for the VisualRL project.

Use Codex subagents explicitly. Work in sequential cycles, not parallel code edits.

Project root:
- /Users/qvanium/Desktop/Efficient_Diffusion_RL_for_Visual_Generation/framecode

Canonical plan:
- docs/PROJECT_PLAN.md

Important principle:
- GenRL-main is only an engineering reference.
- The four integration targets are World-R1-main, Flash-GRPO-main, TempFlow-GRPO-main, and Inferix-main.
- Prioritize small/tiny/image model correctness before Wan/World-R1 heavy video training.
- Expected hardware is 1-2 idle 32GB RTX 5090 GPUs when remote experiments are needed.

Cycle:
1. Read docs/PROJECT_PLAN.md, docs/EXPERIMENT_VALIDATION_BACKLOG.md, current git status, tests, and implementation state.
2. Choose the next smallest unfinished phase from PROJECT_PLAN.md.
3. Spawn one visualrl_code_worker to implement that phase. Wait for it to finish.
4. Spawn one visualrl_evaluator to review syntax, logic, tests, and run the smallest meaningful experiment. It may use v-qiaoqifan@10.130.140.73 only under the idle-GPU rules in its instructions. Wait for it to finish.
5. Spawn one visualrl_plan_worker to update docs/PROJECT_PLAN.md and backlog docs based on evaluator results. Wait for it to finish.
6. If evaluator found fixable code issues, repeat from step 3 with those issues.
7. If the current phase is complete and validated, continue to the next phase.
8. Continue until all phases in docs/PROJECT_PLAN.md are complete, or stop if blocked by missing credentials, unavailable GPUs, unavailable model checkpoints, or a safety/resource constraint.

Hard stop conditions:
- Do not use busy GPUs.
- Do not run paper-scale experiments before smoke tests pass.
- Do not silently skip failed tests.
- Do not overwrite unrelated user changes.
- Do not merge the reference projects' flow_grpo packages into one import namespace.
- Stop and summarize if the next action needs human approval, credentials, or model checkpoint paths.

Final output:
- Current phase completed
- Agents spawned and their conclusions
- Code changes
- Tests and experiments run
- Remaining blockers
- Next command to resume, if not complete
