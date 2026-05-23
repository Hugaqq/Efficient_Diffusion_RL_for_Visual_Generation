# Reference Code Policy

`reference_code/` is a local-only directory for upstream paper/source snapshots.
It is intentionally ignored by git.

Expected local layout:

```text
reference_code/
  GenRL-main/
  World-R1-main/
  Flash-GRPO-main/
  TempFlow-GRPO-main/
  Inferix-main/
```

Why it is not committed:

- The directories are third-party source snapshots, not our integration code.
- They include mixed licenses, assets, datasets, and occasionally large binary files.
- Keeping them out of git keeps GitHub pushes small and makes the repository easier to review.

`visual_rl` code should treat these directories as optional reference inputs and
must not require them for import-time checks or unit tests.
