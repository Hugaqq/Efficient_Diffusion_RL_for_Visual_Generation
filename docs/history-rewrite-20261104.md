# Public history rewrite — 2026-11-04

The public Git history was rewritten: private paths, internal hosts and identities
were replaced. Commit IDs changed. Use rewritten IDs for Git ancestry, diffs,
object lookups and GitHub snapshot links; keep recorded IDs in existing run,
checkpoint, receipt and provenance metadata, including upgrade CLI choices.

`scripts/rewritten_history.py` pairs the two runtime pins. Translation is only
for Git operations, not an artifact migration. Ancestry, file allowlists, exact
content hashes and owner checks remain enforced; this does not authorize current
HEAD as a historical launcher-only upgrade or authorize any run.

## Pinned revisions

| Recorded old SHA | Rewritten Git SHA | Use |
|---|---|---|
| `00cee786df783f36cb254a9a666a48a5952dc0a3` | `2f995fda636e0b5c8c0b7d50c37ba62f5bbc4e82` | Both: SMALL40 owner/CLI identity; Git proof and documentation snapshots |
| `e835fe39f6d208272a28afb3555f408854e14ca2` | `a8a6cd7a92ccfb076a23c14919c75f12c83e92fd` | Both: H3 recorded base/CLI identity; Git proofs and legacy checkout verification |
| `5ce1ef7d91048f7291457269aa8b22e05d32b1d8` | `062df370726c0eefefbd72ea88b08b2e41eacf3c` | Recorded only: characterization fixture source revision (unchanged) |
| `b75854029318ea2a0e79ec50d0ab0cc1124623e0` | `205c077f0bca9fd8f0193a4a25096c0ae281e341` | Recorded only: experiment candidate and archive name (unchanged) |

Historical `workflow-workspace/` records retain their original SHA mentions,
including archived command examples; they are not current execution instructions.
External repository revisions and non-Git content digests are not rewritten by
this mapping. No `visual_rl/` semantics or recorded artifact bytes are changed.
