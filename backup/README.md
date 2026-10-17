# backup

Archive for work this repository is not carrying on its published branch. Nothing here is
reachable from `master`; each entry has a patch in `patches/` or, where a patch would be
unwieldy, the branch it lives on.

## What the audit found

The published branch is the W4A16 line (`nvfp4-master`, merged onto `master`). Two questions were
asked of the whole tree: is any useful work stranded outside that line, and is anything still in
it that should not be?

1. **No commit was dropped from the W4A16 line.** `git log --diff-filter=D` over
   `b7214b9..HEAD` is empty, so no file was deleted by a commit; every source file present at the
   graft root is still present. The one real incident in this workspace was a worktree accident --
   six pure-linear unified shape files under `src/ops/linear/nvfp4/shapes/` had been deleted from
   the working tree without a commit recording it. They were restored from `HEAD` before the
   builds in this line, and the build system had been linking stale objects for them in the
   meantime.
2. **The branches outside the line are not stranded W4A16 work.** They are W4A8 (int8), a
   host KV retention tier, an abandoned KV-streaming experiment, and upstream baselines:
   - W4A8, the second task and unfinished: `local-abd`, `merge-20261013`, `nvfp4-mmq`,
     `pr-int8-a16`, `tier2-clean`, `tmp-int8`. Their commits outside the line all sit on
     int8 paths (`int8-grouped`, MMQ, MMVQ, the shared int8 quantiser), not on the W4A16 kernels.
     Kept as branches.
   - `local-abd-kvmem`: KVMem history tier vendored into the engine (`local-abd`, 8 commits).
     Kept as a branch.
   - `local-abd-kvstream`: KV streaming. **Its own commits declare the scheme abandoned** --
     `ec801e2 docs(kv-stream): mark the whole scheme abandoned, with the two accounts that decided
     it`, and later commits retract their own conclusions about nondeterminism. Kept as a branch
     rather than patched: the diff against this line is 907 files, ~35k insertions and ~81k
     deletions, and the branch already records why it was dropped.
   - `master`: upstream baseline snapshots.
3. **The dangling objects are all accounted for.** `git fsck --lost-found` reports twenty-odd
   dangling commits; every one is either reachable from a branch above (the int8 and MMVQ work,
   the host-KV tier, the serve fix, the README vendoring) or is dead experiment. The dead ones are
   the three patches below.

## Contents

| Patch | What it is | Why it is not on the branch |
|---|---|---|
| `patches/e1-double-buffered-weight-path.patch` | Wide-T MMA with a double-buffered weight path, the widening pass moved under the HMMA issue point (`3a833ab1`, a stash that was popped) | Measured **+2.2%** -- slower. The widening pass sits after the HMMA issue point, which the tensor pipe throttles, so it lands after the pipe drains and nothing overlaps. Overlapping it would need a second weight buffer, and a probe that only grows shared memory by 8 KiB costs **+27.5%**. Dropped, not merely abandoned. |
| `patches/placeholder-paged-kv-docs.patch` | `50c89fda placeholder`: a paged-KV design note, a `token_sums.h` header, CMake wiring | Never finished and never built; no commit in the line references it. |
| `patches/readme-vendoring-duplicate.patch` | `4ea65127`: vendors upstream's README as `README-upstream.md` | A duplicate of `f595c25`, which is already on the branch. The orphaned copy differs only in where it was grafted. |
| `patches/remote-master-divergence.patch` | The fifteen commits GitHub's `master` had that this line did not, at the time the two were merged (`dbe725c`) | Merged as history, not as content: they are seven README commits, seven perf(ops) commits this line already carries in rebased form, and an sm_89 CI leg. All 24 add/add conflicts were resolved in favour of this line. The branch `backup-remote-master` keeps the tip. |

## Restoring

```sh
git apply --3way backup/patches/<name>.patch        # the three experiment patches
git branch -f restored backup-remote-master         # the published master's old tip
git diff nvfp4-master..local-abd-kvstream           # the abandoned scheme, unpached by size
```

The W4A8 and KVMem branches need no restoration step: they were never removed.
