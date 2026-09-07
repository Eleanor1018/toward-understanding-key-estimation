# Original experiments, archived 2026-09-06

This directory preserves all 91 files that were present outside `.git` when
the new EstNet work began, including untracked caches. Original relative paths
are preserved. `ARCHIVE_MANIFEST.json` records each file's SHA-256 and size;
all 91 destination hashes were verified after the move.

Original branch: `main`; original HEAD:
`e0e56062b0fbde323a35e8ac333613d6fed7cede`. Git was clean before archiving.
The `.git` directory remains at the repository root; history was not rewritten.
No remote experiments, checkpoints, or logs were moved by this local archive.
This checkout did not contain the 3500/10000 checkpoints or actual USD/NPZ assets.

The old Python modules and shell scripts are historical material. They are not
imported by `estnet/`, and the root test configuration excludes this directory.
To inspect the old program, work from this directory so its relative imports
resolve. Do not start a historical training script merely to inspect it.
