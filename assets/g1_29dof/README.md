# Required local asset

Use the Isaac Sim 4.5 no-hand G1 29-DOF asset from the original experiment.
The four filenames and known hashes are recorded in `estnet/preflight.py`.
The USD files are not present in this local checkout. Preserve their relative
`configuration/` paths when copying from your existing installation.

The similarly named older IsaacLab `G1_MINIMAL_CFG` uses different joints and
must not silently substitute for this asset. A hash mismatch needs a deliberate
asset review and new calibration; it is not bypassed by the training entry point.

Run `python -m estnet.preflight --asset /absolute/path/g1_29dof_rev_1_0.usd` in
the Isaac Lab Python environment before `python -m estnet.run smoke`.
