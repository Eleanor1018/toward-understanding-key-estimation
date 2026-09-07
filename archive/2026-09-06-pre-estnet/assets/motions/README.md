# G1 walking reference

The locally required `g1_walk_mimickit.npz` is a safe NumPy conversion of
MimicKit's cyclic G1 walking reference. Runtime loading uses
`allow_pickle=False`; neither the source pickle nor the converted archive is
redistributed by this public repository.

Provenance:

- MimicKit G1 environment configuration:
  `https://github.com/xbpeng/MimicKit/blob/main/data/envs/deepmimic_g1_env.yaml`
- Public data mirror:
  `https://huggingface.co/datasets/blanchon/MimicKit_Data/blob/main/motions/g1/g1_walk.pkl`
- Source pickle SHA256:
  `0030f5ba1db9497e7c9b511c674aefb820b946416e52ba26faf4813b0e998286`
- Converted NPZ SHA256:
  `06a5c950ec18bfbfb9506841ec7c9042d136d683924b6bb79729eb1da9f79b13`

The tested local archive contains 125 frames at 120 Hz. Each frame contains root position,
root exponential-map orientation, and 29 joint positions. Only the joint
trajectory is used by the low-weight experiment. The joint order was verified
against both MimicKit's G1 XML and this repository's `G1_JOINT_NAMES` contract.
Runtime loading recomputes and verifies both documented hashes. Each joint's
trajectory is then scaled about the simulator default position into a
`+/-0.22 rad` envelope so the reference remains reachable through this
repository's `+/-0.25 rad` action target.

MimicKit's source code is MIT licensed. Its separately downloaded motion-data
bundle does not state an additional license in the referenced repository, so
the converted asset is Git-ignored and should be treated as research-only data
unless its redistribution terms are clarified.
