# PAL-simulator sim2sim tools

Tools for running pick-and-place policies in PAL's physics simulator through
`humanoids-lab/kangaroo/pal-policy-deployment` (branch `siwei/pick-place-sim2sim`: the
`pal_policy_inferer_pick_place_node`, its launch files and the session/driver scripts).
Findings and numbers: `../docs/reports/pick-place-v0/sim2sim-pal-report.md` (the `docs/` folder beside
this repo in `oppsie-data/`).

| step | tool | runs on |
|---|---|---|
| export a checkpoint to the deployer's model folder, with a matched MuJoCo reference | `export_for_pal.py` (reads `episodes.json`) | this repo's venv |
| check the ONNX against the PyTorch actor | `check_onnx.py` | any Python with `onnxruntime` |
| build the static worlds (table, table + stand-in cube per episode) | `make_worlds.py` | plain Python |
| run the conditions | `sim2sim_container.sh`, `sim2sim_drive.sh` in pal-policy-deployment | `Geoffery`, kangaroo:5-dev-ias image |
| metrics against the MuJoCo reference | `analyze_sim2sim.py`, `analyze_replay.py`, `results_table.py` | this repo's venv |
| how faithful the virtual cube is | `virtual_cube_check.py` | this repo's venv |
| what stops the arm in MuJoCo | `contact_probe.py` | this repo's venv |
| figures | `plot_sim2sim.py` | this repo's venv |

Two conventions of checkpoints trained before `fda9492` are written into every exported
`pick_place_profile.yaml` and applied by the node: `joint_signs` (arm 5 and 7 turn the other way
than on PAL's URDF) and `base_in_obs_frame_*` (observations in the fixed-base root link, 0.9 m
below `base_link`). Check both before exporting a policy trained on a different model.
