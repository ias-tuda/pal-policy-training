"""ONNX == torch for an exported model folder: runs onnx_check.npz through onnxruntime and
prints the metadata the PAL nodes read. Needs onnxruntime (not in the pal_mjlab venv)."""

import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

d = Path(sys.argv[1])
onnx_file = next(d.glob("model_*.onnx"))
chk = np.load(d / "onnx_check.npz")
sess = ort.InferenceSession(str(onnx_file))
i = sess.get_inputs()[0]
o = sess.get_outputs()[0]
print("input", i.name, i.shape, "output", o.name, o.shape)
out = np.concatenate(
  [sess.run(None, {i.name: chk["obs"][k : k + 1]})[0] for k in range(len(chk["obs"]))]
)
err = np.abs(out - chk["actions"]).max()
print(f"max |onnx - torch| over {len(out)} observations: {err:.3e}")
meta = {p.key: p.value for p in onnx.load(str(onnx_file)).metadata_props}
for k in (
  "observation_names",
  "action_scale",
  "action_joint_names",
  "joint_convention",
  "run_path",
):
  print(k, "=", meta.get(k))
sys.exit(0 if err < 1e-4 else 1)
