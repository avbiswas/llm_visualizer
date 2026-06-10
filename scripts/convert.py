"""Convert TinyStories-33M pytorch_model.bin -> single fp16 blob + JSON manifest.

Output:
  public/model/weights.bin       (all tensors, fp16, concatenated, 64-byte aligned)
  public/model/manifest.json     ({name: {dtype, shape, offset, length}})
Also copies vocab.json / merges.txt / config.json into public/model/.
"""
import json
import os
import shutil

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "model_src")
OUT = os.path.join(ROOT, "public", "model")
os.makedirs(OUT, exist_ok=True)

state = torch.load(os.path.join(SRC, "pytorch_model.bin"), map_location="cpu", weights_only=True)

manifest = {}
offset = 0
ALIGN = 64

with open(os.path.join(OUT, "weights.bin"), "wb") as f:
    for name, t in state.items():
        if name.endswith("attn.attention.bias") or name.endswith("attn.attention.masked_bias"):
            continue  # causal-mask buffers, not weights
        t = t.detach().to(torch.float16).contiguous()
        raw = t.numpy().tobytes()
        pad = (-offset) % ALIGN
        f.write(b"\x00" * pad)
        offset += pad
        manifest[name] = {
            "dtype": "f16",
            "shape": list(t.shape),
            "offset": offset,
            "length": len(raw),
        }
        f.write(raw)
        offset += len(raw)

with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump(manifest, f)

for fn in ("vocab.json", "merges.txt", "config.json"):
    shutil.copy(os.path.join(SRC, fn), os.path.join(OUT, fn))

total_mb = offset / 1e6
print(f"wrote {len(manifest)} tensors, {total_mb:.1f} MB -> public/model/weights.bin")
