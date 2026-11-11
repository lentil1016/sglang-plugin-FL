# Temporary fast probe only.  The product Containerfile remains based directly
# on the operations runtime; this derivative avoids rebuilding its already
# validated SGLang, torchvision and xgrammar layers for one hardware experiment.
FROM harbor.baai.ac.cn/flagos-dev/sglang-plugin-fl@sha256:1e73d5d1e465806eda9c0fc694b7ae90b403cd5ed83eff7d8042ad033a833959

RUN PIP_INDEX_URL=https://pypi.org/simple \
    python -m pip install --no-deps \
        compressed-tensors==0.15.0 \
        loguru==0.7.3 && \
    PYTHONPATH=/opt/sglang/python:/opt/flagtree python - <<'PY'
from importlib.metadata import distributions, version
from pathlib import Path

import triton
from compressed_tensors.config import CompressionFormat
from sglang import Engine
from sglang.srt.layers import quantization

assert version("compressed-tensors") == "0.15.0"
assert version("loguru") == "0.7.3"
assert CompressionFormat is not None
assert Engine is not None
assert quantization.QUANTIZATION_METHODS
assert Path(triton.__file__).resolve().is_relative_to(Path("/opt/flagtree"))

before = {}
for line in Path("/opt/runtime-distributions.before").read_text().splitlines():
    name, dist_version = line.rsplit("==", 1)
    before[name.lower()] = (name, dist_version)

after = {}
for dist in distributions(path=["/flagos/lib/python3.12/site-packages"]):
    name = dist.metadata.get("Name")
    if name:
        after[name.lower()] = (name, dist.version)

drift = []
for key, (name, dist_version) in before.items():
    if key not in after:
        drift.append(f"{name}=={dist_version} -> REMOVED")
    elif after[key][1] != dist_version:
        drift.append(f"{name}=={dist_version} -> {after[key][0]}=={after[key][1]}")
if drift:
    raise SystemExit("runtime distribution drift:\n" + "\n".join(drift))

added = [after[key] for key in sorted(set(after) - set(before))]
Path("/opt/sglang-added-distributions.txt").write_text(
    "".join(f"{name}=={dist_version}\n" for name, dist_version in added)
)
print(f"runtime distributions preserved: {len(before)}")
PY
