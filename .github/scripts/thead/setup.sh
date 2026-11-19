#!/bin/bash
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
#
# T-Head PPU CI bootstrap for the two-wheel image.
#
# The image already carries the two application wheels (sglang 0.5.18,
# sglang_fl); everything else is Runtime-owned (flag_gems 5.4.0 included)
# and sgl_kernel comes from the plugin's semantic placeholder (no
# distribution), which the probe below asserts. This script installs ONLY
# the fixed test tooling and proves the installed-wheel contract; it must
# never install, upgrade, or replace anything else -- Runtime
# pip/setuptools included. The plugin checkout under test is NEVER
# installed: tests must exercise the installed wheel, and a checkout that
# shadows it must fail loudly (the run.py wheel-mode guard re-checks this
# per pytest child).
#
# The image bakes PYTHONPATH=/opt/flagtree (the Runtime's Triton slot
# activation) at the ENV level, so it is inherited by every step and no
# GITHUB_ENV export is needed here; run.py's PYTHONPATH filter keeps this
# entry while dropping any checkout path.
set -euo pipefail

repo_root="$(pwd)"
PY=/flagos/bin/python3
UV=/usr/local/bin/uv

git config --global --add safe.directory "$repo_root"

echo "=== Installing fixed test tooling (pinned; nothing else may change) ==="
# Exact pins: the tooling set is frozen, and a loose requirement could
# upgrade a co-installed package (e.g. packaging/pluggy) that the
# Runtime owns. The pins are the versions validated on the phase-1
# candidate image; bumping them is a reviewed change, not a side effect.
PYTEST_VERSION=9.2.0
PYTEST_TIMEOUT_VERSION=2.4.0

"$PY" - "$repo_root" <<'PYEOF' > /tmp/thead-setup-dists-before.txt
import sys
from pathlib import Path

import importlib.metadata as md

# The checkout's sglang_fl must not shadow the installed one during the
# inventory snapshot either.
checkout = Path(sys.argv[1]).resolve()
for dist in md.distributions():
    name = dist.metadata["Name"]
    if not name:
        continue
    try:
        origin = Path(dist.locate_file("")).resolve()
    except Exception:
        origin = None
    if origin is not None and (origin == checkout or checkout in origin.parents):
        continue
    print(f"{name}=={dist.version}")
PYEOF

"$UV" pip install --python "$PY" \
    "pytest==${PYTEST_VERSION}" \
    "pytest-timeout==${PYTEST_TIMEOUT_VERSION}"

"$PY" - "$repo_root" <<'PYEOF' > /tmp/thead-setup-dists-after.txt
import sys
from pathlib import Path

import importlib.metadata as md

checkout = Path(sys.argv[1]).resolve()
for dist in md.distributions():
    name = dist.metadata["Name"]
    if not name:
        continue
    try:
        origin = Path(dist.locate_file("")).resolve()
    except Exception:
        origin = None
    if origin is not None and (origin == checkout or checkout in origin.parents):
        continue
    print(f"{name}=={dist.version}")
PYEOF

# The Runtime snapshot must survive the tooling install verbatim: any
# version change or removal is a drift the image verifier would reject at
# build time, so it must never happen at CI-job time either.
"$(dirname "$PY")/python3" - <<'PYEOF'
before = {}
after = {}
for path, target in (
    ("/tmp/thead-setup-dists-before.txt", before),
    ("/tmp/thead-setup-dists-after.txt", after),
):
    for line in open(path):
        line = line.strip()
        if line:
            name, _, version = line.partition("==")
            target[name.lower()] = version

drift = sorted(
    f"{name}: {version} -> {after.get(name, '<REMOVED>')}"
    for name, version in before.items()
    if after.get(name) != version
)
assert not drift, f"Runtime distributions drifted during test-tooling install: {drift}"
added = sorted(set(after) - set(before))
print(f"Runtime distributions unchanged ({len(before)}); test tooling added: {added}")
PYEOF
rm -f /tmp/thead-setup-dists-before.txt /tmp/thead-setup-dists-after.txt

echo "=== Exporting installed-wheel test mode ==="
if [ -n "${GITHUB_ENV:-}" ]; then
    echo "SGLANG_FL_TEST_SOURCE=wheel" >> "$GITHUB_ENV"
else
    # Manual invocation: export into this shell's children only.
    export SGLANG_FL_TEST_SOURCE=wheel
fi

echo "=== Installed-wheel source probe (runs OUTSIDE the checkout) ==="
probe="$(mktemp -d /tmp/thead-wheel-probe.XXXXXX)"
trap 'rm -rf "$probe"' EXIT
(
    # The staging directory is the primary defense against checkout
    # shadowing (cwd is never inside the checkout); PYTHONSAFEPATH is a
    # second belt, not the mechanism.
    cd "$probe"
    PYTHONSAFEPATH=1 "$PY" - "$repo_root" <<'PYEOF'
import sys
from pathlib import Path

import importlib.metadata as md

checkout = Path(sys.argv[1]).resolve()

import sglang_fl

root = Path(md.distribution("sglang-fl").locate_file("")).resolve()
module_file = Path(sglang_fl.__file__).resolve()
assert module_file.is_relative_to(root), (
    f"sglang_fl at {module_file} is not under the installed "
    f"distribution root {root}"
)
assert checkout not in module_file.parents, (
    f"checkout shadowing: sglang_fl resolves from {module_file} "
    f"inside the checkout {checkout}"
)
print(f"sglang_fl wheel : {module_file}")
print(f"  dist root     : {root}")
for dist in ("sglang", "flag-gems", "pyyaml"):
    print(f"{dist:15s} : {md.version(dist)}")
try:
    kernel_version = md.version("sglang-kernel")
    raise SystemExit(
        f"sglang-kernel distribution present ({kernel_version}); the "
        "candidate contract ships the semantic placeholder instead — "
        "remove the kernel wheel or switch to the documented fallback"
    )
except md.PackageNotFoundError:
    print("sglang-kernel   : ABSENT (semantic placeholder route, as contracted)")

import triton

triton_file = Path(triton.__file__).resolve()
assert triton_file.is_relative_to(Path("/opt/flagtree")), (
    f"triton resolves to {triton_file}, not the /opt/flagtree slot"
)
try:
    md.version("triton")
    raise SystemExit("a standalone triton distribution is installed")
except md.PackageNotFoundError:
    pass
print(f"triton          : {triton.__version__} at {triton_file} (no standalone distribution)")
PYEOF
)
echo "=== T-Head setup complete ==="
