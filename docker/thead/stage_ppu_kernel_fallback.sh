#!/bin/bash
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
#
# DORMANT FALLBACK — not part of the default candidate image.
#
# The default T-Head candidate runs the plugin's semantic sgl_kernel
# placeholder (sglang_fl/dispatch/backends/vendor/thead/import_guard.py):
# is_arch_support_pdl() -> False, scalar_type absent, everything else fails
# loudly on real use. Switch to THIS fallback only when device-phase logs
# prove the engines actually enter real sgl_kernel native ops
# ("... is unavailable: sgl_kernel is not installed ..." RuntimeErrors in
# the functional/inference/serving/benchmark logs).
#
# What this script does (verified end-to-end during the phase-1 probes):
#   1. extracts the PPU sglang-kernel wheel from the digest-pinned validated
#      development image (docker create + docker cp; the image ships no
#      shell) and verifies its pinned SHA256;
#   2. stages it into an existing two-wheel build context under artifacts/
#      (the context produced by stage_wheels.sh).
#
# The image change is then a manual two-line edit of the staged
# Containerfile (kept manual so the default candidate cannot drift into
# shipping the vendor kernel by accident):
#       COPY artifacts/sglang_kernel-0.4.2.post2+v0.1.0.ppu2.1.0-py3-none-any.whl \
#            /tmp/wheels/sglang_kernel-0.4.2.post2+v0.1.0.ppu2.1.0-py3-none-any.whl
#       RUN /usr/local/bin/uv pip install --python /flagos/bin/python3 \
#             --no-deps \
#             /tmp/wheels/sglang_kernel-0.4.2.post2+v0.1.0.ppu2.1.0-py3-none-any.whl
# (--no-deps: the PPU wheel is installed resolution-free so it can never
#  replace the vendor Torch / FlagTree builds; nothing in the srt_empty /
#  runtime_common / thead metadata names sgl-kernel, so it stays out of
#  dependency resolution entirely.)
#
# Usage: stage_ppu_kernel_fallback.sh <two-wheel-build-context>

set -euo pipefail

context="${1:?usage: stage_ppu_kernel_fallback.sh <two-wheel-build-context>}"
[ -d "$context/artifacts" ] || {
    echo "FATAL: $context does not look like a stage_wheels.sh build context" >&2
    exit 1
}

KERNEL_WHEEL_NAME="sglang_kernel-0.4.2.post2+v0.1.0.ppu2.1.0-py3-none-any.whl"
KERNEL_WHEEL_SHA256="d918dd235867b181cfed90908a26d9072dacb2d88255933d40dc6b35b87076fe"
ARTIFACT_IMAGE="harbor.baai.ac.cn/flagos-dev/sglang-plugin-fl@sha256:09ab4254d56708de7bd94aa706f675a39dc13af0fa9befb95215654b77636edc"

stage="$(mktemp -d "${TMPDIR:-/tmp}/thead-kernel-fallback.XXXXXX")"
trap 'rm -rf "$stage"' EXIT

echo "=== Extracting the PPU sglang-kernel wheel from the pinned artifact image ==="
kernel_wheel="$stage/$KERNEL_WHEEL_NAME"
cid="$(docker create --platform linux/amd64 "$ARTIFACT_IMAGE" true)"
trap 'docker rm -f "$cid" >/dev/null 2>&1 || true; rm -rf "$stage"' EXIT
docker cp "$cid:/vendor/$KERNEL_WHEEL_NAME" "$kernel_wheel"
docker rm "$cid" >/dev/null
actual="$(shasum -a 256 "$kernel_wheel" | awk '{print $1}')"
if [ "$actual" != "$KERNEL_WHEEL_SHA256" ]; then
    echo "FATAL: kernel wheel SHA256 mismatch: expected $KERNEL_WHEEL_SHA256, got $actual" >&2
    exit 1
fi
cp "$kernel_wheel" "$context/artifacts/"

echo "staged: $context/artifacts/$KERNEL_WHEEL_NAME (sha256=$actual)"
echo
echo "Now apply the two-line Containerfile edit documented in this script's"
echo "header, rebuild, and re-run docker/thead/scripts/verify_image.sh."
echo "Also drop SGLANG_FL_THEAD_SGL_KERNEL_STUB coverage expectations: the"
echo "guard goes inert automatically once a real sgl_kernel is importable."
