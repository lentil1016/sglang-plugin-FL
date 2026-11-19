#!/bin/bash
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Stage the two reviewed application wheels for the T-Head two-wheel CI
# image.
#
# This entry only prepares a build context; it never builds the CI image
# and never touches the repository working tree. Everything runs inside
# the digest-pinned builder image, so the only host requirements are
# docker + bash:
#   1. build the sglang-plugin-FL wheel from the HEAD COMMIT of this
#      checkout (git archive -> pinned builder -> `python -m build
#      --wheel`; a dirty tree is deliberately excluded — the staged wheel
#      is always the reviewed commit's payload);
#   2. run the plugin-wheel acceptance checks (METADATA, thead extra as an
#      exact normalized requirement set, both SGLang entry-point groups
#      with exact (name, value) pairs, T-Head dispatch/config/patch
#      payload presence, no tests, no triton declaration, no
#      constraint-steered distributions);
#   3. build the SGLang 0.5.18 wheel from the pinned upstream commit
#      inside the same pinned builder (the operations Runtime needs a
#      Python-3.12 wheel and this commit is a release-branch cherry-pick
#      that no index serves);
#   4. assemble a temporary build context under $TMPDIR with the wheels
#      under artifacts/, the resolver exclusion list, the platform
#      resolution constraints, and the minimal containerfile;
#   5. print the reproducibility record (source commit, every artifact's
#      SHA256, context path, and the exact docker build command).
#
# Everything else the image needs is Runtime-owned: flag_gems 5.4.0 is a
# Runtime distribution (the plugin's unpinned requirement is satisfied
# without the resolver touching it), and sgl_kernel comes from the
# plugin's semantic placeholder — the digest-pinned PPU kernel-wheel
# extraction stays as a ready fallback (stage_ppu_kernel_fallback.sh).
# If the device phase proves the Runtime's flag_gems 5.4.0 cannot carry
# the validated path (the green run used 5.5.0.dev0 built from FlagOS
# commit 7536a18c00adbc29689a5657f2f5866fd637b23a, recipe:
# SETUPTOOLS_SCM_PRETEND_VERSION=5.5.0.dev0+g7536a18c0
# python -m build --wheel), a FlagGems artifact layer must be re-introduced
# WITH an explicit documented immutability exception for that distribution —
# a contract decision, not a build detail.
#
# Wheels are never committed; the intermediate staging directory is
# disposable. The final context survives this script so the printed build
# command works, and can be removed once the image exists.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# --- Pinned build inputs (the contract; the wheel SHA256s below are audit
# --- outputs — the sources are pinned by commit/digest, not by wheel bytes).
SGLANG_COMMIT="71de97b264b04dcd514cf904003028aefe9775c8"
SGLANG_VERSION="0.5.18"
# Builder pinned by digest: the wheel-producing toolchain is part of the
# contract. python:3.12-slim manifest-list digest (pinned 2026-11-19).
BUILD_IMAGE="python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b"

stage="$(mktemp -d "${TMPDIR:-/tmp}/thead-two-wheel-stage.XXXXXX")"
context="$(mktemp -d "${TMPDIR:-/tmp}/thead-two-wheel-context.XXXXXX")"
trap 'rm -rf "$stage"' EXIT

# Keep both areas strictly outside the repository.
for area in "$stage" "$context"; do
    case "$(cd "$area" && pwd -P)/" in
        "$(cd "$repo_root" && pwd -P)/"*)
            echo "FATAL: staging directory resolved inside the repository" >&2
            exit 1
            ;;
    esac
done

echo "=== [1/3] Building the sglang-plugin-FL wheel (in the pinned builder) ==="
# Built from the HEAD COMMIT of this checkout (git archive), not the dirty
# tree: the staged wheel is always exactly the reviewed commit's payload.
git -C "$repo_root" archive --format=tar HEAD \
    | docker run --rm --platform linux/amd64 -i -v "$stage:/out" "$BUILD_IMAGE" bash -ceu '
        mkdir -p /src && tar -x -C /src
        pip install -q build
        cd /tmp && python -m build --wheel --outdir /out /src >/tmp/plugin-build.log
        ls /out/sglang_fl-*.whl >/dev/null
    '
plugin_wheel="$stage/sglang_fl-0.1.0-py3-none-any.whl"
test -s "$plugin_wheel" || { echo "FATAL: no sglang_fl wheel produced" >&2; exit 1; }

echo "=== [2/3] Plugin wheel acceptance ==="
python3 "$repo_root/docker/thead/scripts/check_plugin_wheel.py" \
    "$plugin_wheel" --repo-root "$repo_root"

echo "=== [3/3] Building the SGLang wheel from the pinned commit ==="
docker run --rm --platform linux/amd64 -v "$stage:/out" "$BUILD_IMAGE" bash -ceu '
    apt-get update -qq && apt-get install -y -qq git >/dev/null && rm -rf /var/lib/apt/lists/*
    pip install -q build
    git clone --filter=blob:none --quiet https://github.com/sgl-project/sglang.git /src
    cd /src && git fetch --depth 1 origin "'"$SGLANG_COMMIT"'" -q && git checkout --detach FETCH_HEAD -q
    test "$(git rev-parse HEAD)" = "'"$SGLANG_COMMIT"'" || { echo "commit drift" >&2; exit 1; }
    cp python/pyproject_other.toml python/pyproject.toml && cd python
    SGLANG_BUILD_RUST_EXTS=none SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG='"$SGLANG_VERSION"' \
        python -m build --wheel --outdir /out >/dev/null
    ls /out/sglang-'"$SGLANG_VERSION"'-*.whl >/dev/null
'
sglang_wheel="$stage/sglang-$SGLANG_VERSION-py3-none-any.whl"
test -s "$sglang_wheel" || { echo "FATAL: no sglang wheel produced" >&2; exit 1; }

echo "=== Assembling the build context ==="
mkdir -p "$context/artifacts"
cp "$sglang_wheel" "$plugin_wheel" "$context/artifacts/"
cp "$repo_root/docker/thead/runtime-0.5.18.excludes.txt" \
   "$context/artifacts/runtime-0.5.18.excludes.txt"
cp "$repo_root/docker/thead/top-level-constraints.txt" \
   "$context/artifacts/top-level-constraints.txt"
cp "$repo_root/docker/thead/runtime-0.5.18.containerfile" "$context/Containerfile"

echo "=== Cross-checking the Containerfile against the staged inputs ==="
for staged in "$context"/artifacts/*.whl \
              "$context/artifacts/runtime-0.5.18.excludes.txt" \
              "$context/artifacts/top-level-constraints.txt"; do
    base="$(basename "$staged")"
    grep -q "artifacts/$base" "$context/Containerfile" || {
        echo "FATAL: Containerfile does not reference artifacts/$base" >&2
        exit 1
    }
done
excludes_line="$(grep -v -e '^\s*#' -e '^\s*$' "$context/artifacts/runtime-0.5.18.excludes.txt" | tr -d '[:space:]')"
if [ "$excludes_line" != "triton" ]; then
    echo "FATAL: excludes file must contain exactly the single requirement 'triton', found: $excludes_line" >&2
    exit 1
fi
# The default candidate installs NO platform-artifact wheels. The check
# scans Containerfile INSTRUCTIONS only (COPY/ADD/RUN) — comments must
# stay free to document the dormant fallbacks (stage_ppu_kernel_fallback.sh).
if grep -E '^[[:space:]]*(COPY|ADD|RUN)\b' "$context/Containerfile" \
        | grep -qE 'sglang[-_]kernel|flag_gems[-.]'; then
    echo "FATAL: a Containerfile instruction references a platform-artifact wheel (kernel/flaggems); the default candidate must not install them" >&2
    exit 1
fi

echo
echo "=== staging record ==="
echo "source-commit:    $(git -C "$repo_root" rev-parse HEAD)"
echo "builder:          $BUILD_IMAGE (digest-pinned)"
echo "plugin-wheel:     $(basename "$plugin_wheel")"
echo "plugin-sha256:    $(shasum -a 256 "$context/artifacts/$(basename "$plugin_wheel")" | awk '{print $1}')   (regenerated per build; not pinned)"
echo "sglang-source:    sgl-project/sglang @$SGLANG_COMMIT (release/v0.5.18 cherry-pick, tag is ff4c6e64)"
echo "sglang-wheel:     $(basename "$sglang_wheel")"
echo "sglang-sha256:    $(shasum -a 256 "$context/artifacts/$(basename "$sglang_wheel")" | awk '{print $1}')"
echo "constraints:      artifacts/top-level-constraints.txt (numpy/pyarrow, resolver steering only)"
echo "kernel:           NOT staged (semantic placeholder route); fallback: docker/thead/stage_ppu_kernel_fallback.sh"
echo "flaggems:         Runtime-owned 5.4.0 (no artifact layer); device-phase fallback recipe in this script's header"
echo "context:          $context (survives this script; remove manually when done)"
echo "next step:"
echo "  docker build --platform linux/amd64 -f $context/Containerfile -t <tag> $context"
echo "  then: docker/thead/scripts/verify_image.sh <tag>"
