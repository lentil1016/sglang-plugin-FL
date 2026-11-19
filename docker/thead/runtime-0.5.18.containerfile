# T-Head PPU (PPU-ZW810E) SGLang 0.5.18 CI image — two-wheel contract
# candidate on the operations Runtime.
#
# Structure (mirrors docker/cuda/runtime-0.5.18.containerfile; this file is
# orchestration only, every dependency decision comes from wheel METADATA,
# and all verification runs against the completed image via
# docker/thead/scripts/verify_image.sh — never during construction):
#
#   1. FROM the digest-pinned operations Runtime (the build inputs are the
#      containerfile, the two staged application wheels, the excludes list,
#      and the platform resolution constraints file — nothing else; the
#      constraints steer the resolver only and never enter wheel METADATA).
#   2. Activate the Runtime's /opt/flagtree Triton slot as image-level
#      PYTHONPATH. The runtime image exposes no PYTHONPATH at all (a bare
#      python3 cannot import triton — probed on the pinned digest), so the
#      image must select the compiler slot the way the Runtime documents
#      for sessions. Runtime-owned /opt/flagtree files stay untouched.
#   3. One normal resolver install of the two application wheels with the
#      Runtime's own uv: sglang[srt_empty,runtime_common] +
#      sglang_fl[thead]. The single-item excludes file drops the standalone
#      triton distribution from resolution (nothing is installed,
#      uninstalled, or stubbed for it). Everything else the plugin needs is
#      already Runtime-owned:
#        - flag_gems 5.4.0 (metadata Name spelled `flag_gems`) is a Runtime
#          distribution, exactly like on NVIDIA; the plugin's unpinned
#          generic requirement is satisfied without the resolver touching
#          it. Whether 5.4.0 reproduces the device-green path (validated
#          with 5.5.0.dev0 built from commit 7536a18c) is a device-phase
#          question — see the phase report; the wheel-build recipe is
#          preserved in git history (stage_wheels.sh, this branch).
#        - NO PPU sglang-kernel wheel: the plugin's import guard
#          (sglang_fl/dispatch/backends/vendor/thead/import_guard.py)
#          serves a semantic sgl_kernel placeholder instead
#          (is_arch_support_pdl() -> False, scalar_type absent, everything
#          else fails loudly on real use). The digest-pinned kernel-wheel
#          fallback stays designed and verified but dormant — see
#          docker/thead/stage_ppu_kernel_fallback.sh.
#   4. The validated T-Head runtime ENV block (carried over from the green
#      empty-0.5.18 image, minus the FlagCX pair: the operations Runtime
#      ships no FlagCX tree or distribution — that decision is deferred to
#      the device phase, see the report).
#
# Build (from the staged context produced by stage_wheels.sh):
#   docker build --platform linux/amd64 -f Containerfile -t <tag> <context>

ARG BASE_IMAGE=harbor.baai.ac.cn/flagos-runtime/flagos-runtime-thead-ppu2.1.0@sha256:7837256c2dab931c762f86713a37d3fbecc902792392d67731c982ecda632cef
FROM ${BASE_IMAGE}

# [1] Compiler-slot activation: /opt/flagtree is the Runtime's authoritative
# Triton 3.6.0 (the standalone /opt/triton 3.5.0 vendor slot is never put on
# sys.path). Baked as plain ENV so test steps that never source a profile
# inherit it (same rationale as the Hygon two-wheel image).
ENV PYTHONPATH=/opt/flagtree

# [2] Staged artifacts (paths the stage script guarantees). The
# top-level-constraints file carries the platform resolution constraints
# (numpy/pyarrow, see the file) — steering for the resolver only, never
# wheel METADATA.
COPY artifacts/sglang-0.5.18-py3-none-any.whl /tmp/wheels/sglang-0.5.18-py3-none-any.whl
COPY artifacts/sglang_fl-0.1.0-py3-none-any.whl /tmp/wheels/sglang_fl-0.1.0-py3-none-any.whl
COPY artifacts/runtime-0.5.18.excludes.txt /tmp/runtime-0.5.18.excludes.txt
COPY artifacts/top-level-constraints.txt /tmp/top-level-constraints.txt

# [3] The two application wheels resolve together in one graph.
# --default-index is an ARG override (hygon precedent): build hosts that
# cannot reach files.pythonhosted.org reliably pass a mirror via
# --build-arg PYPI_INDEX=...; the default keeps official PyPI semantics.
# UV_HTTP_TIMEOUT is a download-robustness knob passed inline (transpacific
# resets killed a 4-retry default mid-closure); it is not baked into the
# image and does not change resolution.
ARG PYPI_INDEX=https://pypi.org/simple
RUN UV_HTTP_TIMEOUT=180 /usr/local/bin/uv pip install --python /flagos/bin/python3 \
      --default-index "$PYPI_INDEX" \
      --constraints /tmp/top-level-constraints.txt \
      --excludes /tmp/runtime-0.5.18.excludes.txt \
      '/tmp/wheels/sglang-0.5.18-py3-none-any.whl[srt_empty,runtime_common]' \
      '/tmp/wheels/sglang_fl-0.1.0-py3-none-any.whl[thead]'

# [4] Validated T-Head runtime ENV (from the green empty-0.5.18 image):
#   SGLANG_PLUGINS/SGLANG_PLATFORM: plugin + platform activation.
#   USE_FLAGGEMS / SGLANG_FL_*: Layer-1 FlagGems op path and dispatch prefs.
#   SGLANG_IS_FLASHINFER_AVAILABLE=false: no flashinfer in the Runtime (the
#     plugin's import guard also forces this while its placeholder is
#     active; baked for the published-image users who run without the
#     plugin).
#   SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1: the sgl-kernel version probe is
#     meaningless here (semantic placeholder has no dist-info; the fallback
#     PPU wheel 0.4.2.post2 is older than the 0.4.6.post1 NVIDIA pin — the
#     import guard sets the same override while its placeholder is active).
#   Deliberately NOT set: SGLANG_FL_DIST_BACKEND / FLAGCX_PATH (the Runtime
#     carries no FlagCX tree; deferring to the device phase).
ENV SGLANG_PLUGINS=sglang_fl \
    SGLANG_PLATFORM=sglang_fl \
    USE_FLAGGEMS=1 \
    SGLANG_FL_OOT_ENABLED=1 \
    SGLANG_FL_PREFER=flagos \
    SGLANG_FL_THEAD_MOE_FLAGGEMS=1 \
    SGLANG_IS_FLASHINFER_AVAILABLE=false \
    SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1

# CI runs offline against the mounted model store (see .github/configs/
# thead.yml container_options). HF_HOME is pinned because the runner
# overrides HOME=/github/home at job time.
ENV HF_HOME=/opt/hf-cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# The PPU pccl libraries under /opt/accl-p: the validated image bakes this
# prefix ahead of the Runtime's SDK paths because runner hosts differ in
# whether the login profile is sourced (GitHub Actions steps run
# bash --noprofile --norc).
ENV LD_LIBRARY_PATH=/opt/accl-p:${LD_LIBRARY_PATH}

WORKDIR /workspace
