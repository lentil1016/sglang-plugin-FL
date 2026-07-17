#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Install sglang-plugin-FL and test dependencies on Moore Threads MUSA.
set -euo pipefail
git config --global --add safe.directory "$(pwd)"
echo "=== Installing sglang-plugin-FL (MUSA) ==="
pip install --upgrade pip
pip install -e ".[dev]" --no-build-isolation || pip install -e . --no-build-isolation
pip install pytest pytest-timeout
echo "=== Installation complete ==="
python -c "import sglang_fl; print(f'sglang_fl {sglang_fl.__name__} loaded')"
