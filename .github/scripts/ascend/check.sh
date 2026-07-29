#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Check Huawei Ascend NPU availability.
set -euo pipefail
echo "=== Checking Ascend NPU availability ==="
npu-smi info
