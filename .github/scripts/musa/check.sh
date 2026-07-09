#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Check Moore Threads MUSA GPU availability.
set -euo pipefail
echo "=== Checking MUSA GPU availability ==="
mthreads-gmi
