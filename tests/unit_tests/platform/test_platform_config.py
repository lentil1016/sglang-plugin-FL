# Copyright (c) 2026 BAAI. All rights reserved.

"""Unit tests for the test-runner platform config loader.

Covers ``tests/utils/platform_config.py``, specifically the fallback-discovery
scan in ``_resolve_platform_file`` that reads every ``tests/platforms/*.yaml``
when no direct ``<platform>.yaml`` exists.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from tests.utils.platform_config import PlatformConfig


def test_misencoded_sibling_yaml_is_skipped_not_crashed(tmp_path: Path) -> None:
    """A non-UTF-8 sibling file must be skipped during fallback discovery.

    Regression for the UnicodeDecodeError raised when ``--platform <name>``
    had no direct ``<name>.yaml`` and the glob scan hit a GBK/CP1252-encoded
    sibling. The bad file should be skipped with a RuntimeWarning and
    resolution should proceed to a clear FileNotFoundError.
    """
    platforms = tmp_path / "platforms"
    platforms.mkdir()
    (platforms / "cuda.yaml").write_text(
        "platform: cuda\nvendor: nvidia\ndevice_types:\n  a100: {}\n",
        encoding="utf-8",
    )
    # Byte 0xa3 is a GBK/CP1252 lead byte that is invalid as UTF-8 start byte.
    (platforms / "legacy.yaml").write_bytes(
        b"# legacy config \xa3\xac not utf-8\nplatform: legacy\n"
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(FileNotFoundError, match="ascend"):
            PlatformConfig.load("ascend", platforms_dir=platforms)

    assert any(
        issubclass(w.category, RuntimeWarning) and "legacy.yaml" in str(w.message)
        for w in caught
    ), (
        "expected a RuntimeWarning skipping legacy.yaml, got: "
        f"{[str(w.message) for w in caught]}"
    )


def test_direct_platform_file_loads(tmp_path: Path) -> None:
    """A direct ``<platform>.yaml`` is loaded without invoking the scan."""
    platforms = tmp_path / "platforms"
    platforms.mkdir()
    (platforms / "ascend.yaml").write_text(
        "platform: ascend\nvendor: huawei\n"
        "device_types:\n  910b:\n    memory_gb: 64\n"
        "910b:\n  name: 910b\n  tests:\n    unit:\n      include: '*'\n      exclude: []\n",
        encoding="utf-8",
    )

    cfg = PlatformConfig.load("ascend", platforms_dir=platforms)

    assert cfg.platform == "ascend"
    assert cfg.vendor == "huawei"
    assert "910b" in cfg.device_types
    assert cfg.device == "910b"
