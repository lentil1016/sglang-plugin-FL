# Shared fixtures for dispatch unit tests.


import pytest

from sglang_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl
from sglang_fl.dispatch.registry import OpRegistry
from sglang_fl.dispatch.policy import reset_global_policy


@pytest.fixture(autouse=True)
def _reset_policy():
    """Reset global policy before each test."""
    reset_global_policy()
    yield
    reset_global_policy()


@pytest.fixture
def registry():
    """Fresh OpRegistry instance."""
    return OpRegistry()


@pytest.fixture
def dummy_fn():
    """A simple callable for use in OpImpl."""

    def _fn(*args, **kwargs):
        return "dummy_result"

    return _fn


@pytest.fixture
def make_impl():
    """Factory fixture for creating OpImpl instances."""

    def _make(
        op_name="test_op",
        impl_id="test.impl",
        kind=BackendImplKind.DEFAULT,
        fn=None,
        vendor=None,
        priority=BackendPriority.DEFAULT,
    ):
        if fn is None:

            def fn(*a, **kw):
                return f"result_from_{impl_id}"

        return OpImpl(
            op_name=op_name,
            impl_id=impl_id,
            kind=kind,
            fn=fn,
            vendor=vendor,
            priority=priority,
        )

    return _make
