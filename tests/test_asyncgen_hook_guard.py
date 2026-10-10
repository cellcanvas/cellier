"""The guard that keeps a finished test's asyncio hooks from coming back.

See `tests.conftest._forget_finished_asyncio_hooks` for the failure it
prevents; ``scripts/ci_profiling/repro_stale_asyncgen_hooks.py`` shows that
failure end to end.  These tests build the two states the guard repairs by
hand, so they do not depend on which canvases earlier tests left behind.
"""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest

from tests.conftest import _forget_finished_asyncio_hooks


@pytest.fixture
def closed_loop_hooks():
    """The asyncgen hooks of an asyncio loop that has run and is closed."""
    loop = asyncio.new_event_loop()
    seen = {}

    async def body():
        seen["hooks"] = sys.get_asyncgen_hooks()

    loop.run_until_complete(body())
    loop.close()
    return seen["hooks"]


@pytest.fixture
def keep_asyncgen_hooks():
    before = sys.get_asyncgen_hooks()
    yield
    sys.set_asyncgen_hooks(*before)


def test_installed_hooks_of_a_closed_loop_are_dropped(
    closed_loop_hooks, keep_asyncgen_hooks
):
    sys.set_asyncgen_hooks(*closed_loop_hooks)
    _forget_finished_asyncio_hooks()
    assert sys.get_asyncgen_hooks().firstiter is None


def test_a_rendercanvas_loop_will_not_put_them_back(
    closed_loop_hooks, keep_asyncgen_hooks, monkeypatch
):
    interrupt_hooks = object()
    loop = SimpleNamespace(_BaseLoop__hook_data=(closed_loop_hooks, interrupt_hooks))
    monkeypatch.setitem(sys.modules, "rendercanvas.fake", SimpleNamespace(loop=loop))
    _forget_finished_asyncio_hooks()
    assert loop._BaseLoop__hook_data == ((None, None), interrupt_hooks)


def test_other_hooks_are_left_alone(keep_asyncgen_hooks, monkeypatch):
    def firstiter(agen):
        pass

    def finalizer(agen):
        pass

    saved = ((firstiter, finalizer), None)
    loop = SimpleNamespace(_BaseLoop__hook_data=saved)
    monkeypatch.setitem(sys.modules, "rendercanvas.fake", SimpleNamespace(loop=loop))
    sys.set_asyncgen_hooks(firstiter, finalizer)
    _forget_finished_asyncio_hooks()
    assert sys.get_asyncgen_hooks().firstiter is firstiter
    assert loop._BaseLoop__hook_data is saved
