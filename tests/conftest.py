"""Shared fixtures.

The synthetic env (deepEmulator/platforms/synthetic.py, promoted from
scripts/nano_e2e.py) is the ROM-free workhorse for pipeline and
multiprocessing tests — fixtures here save each test the construction
boilerplate.
"""
from __future__ import annotations

import pytest


@pytest.fixture()
def synthetic_cartridge():
    from deepEmulator.platforms.synthetic import SyntheticCartridge

    return SyntheticCartridge()


@pytest.fixture()
def synthetic_env(synthetic_cartridge):
    from deepEmulator.platforms.synthetic import SyntheticEmulatorEnv

    env = SyntheticEmulatorEnv(synthetic_cartridge, episode_length=20, seed=0)
    yield env
    env.close()
