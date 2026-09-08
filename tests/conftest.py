"""Test-wide safety net: never touch the user's real data.

`Config()` defaults the stats database to ``~/.posture/posture.db``, so any
test that builds a default config and a Monitor writes into the real one. That
is exactly what happened: the alert tests drove escalation dozens of times and
left bursts of phantom alerts in a live database, which then showed up as "60
alerts today" in the panel.

Redirecting the default here fixes it for every test at once, including ones
not written yet, rather than relying on each to remember.
"""

from __future__ import annotations

import pytest

from posture import config as config_mod
from posture import store as store_mod


@pytest.fixture(autouse=True, scope="session")
def _isolate_user_data(tmp_path_factory):
    sandbox = tmp_path_factory.mktemp("posture-home")
    real_db = store_mod.DEFAULT_PATH
    real_cfg = config_mod.DEFAULT_CONFIG_PATH
    store_mod.DEFAULT_PATH = sandbox / "posture.db"
    config_mod.DEFAULT_CONFIG_PATH = sandbox / "config.json"
    try:
        yield sandbox
    finally:
        store_mod.DEFAULT_PATH = real_db
        config_mod.DEFAULT_CONFIG_PATH = real_cfg


@pytest.fixture(autouse=True)
def _guard_real_paths(_isolate_user_data):
    """Fail loudly if a test somehow still aims at the real locations."""
    assert "posture-home" in str(store_mod.DEFAULT_PATH)
    assert "posture-home" in str(config_mod.DEFAULT_CONFIG_PATH)
