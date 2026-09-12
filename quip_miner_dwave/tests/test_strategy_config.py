"""Strategy knobs arrive in Configure.backend_toml like the budget's."""

from __future__ import annotations

import logging

from quip_miner_dwave.budget import DWAVE_CONFIG_KEYS, warn_unknown_backend_keys
from quip_miner_dwave.strategy import StrategyConfig, strategy_config_from_toml


def test_defaults_keep_todays_behaviour():
    cfg = StrategyConfig()
    assert cfg.min_win_probability == 0.0
    assert cfg.slot_advantage == 0.25
    assert cfg.explore_fraction == 0.10
    assert strategy_config_from_toml("") == cfg
    assert strategy_config_from_toml("budget = 250m") == cfg  # unparsable: the budget parser reports it


def test_the_three_keys_are_read():
    cfg = strategy_config_from_toml(
        'min_win_probability = 0.05\nslot_advantage = 1\nexplore_fraction = 0\n'
    )
    assert cfg == StrategyConfig(min_win_probability=0.05, slot_advantage=1.0, explore_fraction=0.0)


def test_nonsense_values_warn_and_fall_back(caplog):
    with caplog.at_level(logging.WARNING):
        cfg = strategy_config_from_toml(
            'min_win_probability = 1.5\nslot_advantage = -1\nexplore_fraction = true\n'
        )
    assert cfg == StrategyConfig()
    messages = [r.getMessage() for r in caplog.records]
    assert any("min_win_probability" in m for m in messages)
    assert any("slot_advantage" in m for m in messages)
    assert any("explore_fraction" in m for m in messages)


def test_the_keys_are_known_to_the_dwave_schema(caplog):
    assert {"min_win_probability", "slot_advantage", "explore_fraction"} <= DWAVE_CONFIG_KEYS
    with caplog.at_level(logging.WARNING):
        warn_unknown_backend_keys("min_win_probability = 0.1\nslot_advantage = 0.5\nexplore_fraction = 0.2\n")
    assert not [r for r in caplog.records if "unknown field" in r.getMessage()]
