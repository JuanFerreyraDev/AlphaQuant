"""Tests for src.config.settings_loader."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from src.config.settings_loader import (
    _load_yaml_defaults,
    get_active_market,
    get_active_symbols,
    get_config,
    get_market_config,
    get_project_root,
    get_trading_settings,
    load_bot_state,
    load_settings,
    save_bot_state,
)

SAMPLE_YAML = """
global:
  active_market: "futures"
  timeframe: "1d"
  risk_per_trade_pct: 1.0
futures:
  default_leverage: 5
  margin_type: ISOLATED
  symbols:
    - "BTC_USDT"
    - "ETH_USDT"
spot:
  symbols:
    - "BTC_USDT"
"""


@pytest.fixture
def yaml_file(tmp_path: Path) -> Path:
    """Create a temporary settings.yaml for use in tests."""
    f = tmp_path / "settings.yaml"
    f.write_text(SAMPLE_YAML, encoding="utf-8")
    return f


@pytest.fixture
def state_dir(tmp_path: Path) -> Path:
    """Create a temporary data/ directory for bot_state.json."""
    d = tmp_path / "data"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# _load_yaml_defaults
# ---------------------------------------------------------------------------


class TestLoadYamlDefaults:
    """Tests for _load_yaml_defaults()."""

    def test_loads_valid_yaml(self, yaml_file: Path) -> None:
        with patch("src.config.settings_loader._settings_path", return_value=yaml_file):
            result = _load_yaml_defaults()
            assert isinstance(result, dict)
            assert "global" in result
            assert result["global"]["active_market"] == "futures"

    def test_returns_empty_on_missing_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.yaml"
        with patch("src.config.settings_loader._settings_path", return_value=missing):
            result = _load_yaml_defaults()
            assert result == {}

    def test_returns_empty_on_malformed_yaml(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "settings.yaml"
        bad_file.write_text(": : : invalid yaml [", encoding="utf-8")
        with patch("src.config.settings_loader._settings_path", return_value=bad_file):
            result = _load_yaml_defaults()
            assert result == {}

    def test_returns_empty_when_yaml_contains_only_scalar(self, tmp_path: Path) -> None:
        """YAML that parses as a string (not dict) must return {}."""
        scalar_file = tmp_path / "settings.yaml"
        scalar_file.write_text("just a plain string", encoding="utf-8")

        with patch(
            "src.config.settings_loader._settings_path", return_value=scalar_file
        ):
            result = _load_yaml_defaults()

        assert result == {}

    def test_returns_empty_on_os_error(self, tmp_path: Path) -> None:
        """OSError when reading the file must return {}."""
        real_file = tmp_path / "settings.yaml"
        real_file.write_text(SAMPLE_YAML, encoding="utf-8")

        with patch("src.config.settings_loader._settings_path", return_value=real_file):
            with patch("pathlib.Path.open", side_effect=OSError("permission denied")):
                result = _load_yaml_defaults()

        assert result == {}


# ---------------------------------------------------------------------------
# load_bot_state / save_bot_state
# ---------------------------------------------------------------------------


class TestLoadBotState:
    """Tests for load_bot_state()."""

    def test_creates_file_if_missing(self, tmp_path: Path) -> None:
        """First access creates bot_state.json with default schema."""
        state_path = tmp_path / "data" / "bot_state.json"
        with patch(
            "src.config.settings_loader._bot_state_path", return_value=state_path
        ), patch("src.config.settings_loader._load_yaml_defaults", return_value={}):
            result = load_bot_state()

        assert state_path.exists()
        assert isinstance(result, dict)
        assert "active_market" in result
        assert "symbols" in result
        assert "user_preferences" in result

    def test_reads_existing_state(self, tmp_path: Path) -> None:
        """Reads an existing bot_state.json correctly."""
        state_path = tmp_path / "data" / "bot_state.json"
        state_path.parent.mkdir(parents=True)
        expected = {"active_market": "spot", "symbols": {"futures": ["SOL_USDT"]}}
        state_path.write_text(json.dumps(expected), encoding="utf-8")

        with patch(
            "src.config.settings_loader._bot_state_path", return_value=state_path
        ):
            result = load_bot_state()

        assert result["active_market"] == "spot"
        assert result["symbols"]["futures"] == ["SOL_USDT"]

    def test_returns_default_on_corrupt_json(self, tmp_path: Path) -> None:
        """Corrupt JSON returns the default schema."""
        state_path = tmp_path / "data" / "bot_state.json"
        state_path.parent.mkdir(parents=True)
        state_path.write_text("{not valid json!!!", encoding="utf-8")

        with patch(
            "src.config.settings_loader._bot_state_path", return_value=state_path
        ):
            result = load_bot_state()

        assert isinstance(result, dict)
        assert "active_market" in result

    def test_returns_default_on_non_dict_json(self, tmp_path: Path) -> None:
        """JSON that is a list (not dict) returns default schema."""
        state_path = tmp_path / "data" / "bot_state.json"
        state_path.parent.mkdir(parents=True)
        state_path.write_text("[1, 2, 3]", encoding="utf-8")

        with patch(
            "src.config.settings_loader._bot_state_path", return_value=state_path
        ):
            result = load_bot_state()

        assert isinstance(result, dict)
        assert "active_market" in result


class TestSaveBotState:
    """Tests for save_bot_state()."""

    def test_writes_and_reads_back(self, tmp_path: Path) -> None:
        """Data written by save_bot_state can be read back."""
        state_path = tmp_path / "data" / "bot_state.json"

        with patch(
            "src.config.settings_loader._bot_state_path", return_value=state_path
        ):
            save_bot_state({"active_market": "spot", "custom_key": 42})
            result = load_bot_state()

        assert result["active_market"] == "spot"
        assert result["custom_key"] == 42

    def test_atomic_write_creates_no_temp_file_on_success(self, tmp_path: Path) -> None:
        """After a successful write, no .tmp file remains."""
        state_path = tmp_path / "data" / "bot_state.json"

        with patch(
            "src.config.settings_loader._bot_state_path", return_value=state_path
        ):
            save_bot_state({"active_market": "futures"})

        tmp_files = list(state_path.parent.glob(".bot_state_*.tmp"))
        assert len(tmp_files) == 0

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        """Parent directories are created if they don't exist."""
        state_path = tmp_path / "deep" / "nested" / "bot_state.json"

        with patch(
            "src.config.settings_loader._bot_state_path", return_value=state_path
        ):
            save_bot_state({"active_market": "futures"})

        assert state_path.exists()


class TestEnsureBotState:
    """Tests for _ensure_bot_state()."""

    def test_seeds_from_yaml_defaults(self, tmp_path: Path) -> None:
        """Initial state is seeded from YAML defaults."""
        state_path = tmp_path / "data" / "bot_state.json"
        yaml_data = {
            "global": {"active_market": "spot", "risk_per_trade_pct": 2.5},
            "futures": {
                "default_leverage": 10,
                "margin_type": "CROSSED",
                "symbols": ["ETH_USDT"],
            },
            "spot": {"symbols": ["SOL_USDT"]},
        }

        with patch(
            "src.config.settings_loader._bot_state_path", return_value=state_path
        ), patch(
            "src.config.settings_loader._load_yaml_defaults", return_value=yaml_data
        ):
            result = load_bot_state()

        assert result["active_market"] == "spot"
        assert result["symbols"]["futures"] == ["ETH_USDT"]
        assert result["symbols"]["spot"] == ["SOL_USDT"]
        assert result["user_preferences"]["risk_per_trade_pct"] == 2.5
        assert result["user_preferences"]["default_leverage"] == 10
        assert result["margin_type"] == "CROSSED"

    def test_uses_hardcoded_defaults_when_yaml_empty(self, tmp_path: Path) -> None:
        """If YAML is empty, uses hardcoded defaults."""
        state_path = tmp_path / "data" / "bot_state.json"

        with patch(
            "src.config.settings_loader._bot_state_path", return_value=state_path
        ), patch("src.config.settings_loader._load_yaml_defaults", return_value={}):
            result = load_bot_state()

        assert result["active_market"] == "futures"
        assert result["symbols"]["futures"] == ["BTC_USDT"]
        assert result["user_preferences"]["default_leverage"] == 2


# ---------------------------------------------------------------------------
# get_config (merge reader)
# ---------------------------------------------------------------------------


class TestGetConfig:
    """Tests for get_config() — the merge reader."""

    def test_bot_state_overrides_yaml(self, tmp_path: Path) -> None:
        """Bot state values take priority over YAML defaults."""
        yaml_data = {
            "global": {"active_market": "futures", "risk_per_trade_pct": 1.0},
            "futures": {"default_leverage": 5, "symbols": ["BTC_USDT"]},
        }
        state = {
            "active_market": "spot",
            "symbols": {"futures": ["ETH_USDT", "SOL_USDT"]},
            "user_preferences": {"risk_per_trade_pct": 3.0, "default_leverage": 20},
            "margin_type": "CROSSED",
        }

        with patch(
            "src.config.settings_loader._load_yaml_defaults", return_value=yaml_data
        ), patch("src.config.settings_loader.load_bot_state", return_value=state):
            result = get_config()

        assert result["global"]["active_market"] == "spot"
        assert result["futures"]["symbols"] == ["ETH_USDT", "SOL_USDT"]
        assert result["global"]["risk_per_trade_pct"] == 3.0
        assert result["futures"]["default_leverage"] == 20
        assert result["futures"]["margin_type"] == "CROSSED"

    def test_falls_back_to_yaml_when_state_empty(self, tmp_path: Path) -> None:
        """Missing state keys fall back to YAML values."""
        yaml_data = {
            "global": {
                "active_market": "futures",
                "timeframe": "1d",
                "risk_per_trade_pct": 1.0,
            },
            "futures": {
                "default_leverage": 5,
                "margin_type": "ISOLATED",
                "symbols": ["BTC_USDT"],
            },
        }
        state: dict = {}  # empty bot state

        with patch(
            "src.config.settings_loader._load_yaml_defaults", return_value=yaml_data
        ), patch("src.config.settings_loader.load_bot_state", return_value=state):
            result = get_config()

        assert result["global"]["active_market"] == "futures"
        assert result["global"]["timeframe"] == "1d"
        assert result["futures"]["default_leverage"] == 5
        assert result["futures"]["symbols"] == ["BTC_USDT"]

    def test_preserves_yaml_only_keys(self) -> None:
        """Keys only in YAML (like timeframe) are preserved in merged output."""
        yaml_data = {"global": {"timeframe": "4h", "active_market": "futures"}}
        state = {"active_market": "spot", "symbols": {}, "user_preferences": {}}

        with patch(
            "src.config.settings_loader._load_yaml_defaults", return_value=yaml_data
        ), patch("src.config.settings_loader.load_bot_state", return_value=state):
            result = get_config()

        assert result["global"]["timeframe"] == "4h"
        assert result["global"]["active_market"] == "spot"

    def test_load_settings_is_alias(self) -> None:
        """load_settings and get_config are the same function."""
        assert load_settings is get_config


# ---------------------------------------------------------------------------
# Convenience readers
# ---------------------------------------------------------------------------


class TestGetActiveMarket:
    def test_returns_futures_by_default(self, yaml_file: Path) -> None:
        with patch("src.config.settings_loader._settings_path", return_value=yaml_file):
            with patch(
                "src.config.settings_loader.load_bot_state",
                return_value={"active_market": "futures"},
            ):
                assert get_active_market() == "futures"


class TestGetActiveSymbols:
    def test_returns_futures_symbols(self, yaml_file: Path) -> None:
        state = {
            "active_market": "futures",
            "symbols": {"futures": ["BTC_USDT", "ETH_USDT"]},
            "user_preferences": {},
        }
        with patch("src.config.settings_loader._settings_path", return_value=yaml_file):
            with patch("src.config.settings_loader.load_bot_state", return_value=state):
                symbols = get_active_symbols()
                assert "BTC_USDT" in symbols
                assert "ETH_USDT" in symbols

    def test_both_merges_symbols(self, tmp_path: Path) -> None:
        both_yaml = tmp_path / "settings.yaml"
        both_yaml.write_text(
            'global:\n  active_market: "both"\n'
            "futures:\n  symbols:\n    - BTC_USDT\n"
            "spot:\n  symbols:\n    - BNB_USDT\n",
            encoding="utf-8",
        )
        state = {
            "active_market": "both",
            "symbols": {"futures": ["BTC_USDT"], "spot": ["BNB_USDT"]},
            "user_preferences": {},
        }
        with patch("src.config.settings_loader._settings_path", return_value=both_yaml):
            with patch("src.config.settings_loader.load_bot_state", return_value=state):
                symbols = get_active_symbols()
                assert "BTC_USDT" in symbols
                assert "BNB_USDT" in symbols


class TestGetMarketConfig:
    def test_returns_futures_config(self, yaml_file: Path) -> None:
        state = {
            "user_preferences": {"default_leverage": 5},
            "symbols": {"futures": ["BTC_USDT"]},
            "margin_type": "ISOLATED",
        }
        with patch("src.config.settings_loader._settings_path", return_value=yaml_file):
            with patch("src.config.settings_loader.load_bot_state", return_value=state):
                cfg = get_market_config("futures")
                assert cfg["default_leverage"] == 5


class TestGetTradingSettings:
    """Tests for get_trading_settings()."""

    def test_loads_fee_rate_and_slippage_from_yaml(self) -> None:
        """Reads custom fee_rate and slippage from the trading section."""
        yaml_data = {"trading": {"fee_rate": 0.002, "slippage": 0.001}}
        with patch(
            "src.config.settings_loader._load_yaml_defaults", return_value=yaml_data
        ):
            result = get_trading_settings()
        assert result["fee_rate"] == 0.002
        assert result["slippage"] == 0.001

    def test_uses_defaults_when_keys_missing(self) -> None:
        """Missing keys inside trading section fall back to defaults."""
        with patch(
            "src.config.settings_loader._load_yaml_defaults",
            return_value={"trading": {}},
        ):
            result = get_trading_settings()
        assert result["fee_rate"] == 0.001
        assert result["slippage"] == 0.0005

    def test_uses_defaults_when_section_absent(self) -> None:
        """Absent trading section falls back to defaults entirely."""
        with patch(
            "src.config.settings_loader._load_yaml_defaults", return_value={}
        ):
            result = get_trading_settings()
        assert result["fee_rate"] == 0.001
        assert result["slippage"] == 0.0005


class TestGetProjectRoot:
    def test_returns_path(self) -> None:
        root = get_project_root()
        assert isinstance(root, Path)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestGetActiveMarketEdgeCases:
    def test_returns_default_when_global_section_missing(self, tmp_path: Path) -> None:
        """If the 'global' section is missing, returns 'futures'."""
        no_global = tmp_path / "settings.yaml"
        no_global.write_text("futures:\n  symbols:\n    - BTC_USDT\n", encoding="utf-8")

        with patch("src.config.settings_loader._settings_path", return_value=no_global):
            with patch("src.config.settings_loader.load_bot_state", return_value={}):
                result = get_active_market()

        assert result == "futures"


class TestGetActiveSymbolsEdgeCases:
    def test_returns_empty_list_when_market_has_no_symbols(
        self, tmp_path: Path
    ) -> None:
        """Active market without a 'symbols' key returns an empty list."""
        no_symbols = tmp_path / "settings.yaml"
        no_symbols.write_text(
            'global:\n  active_market: "futures"\nfutures:\n  leverage: 5\n',
            encoding="utf-8",
        )

        with patch(
            "src.config.settings_loader._settings_path", return_value=no_symbols
        ):
            with patch("src.config.settings_loader.load_bot_state", return_value={}):
                symbols = get_active_symbols()

        assert symbols == []

    def test_deduplicates_when_both_markets_share_symbol(self, tmp_path: Path) -> None:
        """active_market='both' deduplicates shared symbols."""
        both_shared = tmp_path / "settings.yaml"
        both_shared.write_text(
            'global:\n  active_market: "both"\n'
            "futures:\n  symbols:\n    - BTC_USDT\n    - ETH_USDT\n"
            "spot:\n  symbols:\n    - BTC_USDT\n    - SOL_USDT\n",
            encoding="utf-8",
        )

        state = {
            "active_market": "both",
            "symbols": {
                "futures": ["BTC_USDT", "ETH_USDT"],
                "spot": ["BTC_USDT", "SOL_USDT"],
            },
            "user_preferences": {},
        }

        with patch(
            "src.config.settings_loader._settings_path", return_value=both_shared
        ):
            with patch("src.config.settings_loader.load_bot_state", return_value=state):
                symbols = get_active_symbols()

        assert len(symbols) == 3
        assert set(symbols) == {"BTC_USDT", "ETH_USDT", "SOL_USDT"}


class TestGetMarketConfigEdgeCases:
    def test_returns_empty_dict_for_unknown_market(self, yaml_file: Path) -> None:
        """Non-existent market returns {}."""
        with patch("src.config.settings_loader._settings_path", return_value=yaml_file):
            with patch("src.config.settings_loader.load_bot_state", return_value={}):
                cfg = get_market_config("options")

        assert cfg == {}
