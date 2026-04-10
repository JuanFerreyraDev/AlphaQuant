"""Tests for src.config.settings_loader."""

from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
import yaml

from src.config.settings_loader import (
    get_active_market,
    get_active_symbols,
    get_market_config,
    get_project_root,
    load_settings,
)

SAMPLE_YAML = """
global:
  active_market: "futures"
  timeframe: "1d"
futures:
  default_leverage: 5
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


class TestLoadSettings:
    """Tests for load_settings()."""

    def test_loads_valid_yaml(self, yaml_file: Path) -> None:
        with patch(
            "src.config.settings_loader._settings_path", return_value=yaml_file
        ):
            result = load_settings()
            assert isinstance(result, dict)
            assert "global" in result
            assert result["global"]["active_market"] == "futures"

    def test_returns_empty_on_missing_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.yaml"
        with patch(
            "src.config.settings_loader._settings_path", return_value=missing
        ):
            result = load_settings()
            assert result == {}

    def test_returns_empty_on_malformed_yaml(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "settings.yaml"
        bad_file.write_text(": : : invalid yaml [", encoding="utf-8")
        with patch(
            "src.config.settings_loader._settings_path", return_value=bad_file
        ):
            result = load_settings()
            assert result == {}


class TestGetActiveMarket:
    def test_returns_futures_by_default(self, yaml_file: Path) -> None:
        with patch(
            "src.config.settings_loader._settings_path", return_value=yaml_file
        ):
            assert get_active_market() == "futures"


class TestGetActiveSymbols:
    def test_returns_futures_symbols(self, yaml_file: Path) -> None:
        with patch(
            "src.config.settings_loader._settings_path", return_value=yaml_file
        ):
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
        with patch(
            "src.config.settings_loader._settings_path", return_value=both_yaml
        ):
            symbols = get_active_symbols()
            assert "BTC_USDT" in symbols
            assert "BNB_USDT" in symbols


class TestGetMarketConfig:
    def test_returns_futures_config(self, yaml_file: Path) -> None:
        with patch(
            "src.config.settings_loader._settings_path", return_value=yaml_file
        ):
            cfg = get_market_config("futures")
            assert cfg["default_leverage"] == 5


class TestGetProjectRoot:
    def test_returns_path(self) -> None:
        root = get_project_root()
        assert isinstance(root, Path)


# ============================================================================
#  EDGE CASES
# ============================================================================


class TestLoadSettingsEdgeCases:
    """Edge cases and error paths for load_settings()."""

    def test_returns_empty_when_yaml_contains_only_scalar(self, tmp_path: Path) -> None:
        """YAML that parses as a string (not dict) must return {}."""
        # Arrange
        scalar_file = tmp_path / "settings.yaml"
        scalar_file.write_text("just a plain string", encoding="utf-8")

        # Act
        with patch(
            "src.config.settings_loader._settings_path", return_value=scalar_file
        ):
            result = load_settings()

        # Assert
        assert result == {}

    def test_returns_empty_on_os_error(self, tmp_path: Path) -> None:
        """OSError when reading the file must return {}."""
        # Arrange
        real_file = tmp_path / "settings.yaml"
        real_file.write_text(SAMPLE_YAML, encoding="utf-8")

        # Act
        with patch(
            "src.config.settings_loader._settings_path", return_value=real_file
        ):
            with patch("pathlib.Path.open", side_effect=OSError("permission denied")):
                result = load_settings()

        # Assert
        assert result == {}


class TestGetActiveMarketEdgeCases:
    def test_returns_default_when_global_section_missing(self, tmp_path: Path) -> None:
        """If the 'global' section is missing, returns 'futures'."""
        # Arrange
        no_global = tmp_path / "settings.yaml"
        no_global.write_text("futures:\n  symbols:\n    - BTC_USDT\n", encoding="utf-8")

        # Act
        with patch(
            "src.config.settings_loader._settings_path", return_value=no_global
        ):
            result = get_active_market()

        # Assert
        assert result == "futures"


class TestGetActiveSymbolsEdgeCases:
    def test_returns_empty_list_when_market_has_no_symbols(self, tmp_path: Path) -> None:
        """Active market without a 'symbols' key returns an empty list."""
        # Arrange
        no_symbols = tmp_path / "settings.yaml"
        no_symbols.write_text(
            'global:\n  active_market: "futures"\nfutures:\n  leverage: 5\n',
            encoding="utf-8",
        )

        # Act
        with patch(
            "src.config.settings_loader._settings_path", return_value=no_symbols
        ):
            symbols = get_active_symbols()

        # Assert
        assert symbols == []

    def test_deduplicates_when_both_markets_share_symbol(self, tmp_path: Path) -> None:
        """active_market='both' deduplicates shared symbols."""
        # Arrange
        both_shared = tmp_path / "settings.yaml"
        both_shared.write_text(
            'global:\n  active_market: "both"\n'
            "futures:\n  symbols:\n    - BTC_USDT\n    - ETH_USDT\n"
            "spot:\n  symbols:\n    - BTC_USDT\n    - SOL_USDT\n",
            encoding="utf-8",
        )

        # Act
        with patch(
            "src.config.settings_loader._settings_path", return_value=both_shared
        ):
            symbols = get_active_symbols()

        # Assert
        assert len(symbols) == 3
        assert set(symbols) == {"BTC_USDT", "ETH_USDT", "SOL_USDT"}


class TestGetMarketConfigEdgeCases:
    def test_returns_empty_dict_for_unknown_market(self, yaml_file: Path) -> None:
        """Non-existent market returns {}."""
        # Arrange / Act
        with patch(
            "src.config.settings_loader._settings_path", return_value=yaml_file
        ):
            cfg = get_market_config("options")

        # Assert
        assert cfg == {}
