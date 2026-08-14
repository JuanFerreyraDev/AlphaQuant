from __future__ import annotations

import pytest
from src.utils.oos_validation import THRESHOLD_NOT_FOUND


def is_threshold_failed_prefix(thr: float) -> bool:
    """Pre-fix sentinel check logic: treats any thr < 0 as threshold_failed."""
    return thr < 0


def is_threshold_failed_postfix(thr: float) -> bool:
    """Post-fix sentinel check logic: only treats thr == THRESHOLD_NOT_FOUND (-1.0) as threshold_failed."""
    return thr == THRESHOLD_NOT_FOUND


def test_sentinel_logic_prefix_rejects_valid_negative_threshold_demonstration():
    """Demonstrate that the pre-fix logic (thr < 0) incorrectly rejects valid negative thresholds."""
    valid_negative_thr = -0.0035  # Extreme threshold from regression_return grid

    failed_prefix = is_threshold_failed_prefix(valid_negative_thr)

    # Under pre-fix logic, thr=-0.0035 < 0 evaluates to True (falsely flagging threshold_failed)
    assert failed_prefix is True, "Pre-fix logic evaluates -0.0035 < 0 as True (the bug)"


def test_sentinel_logic_postfix_accepts_valid_negative_threshold():
    """Verify that the post-fix logic (thr == THRESHOLD_NOT_FOUND) correctly accepts a valid negative threshold."""
    valid_negative_thr = -0.0035  # Extreme threshold from regression_return grid

    failed_postfix = is_threshold_failed_postfix(valid_negative_thr)

    # Under post-fix logic, thr=-0.0035 == THRESHOLD_NOT_FOUND (-1.0) evaluates to False (correct)
    assert failed_postfix is False, "Post-fix logic evaluates -0.0035 == -1.0 as False (valid threshold accepted)"


def test_sentinel_logic_both_reject_true_sentinel():
    """Verify that both pre-fix and post-fix logic correctly reject the actual sentinel (-1.0)."""
    assert is_threshold_failed_prefix(THRESHOLD_NOT_FOUND) is True
    assert is_threshold_failed_postfix(THRESHOLD_NOT_FOUND) is True
