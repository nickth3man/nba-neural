"""Unit tests for the training loop helpers."""

from era_translator.config import LAMBDA_MAX, MAX_EPOCHS
from era_translator.train import _lambda_at


def test_lambda_schedule_ramps():
    l0 = _lambda_at(0)
    lmid = _lambda_at(MAX_EPOCHS // 2)
    llast = _lambda_at(MAX_EPOCHS - 1)
    assert l0 == 0.0
    assert 0.0 < lmid < LAMBDA_MAX
    assert llast > 0.99 * LAMBDA_MAX
    assert l0 < lmid < llast
