"""One account setting for the v4 supervisor and Agent dispatcher.

The current Cashew contract uses jhyapi. An explicit override may name the
same directory (including an equivalent symlink), never a different account.
"""

from __future__ import annotations

import os
from pathlib import Path


ACCOUNT_ENV_VAR = "CASHEW_FADA_CODEX_HOME"
DEFAULT_CODEX_HOME = Path("@WORKSPACE@/.codex_training_current_account")


def codex_home() -> Path:
    raw = os.environ.get(ACCOUNT_ENV_VAR)
    if raw is not None and not raw.strip():
        raise ValueError(f"{ACCOUNT_ENV_VAR} must be a non-empty directory path")
    path = Path(raw).expanduser() if raw is not None else DEFAULT_CODEX_HOME
    if not path.is_absolute():
        raise ValueError(f"{ACCOUNT_ENV_VAR} must be an absolute directory path: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"Codex account directory does not exist: {path}")
    if path.resolve() != DEFAULT_CODEX_HOME.resolve():
        raise ValueError('Cashew account must be @WORKSPACE@/.codex_training_current_account')
    return path.resolve()
