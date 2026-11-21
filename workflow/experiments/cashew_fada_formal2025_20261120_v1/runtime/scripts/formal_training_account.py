"""Use a dedicated Codex account directory with the exact installed release skills."""

import os
from pathlib import Path
import shutil


TRAINING_HOME = Path('@WORKSPACE@/.codex_training_current_account')


def codex_binary():
    return os.environ.get('CASHEW_CODEX_BIN') or shutil.which('codex') or 'codex'


def verify_training_account_environment(environment):
    configured = environment.get('CASHEW_CODEX_ACCOUNT_HOME', str(TRAINING_HOME))
    selected = environment.get('CODEX_HOME')
    if not selected or Path(selected).resolve() != Path(configured).resolve():
        raise ValueError('Select the dedicated training CODEX_HOME')
    home = Path(selected).resolve(strict=True)
    if (home / 'skills').resolve(strict=True) != (TRAINING_HOME / 'skills').resolve(strict=True):
        raise ValueError('Install the bound release skills in the training CODEX_HOME')
    authentication = home / 'auth.json'
    if authentication.exists():
        status = authentication.stat()
        if status.st_mode & 0o077 or status.st_uid != os.getuid():
            raise ValueError('Training authentication must be private to its owner')
