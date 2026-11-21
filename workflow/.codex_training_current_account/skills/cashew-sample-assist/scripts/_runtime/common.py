import hashlib
import json
import os
from pathlib import Path

PROTOCOL_ROOT = Path(__file__).resolve().parent / 'protocol_schema'
SKILL_ROOT = Path(__file__).resolve().parents[2]
SKILL = SKILL_ROOT / 'SKILL.md'
MODEL, EFFORT = 'gpt-6.2-sol', 'max'
SERVICE_TIER = 'default'


def training_account_home(environment=None):
    environment = os.environ if environment is None else environment
    configured = environment.get('CASHEW_CODEX_ACCOUNT_HOME')
    return (Path(configured) if configured else SKILL_ROOT.parents[1]).resolve()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
