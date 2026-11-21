"""User-authorized, reversible account routing; no credentials are copied."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROXY_KEYS = ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy')

def apply_worker_route(environment):
    route = json.loads((ROOT / 'agent_account_route.json').read_text())
    if route.get('mode') != 'current_account_proxy':
        raise ValueError('Cashew requires the isolated current-account proxy route')
    expected_home=Path('@WORKSPACE@/.codex_training_current_account')
    if not expected_home.is_dir() or Path(route.get('codex_home','')).resolve()!=expected_home.resolve():
        raise ValueError('Cashew requires the isolated training account directory')
    binary=ROOT/'codex_scientific'
    if not binary.is_file() or Path(route.get('codex_binary','')).resolve()!=binary.resolve():
        raise ValueError('Cashew requires its pinned Codex launcher')
    if not any(environment.get(k) for k in ('HTTPS_PROXY', 'https_proxy')):
        raise ValueError('Training account requires an explicit HTTPS proxy')
    environment.update(CODEX_HOME=route['codex_home'],
                       CASHEW_FADA_CODEX_HOME=route['codex_home'],
                       CASHEW_FADA_CODEX_BINARY=route['codex_binary'])
    return environment
