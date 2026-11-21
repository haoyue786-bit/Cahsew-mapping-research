#!/usr/bin/env python3
"""Verify the installed sample-annotation skill route, not evidence isolation."""
import argparse
import hashlib
import json
from pathlib import Path

from _runtime.common import EFFORT, MODEL

SKILL_NAME = 'cashew-sample-assist'
SKILL_DIR = Path(__file__).resolve().parents[1]


def validate_binding(thread, turn, expected_model=MODEL, expected_effort=EFFORT):
    params = turn.get('params', turn)
    matches = [x for x in params.get('input', [])
               if x.get('type') == 'skill' and x.get('name') == SKILL_NAME]
    expected = SKILL_DIR / 'SKILL.md'
    if len(matches) != 1 or Path(matches[0].get('path', '')).resolve() != expected.resolve():
        raise ValueError('Turn must bind exactly one installed ' + SKILL_NAME)
    text = expected.read_text()
    if thread.get('baseInstructions') != text:
        raise ValueError('Thread baseInstructions differ from the installed sample-annotation skill')
    if thread.get('model') != expected_model or params.get('model') != expected_model:
        raise ValueError('Actual model differs from the selected research model')
    effort = params.get('effort')
    if not isinstance(effort, str) or not effort or (expected_effort is not None and effort != expected_effort):
        raise ValueError('Actual effort differs from the selected reasoning effort')
    if thread.get('environments') != [] or params.get('environments') != []:
        raise ValueError('Sealed hosted research must disable environment access')
    if thread.get('runtimeWorkspaceRoots') != [] or params.get('runtimeWorkspaceRoots') != []:
        raise ValueError('Sealed hosted research must disable runtime workspace access')
    return {'valid': True, 'skill_name': SKILL_NAME, 'skill_path': str(expected),
            'skill_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'model': expected_model, 'reasoning_effort': effort,
            'evidence_isolation_certified': False, 'training_started': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--thread-request', required=True, type=Path)
    parser.add_argument('--turn-request', required=True, type=Path)
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--effort', default=EFFORT)
    args = parser.parse_args()
    print(json.dumps(validate_binding(json.loads(args.thread_request.read_text()),
                                     json.loads(args.turn_request.read_text()),
                                     args.model, args.effort), indent=2))


if __name__ == '__main__':
    main()
