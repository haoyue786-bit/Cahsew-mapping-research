#!/usr/bin/env python3
"""Offline validation of actual explicit Cashew thread/turn bindings."""
import argparse
import hashlib
import json
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parents[1]
SKILL_NAME = 'cashew-fada-training-expert'


def validate_binding(thread, turn, expected_model="gpt-6.2-sol", expected_effort=None):
    allowed_by_model = {'gpt-6.2-sol': {'max'}, 'gpt-6.1-sol': {'high', 'max'}, 'gpt-6-luna': {'high'}}
    if expected_model not in allowed_by_model:
        raise ValueError("Unsupported explicitly bound Cashew model")
    params = turn.get('params', turn)
    entries = [x for x in params.get('input', []) if x.get('type') == 'skill']
    matches = [x for x in entries if x.get('name') == SKILL_NAME]
    if len(matches) != 1:
        raise ValueError('Actual turn must explicitly include exactly one '+SKILL_NAME+' skill input')
    if any(x.get('name') == 'deep-learning-training-expert' for x in entries):
        raise ValueError('Generic training entry must not replace the Cashew entry')
    expected_path = SKILL_DIR/'SKILL.md'
    if Path(matches[0].get('path', '')).resolve() != expected_path.resolve():
        raise ValueError('Actual turn points to a different Cashew skill installation')
    text = expected_path.read_text()
    if thread.get('baseInstructions') != text:
        raise ValueError('Thread baseInstructions differ from the installed Cashew skill')
    if thread.get('model') != expected_model or params.get('model') != expected_model:
        raise ValueError('Thread and turn must bind the explicitly selected model')
    effort = params.get('effort')
    allowed = allowed_by_model[expected_model]
    if effort not in allowed:
        raise ValueError('Unsupported explicitly bound reasoning effort')
    if expected_effort is not None and (expected_effort not in allowed or effort != expected_effort):
        raise ValueError('Actual turn reasoning effort differs from the selected effort')
    return dict(valid=True, skill_name=SKILL_NAME, skill_path=str(expected_path),
                skill_sha256=hashlib.sha256(text.encode()).hexdigest(),
                model=expected_model, reasoning_effort=effort,
                evidence_isolation_certified=False, training_started=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--thread-request', required=True, type=Path)
    parser.add_argument('--turn-request', required=True, type=Path)
    parser.add_argument('--model', choices=['gpt-6.2-sol','gpt-6.1-sol','gpt-6-luna'], default='gpt-6.2-sol')
    parser.add_argument('--effort', choices=['high', 'max'], help='Require this effort for a new call; omit when auditing historical turns')
    args = parser.parse_args()
    print(json.dumps(validate_binding(json.loads(args.thread_request.read_text()),
                                     json.loads(args.turn_request.read_text()), args.model, args.effort), indent=2))


if __name__ == '__main__':
    main()
