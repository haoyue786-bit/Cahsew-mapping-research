# run_info.py
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
from datetime import datetime
from typing import Any, Dict, Optional, Union


def _to_str(v: Any) -> str:
    if v is None:
        return "None"
    if isinstance(v, (list, tuple, dict)):
        s = str(v)
        return s if len(s) <= 240 else s[:240] + " ... (truncated)"
    s = str(v)
    return s if len(s) <= 320 else s[:320] + " ... (truncated)"


def _make_table(rows, headers=("Key", "Value")) -> str:
    col1 = [headers[0]] + [r[0] for r in rows]
    col2 = [headers[1]] + [r[1] for r in rows]
    w1 = max(len(x) for x in col1) if col1 else len(headers[0])
    w2 = max(len(x) for x in col2) if col2 else len(headers[1])

    def line():
        return f"+-{'-'*w1}-+-{'-'*w2}-+"

    def fmt(a, b):
        return f"| {a.ljust(w1)} | {b.ljust(w2)} |"

    out = [line(), fmt(headers[0], headers[1]), line()]
    for k, v in rows:
        out.append(fmt(k, v))
    out.append(line())
    return "\n".join(out)


def print_run_info(
    info: Union[Dict[str, Any], Dict[str, Dict[str, Any]]],
    *,
    title: str = "Run Info",
    save_path: Optional[str] = None,
    sort_keys: bool = False,
    print_time: bool = True,
) -> str:
    blocks = []
    header = f"==== {title} ===="
    if print_time:
        header += f"  ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})"
    blocks.append(header)

    is_grouped = bool(info) and all(isinstance(v, dict) for v in info.values())

    if is_grouped:
        for section, sec_dict in info.items():
            blocks.append(f"\n[{section}]")
            items = list(sec_dict.items())
            if sort_keys:
                items.sort(key=lambda x: x[0])
            rows = [(str(k), _to_str(v)) for k, v in items]
            blocks.append(_make_table(rows))
    else:
        items = list(info.items())
        if sort_keys:
            items.sort(key=lambda x: x[0])
        rows = [(str(k), _to_str(v)) for k, v in items]
        blocks.append(_make_table(rows))

    text = "\n".join(blocks)
    print(text)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")

    return text