"""Deterministic, checkpointed schedules for the formal v4 controller.

Base controls remain Agent-tunable.  A phase is a fraction of the fixed
100-epoch horizon; review decisions may hold it or restart its progression
from a chosen earlier/later phase without changing the schedule family.
"""

from __future__ import annotations

import math
import supplementary_control_v4 as negative
import cosine_finish_v1 as finish


CURVES = {
    "main_lr": {"constant", "cosine", "poly"},
    "D_lr": {"constant", "cosine", "poly"},
    "lambda_adv": {"constant", "dann_sigmoid"},
}
HORIZON = 100


def default_state():
    return {name: {"curve": "constant", "mode": "advance", "anchor_epoch": 0,
                   "phase_at_anchor": 0.0} for name in CURVES}


def phase(state, epoch):
    value = float(state["phase_at_anchor"])
    if state["mode"] == "advance":
        value += max(0, int(epoch) - int(state["anchor_epoch"])) / HORIZON
    return min(1.0, max(0.0, value))


def multiplier(curve, phase_value):
    p = min(1.0, max(0.0, float(phase_value)))
    if curve == "constant":
        return 1.0
    if curve == "cosine":
        return 0.1 + 0.9 * (1.0 + math.cos(math.pi * p)) / 2.0
    if curve == "poly":
        return 0.1 + 0.9 * (1.0 - p) ** 0.9
    if curve == "dann_sigmoid":
        # Normalize DANN's logistic ramp so p=0 gives exactly zero and p=1 one.
        raw = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0
        end = 2.0 / (1.0 + math.exp(-10.0)) - 1.0
        return raw / end
    raise ValueError(f"unknown schedule curve: {curve}")


def effective(config, epoch):
    states = config.get("schedule_state") or default_state()
    values = {name: float(config[name]) * multiplier(states[name]["curve"], phase(states[name], epoch))
              for name in CURVES}
    values['main_lr'] = finish.value(config, epoch, values['main_lr'])
    return values


def apply_to_optimizers(config, epoch, model_optimizer, local_optimizer, global_optimizer):
    from transfer_parameter_policy import apply
    apply(config, model_optimizer)
    values = effective(config, epoch)
    for optimizer, lr, decay in (
        (model_optimizer, values["main_lr"], config["main_weight_decay"]),
        (local_optimizer, values["D_lr"], config["D_weight_decay"]),
        (global_optimizer, values["D_lr"], config["D_weight_decay"]),
    ):
        for group in optimizer.param_groups:
            group["lr"] = float(lr) * float(group.get("lr_multiplier", 1.0))
            group["weight_decay"] = float(decay)
    return values


def update_state(config, intervention, epoch):
    states = config.setdefault("schedule_state", default_state())
    choices = intervention.get("schedule_choices", [])
    updates = intervention.get("schedule_updates", [])
    if choices:
        for item in choices:
            states[item["component"]] = {"curve": item["curve"], "mode": "advance",
                                          "anchor_epoch": 0, "phase_at_anchor": 0.0}
    for item in updates:
        name = item["component"]
        if item['operation'] == finish.OPERATION:
            if name != 'main_lr' or item['phase'] is not None:
                raise ValueError('cosine finish is a main_lr operation with phase=null')
            finish.start(config, epoch, effective(config, epoch)['main_lr'])
            continue
        if name == negative.WEIGHT:
            negative.update_state(config, item, epoch)
            continue
        old = states[name]
        operation = item["operation"]
        next_phase = phase(old, epoch) if operation not in {"set_phase", "set_phase_hold"} else float(item["phase"])
        states[name] = {"curve": old["curve"],
                        "mode": "hold" if operation in {"hold", "set_phase_hold"} else "advance",
                        "anchor_epoch": int(epoch), "phase_at_anchor": next_phase}
