"""Bounded paired continuation with actual epoch/review boundaries."""
from __future__ import annotations


def continue_arm(*, total_steps, steps_per_epoch, review_interval, train_one_epoch, review):
    if type(total_steps) is not int or total_steps <= 0 or total_steps % steps_per_epoch:
        raise ValueError("long paired horizon must contain complete actual epochs")
    epochs = total_steps // steps_per_epoch
    if epochs > 10 or epochs < review_interval:
        raise ValueError("long paired horizon must span one or two five-epoch reviews")
    observations = []
    metrics = None
    for offset in range(1, epochs + 1):
        metrics = train_one_epoch(offset)
        if metrics.get("steps") != steps_per_epoch:
            raise ValueError("paired epoch did not complete advertised updates")
        if offset % review_interval == 0 or offset == epochs:
            observation = review(offset, metrics)
            observation.update(completed_steps=offset * steps_per_epoch, elapsed_epochs=offset)
            observations.append(observation)
    return metrics, observations
