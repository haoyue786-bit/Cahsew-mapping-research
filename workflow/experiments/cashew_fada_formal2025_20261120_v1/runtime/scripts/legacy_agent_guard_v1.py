"""Fail-closed routing fence for the isolated experimental worker.

The experimental decision transaction and authenticated producer pipeline have
not yet been wired into main.  Consequently no experimental job may reach a
legacy Agent, legacy memory, or legacy joint-negative publication call.  This
fence is a second line of defence behind the launch block, not authorization to
run, and does not itself provide a sealed Agent decision.
"""

class UnwiredExperimentalDecision(RuntimeError):
    pass


def is_experimental(job):
    if not isinstance(job, dict):
        raise ValueError('worker job must be an object')
    # Presence is deliberate: a malformed null/empty policy must not bypass
    # the fence and accidentally inherit a legacy evidence path.
    return 'experimental_gate' in job


def block_legacy(job, entrypoint):
    if is_experimental(job):
        raise UnwiredExperimentalDecision(
            'experimental Agent path is not fully wired; legacy '+entrypoint+' forbidden')


def light_review(job, epoch, metrics, config, recent):
    block_legacy(job, 'light_review')
    from light_review_dispatch import review
    return review(job, epoch, metrics, config, recent)


def pool_prepare(job, controller, epoch, checkpoint, review, config):
    block_legacy(job, 'automatic_negative_prepare')
    return controller.prepare_joint(epoch, checkpoint, review, config)


def pool_refresh(job, controller, epoch, checkpoint, review, config):
    block_legacy(job, 'automatic_negative_refresh')
    return controller.refresh(epoch, checkpoint, review, config)
