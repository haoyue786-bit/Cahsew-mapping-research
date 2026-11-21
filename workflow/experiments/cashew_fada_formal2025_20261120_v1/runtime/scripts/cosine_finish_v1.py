"""Agent-triggered, checkpointed cosine finish; no automatic STOP or restart."""
import copy
import math

VERSION = 'agent_cosine_finish_v1'
CONTROL = 'main_lr_finish_epochs'
OPERATION = 'start_cosine_finish'


def enabled(config):
    return (config.get('cosine_finish_policy') or {}).get('version') == VERSION


def bounds(config):
    return {CONTROL: [5, 30]} if enabled(config) else {}


def initialize(config, job):
    policy = job.get('cosine_finish_policy')
    if not policy:
        return
    if policy.get('version') != VERSION:
        raise ValueError('unsupported cosine finish policy')
    factor = policy.get('minimum_factor', .1)
    if isinstance(factor, bool) or not isinstance(factor, (int, float)) or not 0 < factor < 1:
        raise ValueError('cosine finish minimum factor must be in (0,1)')
    config['cosine_finish_policy'] = copy.deepcopy(policy)
    config[CONTROL] = 15


def validate(config):
    if not enabled(config):
        return
    n = config.get(CONTROL)
    if type(n) is not int or not 5 <= n <= 30:
        raise ValueError('cosine finish duration must be an integer in [5,30]')


def start(config, epoch, current_lr):
    validate(config)
    if not enabled(config) or config.get('cosine_finish_state'):
        raise ValueError('cosine finish is unavailable or already used')
    maximum = int(config['cosine_finish_policy'].get('maximum_epoch', 200))
    duration = min(config[CONTROL], maximum-int(epoch))
    if epoch <= 0 or duration <= 0 or not math.isfinite(current_lr) or current_lr <= 0:
        raise ValueError('cosine finish requires a trained state and remaining budget')
    config['cosine_finish_state'] = {
        'start_epoch': int(epoch), 'duration_epochs': duration,
        'requested_duration_epochs': config[CONTROL],
        'start_lr': float(current_lr),
        'end_lr': float(current_lr) * config['cosine_finish_policy'].get('minimum_factor', .1),
        'restart_allowed': False,
    }


def value(config, epoch, ordinary_lr):
    state = config.get('cosine_finish_state') if enabled(config) else None
    if not state:
        return ordinary_lr
    p = max(0., min(1., (float(epoch)-state['start_epoch']) / state['duration_epochs']))
    return state['end_lr'] + (state['start_lr']-state['end_lr']) * .5*(1+math.cos(math.pi*p))


def rebase(config, before):
    """A later explicit base-LR change rescales the active finish as well."""
    state = config.get('cosine_finish_state')
    if enabled(config) and state and before['main_lr'] != config['main_lr']:
        ratio = config['main_lr'] / before['main_lr']
        state['start_lr'] *= ratio
        state['end_lr'] *= ratio


def packet(packet, job):
    if enabled(job):
        packet['cosine_finish_capability'] = {
            'version': VERSION, 'operation': OPERATION,
            'component': 'main_lr', 'phase': None, 'duration_control': CONTROL,
            'minimum_factor': job['cosine_finish_policy'].get('minimum_factor', .1),
            'instruction': 'When marginal benefit weakens but smaller updates could still help, choose a duration and start one smooth cosine finish from the current effective main learning rate. Reassess at ordinary reviews; no warm restart or automatic repeated extension. Duration is clipped to the remaining epoch budget. Check discriminator balance separately. The finish does not guarantee target accuracy or an optimum. STOP and checkpoint SELECT remain separate Agent decisions.',
        }
    return packet
