"""User-pinned minimum training horizon, independent of checkpoint selection."""
VERSION='user_minimum100_then_Agent_stop_v1'
def floor(job):
    value=job.get('minimum_stop_epoch',0)
    if type(value) is not int or value<0 or value>int(job.get('max_epoch',200)):raise ValueError('Invalid minimum training epoch')
    policy=job.get('minimum_training_policy')
    if policy and (policy.get('version')!=VERSION or policy.get('minimum_stop_epoch')!=value or not policy.get('user_authorized')):raise ValueError('Minimum training policy identity mismatch')
    return value
def guard(job,epoch,continue_training,source_collapsed=False):
    if not continue_training and int(epoch)<floor(job) and not source_collapsed:
        raise ValueError('Ordinary STOP before user minimum training epoch is forbidden')
    return True
