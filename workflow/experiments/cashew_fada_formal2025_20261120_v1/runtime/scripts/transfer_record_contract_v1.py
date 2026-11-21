"""Strict, point-free transfer identity for initialization evidence."""
import re

FIELDS={'version','source_job_id','source_run_id','source_period','source_epoch','source_checkpoint_sha256','selection_decision_sha256','model_state_sha256','target_job_id','target_run_id','target_period','target_gid','architecture','optimizer_and_discriminators_transferred','provenance_certified'}

def validate(record):
    if not isinstance(record,dict) or set(record)!=FIELDS:raise ValueError('invalid transfer identity fields')
    if (record['version']!='two_period_frozen_handoff_candidate_v1' or record['source_period']!='2024-2025' or record['target_period']!='2018-2019' or record['optimizer_and_discriminators_transferred'] is not False or record['provenance_certified'] is not False):raise ValueError('invalid transfer semantics')
    if type(record['source_epoch']) is not int or not 0<=record['source_epoch']<=200:raise ValueError('invalid transfer epoch')
    for k in ('source_checkpoint_sha256','selection_decision_sha256','model_state_sha256'):
        if not isinstance(record[k],str) or not re.fullmatch('[0-9a-f]{64}',record[k]):raise ValueError('invalid transfer digest')
    for k in ('source_job_id','source_run_id','target_job_id','target_run_id','target_gid','architecture'):
        if not isinstance(record[k],str) or not re.fullmatch('[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*',record[k]) or len(record[k])>160 or any(part in ('.','..') for part in record[k].split('/')):raise ValueError('invalid transfer identifier')
    return dict(record)


def model_digest(state):
    """Hash each named tensor's type, shape and bytes, independent of device."""
    import hashlib,json,torch
    if not isinstance(state,dict) or not state or any(not isinstance(k,str) or not isinstance(v,torch.Tensor) for k,v in state.items()):
        raise ValueError('invalid model state for transfer digest')
    h=hashlib.sha256()
    for key in sorted(state):
        tensor=state[key].detach().cpu().contiguous()
        header=json.dumps([key,str(tensor.dtype),list(tensor.shape)],separators=(',',':')).encode()
        data=tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        for value in (header,data):
            h.update(len(value).to_bytes(8,'big'));h.update(value)
    return h.hexdigest()
