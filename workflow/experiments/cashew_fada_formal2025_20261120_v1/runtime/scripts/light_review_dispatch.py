"""Lightweight, point-sealed per-epoch scientific review.

A light review may adjust only the named optimization controls; STOP, selection,
branch experiments and pseudo-negative changes require a full review. Any
ambiguous submitted turn is preserved and never automatically duplicated.
"""
import copy
import importlib.util
import json
import math
from pathlib import Path
from jsonschema import Draft202012Validator
from agent_account_route import apply_worker_route
from expert_policy_v4 import MODEL, EFFORT, flat, parameter_bounds
from on_demand_memory import write, read, sha
from online_boundary import verify_online
from persistent_luna import PersistentLunaClient
from two_tier_review_policy import light_observation

SKILL = Path('@WORKSPACE@/.codex_training_current_account/skills/cashew-fada-training-expert/SKILL.md')
ALLOWED_CONTROLS = ('main_lr', 'D_lr', 'lambda_adv')
SCHEMA = {'type':'object','additionalProperties':False,
 'required':['action','parameter_updates','observations','mechanism','expected_response','reversal_condition'],
 'properties':{'action':{'enum':['KEEP','ADJUST','DISABLE_AUTOMATIC_NEGATIVE']},
   'parameter_updates':{'type':'array','maxItems':3,'items':{'type':'object','additionalProperties':False,
      'required':['parameter','value'],'properties':{'parameter':{'enum':list(ALLOWED_CONTROLS)},'value':{'type':'number'}}}},
   **{k:{'type':'string','minLength':1} for k in ('observations','mechanism','expected_response','reversal_condition')}}}


def normalize_light(answer, config, epoch, request_hash):
    error = next(Draft202012Validator(SCHEMA).iter_errors(answer), None)
    if error:
        raise ValueError('Invalid light review: ' + error.message)
    bounds = parameter_bounds(config)
    before = flat(config)
    after = copy.deepcopy(before)
    seen = set()
    for item in answer['parameter_updates']:
        key, value = item['parameter'], item['value']
        if key in seen or key not in bounds or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError('Duplicate or invalid lightweight control')
        lo, hi = bounds[key]
        if not lo <= value <= hi:
            raise ValueError('Lightweight control outside permitted bounds')
        seen.add(key)
        after[key] = value
    changed = [key for key in seen if before[key] != after[key]]
    if (answer['action'] in {'KEEP','DISABLE_AUTOMATIC_NEGATIVE'} and seen) or (answer['action'] == 'ADJUST' and not changed):
        raise ValueError('Action and control changes do not agree')
    if answer['action'] == 'DISABLE_AUTOMATIC_NEGATIVE' and not (config.get('_automatic_negative_state') or {}).get('weight', 0) > 0:
        raise ValueError('No active automatic-negative pool to disable')
    return {'epoch':epoch, 'decision_kind':'light_review', 'request_sha256':request_hash,
        'action':answer['action'], 'continue_training':True,
        'automatic_negative_action': ('DISABLE' if answer['action']=='DISABLE_AUTOMATIC_NEGATIVE' else None),
        'analysis_summary':answer['observations'],
        'intervention':{'parameters_before':before,'parameters_after':after,
           'parameter_updates':answer['parameter_updates'],'changed_parameters':sorted(changed),
           'schedule_updates':[],'schedule_choices':[], 'adversarial_state_after':before['adversarial_state'],
           'emergency':False, 'rationale':answer['mechanism']},
        'expected_response':answer['expected_response'], 'reversal_condition':answer['reversal_condition']}


def light_request(epoch, metrics, config, history):
    observation = light_observation(epoch, metrics, {k:config[k] for k in ALLOWED_CONTROLS})
    if observation['emergency_signal']:
        raise ValueError('Nonfinite or invalid loss: stop worker for numerical diagnosis')
    pool = config.get('_automatic_negative_state') or {}
    observation['automatic_negative_pool'] = {k:pool[k] for k in ('weight','cursor','selected_patch_count') if k in pool}
    request = {'decision_kind':'light_review', 'epoch':epoch, 'current':observation,
               'recent_training_observations':history[-4:],
               'allowed_parameter_bounds':{k:parameter_bounds(config)[k] for k in ALLOWED_CONTROLS},
               'instruction':'Use only observed training dynamics. Base control values can differ from measured scheduled values; updates affect base controls and not the schedule family. KEEP when an intervention direction is unsupported; a reversible ADJUST requires observations, mechanism, expected response and reversal. Do not infer target accuracy. No normal STOP, model selection, new pseudo-negative pool or complete-map diagnosis in this light review. If evidence indicates a currently active automatic-negative pool is harmful, DISABLE_AUTOMATIC_NEGATIVE stops its next-epoch use; otherwise do not disable without evidence.'}
    verify_online(request)
    return request


def review(job, epoch, metrics, config, history):
    from legacy_agent_guard_v1 import block_legacy
    block_legacy(job, 'direct_light_review')
    import os
    stage=Path(job['output_root'])/'agent'/('epoch_%03d'%epoch)/'light_review'
    stage.mkdir(parents=True,exist_ok=True)
    request=light_request(epoch,metrics,config,history)
    request['job']={key:job[key] for key in ('job_id','run_id','country','target_year',
                    'source_year','two_tier_review_policy','deep_review_interval') if key in job}
    verify_online(request)
    req=stage/'request.json';decision=stage/'decision.json'
    if req.exists():
        if read(req)!=request:
            raise ValueError('Existing light review request differs; do not overwrite')
    else:write(req,request)
    if decision.exists():
        saved=read(decision)
        receipt=stage/'receipt.json'
        if not receipt.is_file():
            raise RuntimeError('Light decision lacks publication receipt; reconcile before replay')
        proof=read(receipt)
        if (saved.get('request_sha256')!=sha(req) or proof.get('request_sha256')!=sha(req)
                or proof.get('decision_sha256')!=sha(decision)):
            raise ValueError('Light decision publication identity mismatch')
        return saved
    if (stage/'submission.json').exists():
        raise RuntimeError('Unresolved light review submission: reconcile exact turn before retry')
    env=apply_worker_route(os.environ)
    frozen=Path(__file__).resolve().parents[1]/'.codex/skills/cashew-fada-training-expert/SKILL.md'
    if sha(frozen)!=sha(SKILL):raise ValueError('Installed Cashew skill differs from frozen release')
    spec=importlib.util.spec_from_file_location('cashew_light_binding', SKILL.parent/'scripts/verify_explicit_skill_binding.py')
    binding=importlib.util.module_from_spec(spec);spec.loader.exec_module(binding)
    class BoundClient(PersistentLunaClient):
        def _send(self,payload):
            if payload.get('method')=='turn/start':
                audit=binding.validate_binding(read(stage/'thread.json'),payload,expected_model=MODEL)
                write(stage/'actual_turn.json',payload);write(stage/'binding.json',audit)
            return super()._send(payload)
    client=BoundClient(env['CASHEW_FADA_CODEX_BINARY'],1800)
    try:
        client.start_server()
        thread={'baseInstructions':SKILL.read_text(),'cwd':str(stage),'model':MODEL,'sandbox':'read-only',
            'approvalPolicy':'never','ephemeral':False,
            'developerInstructions':'Lightweight Cashew review; only supplied point-sealed evidence. No external tools.',
            'config':{'project_doc_max_bytes':0,'features.skip_host_skill_discovery':True,
                'features.multi_agent':False,'features.shell_tool':False,'features.unified_exec':False,
                'features.apps':False,'web_search':'disabled'}}
        write(stage/'thread.json',thread)
        reply,_=client._request('thread/start',thread)
        tid=reply['thread']['id']
        write(stage/'submission.json',{'thread_id':tid,'request_sha256':sha(req)})
        prompt='$cashew-fada-training-expert\nReturn only the requested JSON. '+json.dumps(request,ensure_ascii=False,allow_nan=False)
        result=client.run_structured_turn(thread_id=tid,text=prompt,skill_name='cashew-fada-training-expert',
            skill_path=SKILL,output_schema=SCHEMA,model=MODEL,reasoning_effort=EFFORT,timeout_seconds=1800)
        write(stage/'raw_response.json',result)
        answer=json.loads(result['final_text'])
        value=normalize_light(answer,config,epoch,sha(req));write(decision,value)
        write(stage/'receipt.json',{'request_sha256':sha(req),'decision_sha256':sha(decision),'thread_id':tid,'turn_id':result['turn_id']})
        return value
    finally:client.close()
