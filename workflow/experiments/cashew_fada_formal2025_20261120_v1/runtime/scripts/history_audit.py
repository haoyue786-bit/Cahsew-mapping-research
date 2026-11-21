"""Verify and commit dynamic history tool transcript before publishing a decision."""
import hashlib,json
from pathlib import Path
from journal_history import fetch
from history_tool_client import valid_args

class HistoryQueryRequired(RuntimeError):
    """The model finished without inspecting the required same-run history."""

def verify(stage, *, root, job_id, run_id, cutoff_epoch, thread_id, require_history=False, allowed_turn_ids=None, selection_request=None, selection_answer=None):
    if type(require_history) is not bool:raise ValueError('invalid history audit policy')
    if require_history and (not isinstance(allowed_turn_ids,(list,tuple)) or not allowed_turn_ids or
                            any(not isinstance(x,str) or not x for x in allowed_turn_ids) or len(set(allowed_turn_ids))!=len(allowed_turn_ids)):
        raise ValueError('required history must be bound to completed turn identities')
    from query_progress_v1 import numbered_files, verify_progress
    stage=Path(stage); files=numbered_files(stage, 'tool_call_')
    verify_progress([json.loads(p.read_text()) for p in files])
    hashes=[];initialization_seen=False;prior_record_seen=False;coverage={};ledger_seen=set()
    for i,p in enumerate(files,1):
        if p.name != 'tool_call_%02d.json'%i or p.is_symlink() or not p.is_file():raise ValueError('invalid tool transcript')
        raw=p.read_bytes();record=json.loads(raw);call=record['request'];params=call['params'];args=params['arguments']
        from training_tool_output_v1 import verify_delivery
        verify_delivery(record,'query_training_history',i)
        if call.get('method')!='item/tool/call' or params.get('tool')!='query_training_history' or params.get('threadId')!=thread_id:
            raise ValueError('unexpected tool identity')
        if require_history and params.get('turnId') not in allowed_turn_ids:
            raise ValueError('history query does not belong to the audited completed turn')
        if valid_args(args,cutoff_epoch):
            actual=fetch(root,job_id=job_id,run_id=run_id,cutoff_epoch=cutoff_epoch,start_epoch=args['start_epoch'],end_epoch=args['end_epoch'],kinds=args['kinds'])
            from online_boundary import verify_online
            verify_online(actual)
            if require_history:
                from online_boundary import isolate_history_text
                actual=isolate_history_text(actual)
        else:
            actual={'error':'Invalid query. Use prior epochs, the advertised evidence kinds, and at most 40 epochs per request.'}
        if actual != record['result']:raise ValueError('history response differs from authenticated records')
        if valid_args(args,cutoff_epoch):
            for item in actual.get('records',[]):
                if not isinstance(item,dict):raise ValueError('invalid audited history record')
                if item.get('epoch')==0 and isinstance(item.get('initialization'),dict):initialization_seen=True
                if (type(item.get('epoch')) is int and 0<item['epoch']<cutoff_epoch and
                    any(item.get(k) is not None for k in ('training','controls','source','unlabelled_panel','action','reasoning','review','queries','candidate','candidate_details','human_training','annotated_training'))):
                    prior_record_seen=True
                seen=coverage.setdefault(item.get('epoch'),set())
                if isinstance(item.get('interventions'),dict):ledger_seen.add(item['epoch'])
                for name in ('training','controls','source','unlabelled_panel','human_training'):
                    if isinstance(item.get(name),dict) and item[name]:
                        if name!='human_training' or isinstance(item[name].get('measurement'),dict):seen.add(name)
                if isinstance(item.get('candidate_details'),dict):
                    from checkpoint_selection_contract_v2 import validate_detail
                    detail=validate_detail(item['candidate_details'])
                    for name,field in (('training','training'),('controls','controls'),('source','source_validation'),('unlabelled_panel','unlabelled_panel'),('human_training','human_training'),('annotated_training','annotated_training')):
                        if detail.get(field):seen.add(name)
        hashes.append(hashlib.sha256(raw).hexdigest())
    if require_history and (not initialization_seen or (cutoff_epoch>1 and not prior_record_seen)):
        raise HistoryQueryRequired('Agent must retrieve the initialization and at least one earlier committed record when available')
    request=selection_request or {}
    if require_history and 'intervention_review' in request.get('schema',{}).get('properties',{}):
        earlier=[]
        for e in range(1,cutoff_epoch):
            path=Path(root)/'journal'/('epoch_%06d'%e)/'commit.json'
            if path.exists() and json.loads(path.read_text()).get('action')=='ADJUST':earlier.append(e)
        if earlier:
            last=earlier[-1]
            if last not in ledger_seen:raise HistoryQueryRequired('Retrieve interventions for latest ADJUST at epoch %d'%last)
            if selection_answer is not None and last not in {x['origin_epoch'] for x in selection_answer.get('intervention_review',[])}:
                raise HistoryQueryRequired('Assess latest ADJUST in intervention_review; pending is valid when its response is not yet observable')
    if selection_request and selection_request.get('selection_policy'):
        from checkpoint_selection_contract_v2 import validate_comparison
        if not require_history:raise ValueError('complete selection requires audited history')
        comparison=validate_comparison(selection_request,selection_answer)
        for epoch in {0,selection_answer['selected_epoch'],comparison['strongest_alternative']['epoch']}:
            active = 'annotated_training' in selection_request['candidate_details'][0]['values']
            required={'source','unlabelled_panel','controls', 'annotated_training' if active else 'human_training'}
            if epoch:required.add('training')
            if required-coverage.get(epoch,set()):
                raise HistoryQueryRequired('SELECT must retrieve candidate_details (or equivalent complete evidence) for E0, the selected epoch and the strongest alternative; missing epoch %d kinds %s'%(epoch,sorted(required-coverage.get(epoch,set()))))
    from paired_trial_v1 import verify_transcripts
    hashes += verify_transcripts(stage,root,cutoff_epoch,selection_request or {},thread_id,allowed_turn_ids or [])
    from training_tool_output_v1 import verify_read_transcripts
    hashes += verify_read_transcripts(stage,thread_id,allowed_turn_ids or [])
    return hashlib.sha256(json.dumps(hashes,separators=(',',':')).encode()).hexdigest()
