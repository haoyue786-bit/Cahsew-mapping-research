"""On-demand intervention ledger derived solely from committed same-run facts."""
from pathlib import Path
import json

VERSION='complete_training_recall_v3'

def ledger(root,job_id,run_id,epoch,cutoff):
    from complete_history_store import _transaction,_read_json,_identity,_csv_view
    root=Path(root);_identity(root,job_id,run_id)
    commit=_read_json(root/'journal'/('epoch_%06d'%epoch)/'commit.json')
    if commit.get('decision_kind')!='agent':return None
    route=commit['route'];values,txn=_transaction(root,job_id,run_id,epoch,route)
    answer=values['response'];decision=values['decision'];intervention=decision.get('intervention',{})
    record=dict(version=VERSION,origin_epoch=epoch,origin_action=decision['action'],
        applied_after_epoch=epoch,decision_sha256=txn['decision_sha256'],
        parameters_before=intervention.get('parameters_before'),parameters_after=intervention.get('parameters_after'),
        changed_parameters=intervention.get('changed_parameters'),schedule_updates=intervention.get('schedule_updates'),
        observations=answer.get('observations'),diagnosis=answer.get('diagnosis'),alternative=answer.get('alternative'),
        expected_response=answer.get('expected_response'),reversal_condition=answer.get('reversal_condition'),
        prospective_hypotheses=answer.get('response_hypotheses'),prior_intervention_assessment=answer.get('intervention_review'),
        subsequent_observations=[],subsequent_agent_assessments=[],
        interpretation='Same-run temporal response, not causal proof. Restoring a control does not restore old weights or moments.')
    for e in range(epoch+1,cutoff):
        cp=root/'journal'/('epoch_%06d'%e)/'commit.json'
        if not cp.exists():continue
        snap=_read_json(root/'evidence_store/snapshots'/('epoch_%06d.json'%e))
        obsfile=root/'journal'/('epoch_%06d'%e)/'review_observation.json'
        obs=_read_json(obsfile) if obsfile.exists() else {}
        train=snap['training'];human=obs.get('human_training') or {}
        record['subsequent_observations'].append(dict(epoch=e,controls=snap['controls'],
            cls=train.get('cls'),regularization_loss=train.get('regularization_loss'),
            actual_exposure=train.get('supplementary_actual_exposure'),source=obs.get('source'),panel=obs.get('panel'),
            human_training_probe=human.get('probe'),module_detail_available='module_observation' in train,
            module_gradients_available='module_gradient_probes' in train))
        c=_read_json(cp)
        if c.get('decision_kind')=='agent':
            v,_=_transaction(root,job_id,run_id,e,c['route'])
            for assessment in v['response'].get('intervention_review') or []:
                if assessment['origin_epoch']==epoch:record['subsequent_agent_assessments'].append(dict(review_epoch=e,**assessment))
    # Response evidence grows as later committed epochs become available.
    # Keep each query's temporal view immutable without replacing the original
    # action or changing the evidence returned to historical callers.
    _csv_view(root/'evidence_store/interventions'/('epoch_%06d_before_%06d.csv'%(epoch,cutoff)),record)
    return record
