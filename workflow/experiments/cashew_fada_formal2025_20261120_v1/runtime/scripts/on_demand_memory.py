"""Snapshot-bound evidence retrieval: no LLM, no NAS traversal, no future results."""
from pathlib import Path
import copy,json,hashlib,math,datetime
from negative_usage_facts import build_usage,compact_usage

def read(p):return json.loads(Path(p).read_text())
def compact(v):return json.dumps(v,ensure_ascii=False,separators=(',',':'),allow_nan=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n');q.replace(p)
def pick(d,keys):return {k:d[k] for k in keys if k in d}

def metrics(review):
 if not review:return {}
 r=review.get('review',review);m=r.get('metrics') or {};human=r.get('human_supervision_evaluation') or {};target=r.get('target_unsupervised',r.get('target',{})) or {}
 result=pick(r,['epoch','source_class2_f1','target_class2_fraction','adversarial_state'])
 if 'source_class2_f1' not in result:
  src=r.get('source_validation',r.get('source',{})) or {};f=src.get('f1_per_class');result['source_class2_f1']=f[2] if isinstance(f,list) and len(f)>2 else None
 if 'target_class2_fraction' not in result:
  arr=target.get('class_distribution');result['target_class2_fraction']=arr[2] if isinstance(arr,list) and len(arr)>2 else None
 result['training']=pick(m,['cls','seg_cls','adv','D_loss','local_D_acc','global_D_acc','effective_gradient_ratio','steps','supp_unique_samples','supp_scheduled_samples','supp_selected_pixels'])
 result['actual_effective_controls']=m.get('scheduled_controls')
 result['human_exposure']=m.get('polygon_supervision')
 result['automatic_negative_exposure']=pick(m.get('automatic_negative') or {},['updates','patch_occurrences','unique_patches','pixel_occurrences','selected_patch_count','configured_weight','effective_weight','generation_start_cursor','generation_end_cursor'])
 g=m.get('gradient_direction') or {};result['sampled_gradient']=pick(g,['measured_steps','completed_steps','opposing_steps','source_adversarial_cosine_median','weighted_opposition_relative_to_source_median','scope'])
 result['human_training_fit_not_independent']={key:pick(human.get(key,{}).get('aggregate',{}),['positive_pixels','negative_pixels','false_positive_pixels','false_negative_pixels','true_positive_pixels','false_positive_rate','cashew_recall','cashew_f1']) for key in ['training_difficulty_only'] if isinstance(human.get(key),dict)}
 result['meaning']='source F1 is not target accuracy; area/temporal overlap are predictions, not target labels; human feedback is supplementary TRAINING FIT, not independent target accuracy; absent metrics are unknown; gradient norms are sampled'
 return result

def trajectory_row(epoch,record):
 m=metrics(record);t=m.get('training') or {};h=m.get('human_exposure') or {};a=m.get('automatic_negative_exposure') or {}
 return dict(epoch=epoch,source_class2_f1=m.get('source_class2_f1'),predicted_class2_fraction=m.get('target_class2_fraction'),**{k:t.get(k) for k in ['cls','adv','D_loss','effective_gradient_ratio']},effective_controls=m.get('actual_effective_controls'),human_negative_pixel_occurrences=h.get('negative_pixel_occurrences'),human_positive_pixel_occurrences=h.get('positive_pixel_occurrences'),automatic_patch_occurrences=a.get('patch_occurrences'))

def exposure_log(job_dir,cutoff):
 """Recover printed actual exposure counts; epoch is inferred from log order.

 Fresh-only worker: each committed generation begins after its review epoch;
 exactly one exposure summary follows each completed ordinary epoch. Retain
 that provenance distinction instead of calling an expected schedule observed.
 """
 p=Path(job_dir)/'training.log';records=[];generation=None;epoch=None
 if not p.is_file():return records
 for line in p.read_text().splitlines():
  try:d=json.loads(line)
  except (ValueError,TypeError):continue
  if not isinstance(d,dict):continue
  if d.get('automatic_stage')=='Agent_pool_committed':generation=int(d['epoch']);epoch=generation
  elif d.get('automatic_stage')=='actual_training_exposure' and epoch is not None:
   epoch+=1
   if epoch<=cutoff:records.append(dict(epoch=epoch,generation_epoch=generation,epoch_identity='inferred_from_committed_generation_and_sequential_main_epoch_log; no worker restarts in this fresh run',record=d,meaning='Printed ACTUAL update/patch/pixel counts, not a promised schedule; per-patch identities are not in this log.'))
 return records

def evidence_snapshot(packet,request_path,job_dir,destination,negative=None):
 """Read only this group's observed reviews/accepted decisions strictly before action."""
 packet=copy.deepcopy(packet);epoch=int(packet.get('epoch',0));job_dir=Path(job_dir);dest=Path(destination)
 if (dest/'manifest.json').exists():
  old=read(dest/'manifest.json')
  # A request can be normalized during preparation (for example when the
  # current packet's descriptive boundary is corrected).  In that case the
  # existing evidence store belongs to the pre-normalization request and must
  # be rebuilt for the new immutable request hash.  Treating this as an
  # assertion failure made an otherwise valid review abort at epoch 5.
  if old.get('request_sha256')==sha(request_path):
   return old
 current=packet.get('current_review',packet.get('baseline_review',{}));reviews=[];sources=[];decisions=[]
 for p in sorted(job_dir.glob('review_epoch_*.json')):
  e=int(p.stem.rsplit('_',1)[1])
  if e<=epoch:
   d=read(p);reviews.append(dict(epoch=e,record=d));sources.append(dict(path=str(p),sha256=sha(p)))
 # Include only prior accepted decisions. The current request already supplies
 # same-epoch previous experiment feedback when legal; do not open future calls.
 for p in sorted((job_dir/'agent').rglob('decision.json')):
  if p.is_file():
   d=read(p);e=d.get('epoch',0)
   if isinstance(e,int) and e<epoch:decisions.append(d);sources.append(dict(path=str(p),sha256=sha(p)))
 decisions.sort(key=lambda d:(int(d.get('epoch',0)),d.get('decision_kind')=='experiment_review'))
 past_experiments=[]
 for folder in sorted((job_dir/'experiments').glob('epoch_*/*')):
  if not folder.is_dir():continue
  e=int(folder.parent.name.rsplit('_',1)[1])
  if e>=epoch:continue
  record=dict(epoch=e,experiment_id=folder.name,artifacts={})
  for name in ['manifest.json','comparison.json','result.json','failure.json','error.json']:
   p=folder/name
   if p.is_file():record['artifacts'][name]=read(p);sources.append(dict(path=str(p),sha256=sha(p)))
  if record['artifacts']:past_experiments.append(record)
 generations=[]
 for p in sorted((job_dir/'automatic_negative').glob('epoch_*/manifest.json')):
  e=int(p.parent.name.rsplit('_',1)[1])
  if e<epoch:
   d=read(p);entry=pick(d,['generation_epoch','selected_patch_count','selected_unique_pixels','agent_selected_weight','requested_patch_count','ranking_weights','exposure_horizon_updates','source_checkpoint_sha256']);entry['epoch']=e
   q=p.with_name('selection_decision.json')
   if q.is_file():entry['accepted_selection']=read(q);sources.append(dict(path=str(q),sha256=sha(q)))
   generations.append(entry);sources.append(dict(path=str(p),sha256=sha(p)))
 documents={
 'current':current,
 'controls':pick(packet,['current_config','historical_anchor_config','allowed_parameter_bounds','gradient_control_capability','cosine_finish_capability']),
 'job_and_permissions':pick(packet,['job','fada_variant','information_boundary','source_protection_policy','safety_envelope','early_stopping_policy','allowed_actions','instruction','experiment_capability']),
 'history':reviews,
 'accepted_decisions':dict(meaning='Accepted proposals; verify actual application using next measured controls/exposure. A saved decision alone is not proof of execution.',records=decisions,action_score_history=packet.get('action_score_history',[]),previous_decision=packet.get('previous_agent_decision')),
 'trajectory':[trajectory_row(x['epoch'],x['record']) for x in reviews],
 'prior':{k:v for k,v in packet.items() if 'prior' in k or k in ['area_alert_evidence','user_reported_target_behavior']},
 'auxiliary':pick(packet,['supplementary_training','conditional_negative_evidence','automatic_negative_evidence','negative_response_guidance']),
 'negative_generations':dict(meaning='Earlier committed manifests are proposals, not proof of exposure. Actual exposure history remains available after the finite pool expires; zero in the latest epoch does not mean it was never used.',records=generations,actual_exposure_by_epoch=exposure_log(job_dir,epoch)),
 'experiments':dict(**pick(packet,['experiment_capability','experiment_result','intervention_evidence','next_review_evidence']),past_records=past_experiments,meaning='Complete saved earlier branch outcomes, including failed/null/adverse results; same-epoch result only from supplied current packet. Branch weights never adopted.'),
 'initialization':pick(packet,['pre_run','baseline_review','historical_anchor_config']),
 'selection':{k:v for k,v in packet.items() if 'candidate' in k or 'selection' in k},
 'complete_packet':packet,
 }
 observed=[]
 for p in sorted((job_dir/'epoch_facts').glob('epoch_*.json')):
  d=read(p)
  if int(d['epoch'])<=epoch:observed.append(d);sources.append(dict(path=str(p),sha256=sha(p)))
 documents['epoch_observations']=observed
 documents['automatic_negative_usage']=build_usage(epoch,[x['epoch'] for x in reviews],documents['negative_generations']['actual_exposure_by_epoch'],observed,reviews,current)
 documents['negative_generations']['measured_interval_usage']=documents['automatic_negative_usage']['since_previous_review']
 log_path=job_dir/'training.log'
 if log_path.is_file():sources.append(dict(path=str(log_path),sha256=sha(log_path),role='actual exposure lines, temporal filter applied before publication'))
 library=Path(__file__).resolve().parents[1]/'.codex/skills/cashew-fada-training-expert/references'
 allowed=['online-evidence-and-sol-refinement.md','on-demand-group-memory.md','negative-availability-current.md','experiment-library.md','schedule-and-stopping.md','runtime-contract.md']
 documents['knowledge']={n:(library/n).read_text() for n in allowed}
 from online_boundary import verify_online
 if negative is not None:documents['negative_candidates']=negative
 verify_online(documents)
 paths={}
 for key,value in documents.items():
  p=dest/(key+'.json');write(p,value);paths[key]=dict(file=p.name,sha256=sha(p),characters=len(compact(value)),kind=type(value).__name__,items=len(value) if isinstance(value,(list,dict)) else None)
 manifest=dict(version='on_demand_own_group_evidence_v1',job_id=packet.get('job',{}).get('job_id'),run_id=packet.get('job',{}).get('run_id'),epoch=epoch,decision_kind=packet.get('decision_kind'),request_sha256=sha(request_path),documents=paths,sources=sources,temporal_boundary='Only own-run observed records <=epoch; accepted prior actions <epoch; current supplied packet exact. No baseline outcomes, target labels or posthoc tests.',raw_history_not_read_by_agent_yet=True)
 write(dest/'manifest.json',manifest);return manifest

def query(store,arguments):
 store=Path(store);manifest=read(store/'manifest.json');allowed={'topic','pointer','start_epoch','end_epoch','offset','limit','char_offset','char_limit'}
 if not isinstance(arguments,dict) or set(arguments)-allowed:raise ValueError('Invalid query arguments')
 topic=arguments.get('topic','catalog')
 if topic=='catalog':return dict(epoch=manifest['epoch'],documents=manifest['documents'],boundary=manifest['temporal_boundary'])
 if topic not in manifest['documents']:raise ValueError('Unknown evidence topic; use catalog')
 record=manifest['documents'][topic];p=store/record['file']
 if p.resolve().parent!=store.resolve() or not p.is_file() or sha(p)!=record['sha256']:
  raise ValueError('Evidence identity changed')
 value=read(p)
 from online_boundary import verify_online
 verify_online(value)
 pointer=arguments.get('pointer','')
 if pointer:
  if not pointer.startswith('/'):raise ValueError('Use a JSON pointer beginning with /')
  for token in pointer[1:].split('/'):
   token=token.replace('~1','/').replace('~0','~');value=value[int(token)] if isinstance(value,list) else value[token]
 start=arguments.get('start_epoch');end=arguments.get('end_epoch')
 if end is not None and end>manifest['epoch']:raise ValueError('Future epoch prohibited')
 if start is not None and start>manifest['epoch']:raise ValueError('Future epoch prohibited')
 char_paging='char_offset' in arguments or 'char_limit' in arguments
 if char_paging and (not isinstance(value,str) or any(k in arguments for k in ('offset','limit','start_epoch','end_epoch'))):
  raise ValueError('char_offset/char_limit apply only to string values and cannot be combined with other paging; for objects omit them and use a JSON pointer; for arrays use offset/limit. char_limit must be 1..2500.')
 if isinstance(value,list):
  if start is not None or end is not None:
   def event_epoch(v):return v.get('epoch',v.get('decision_epoch',v.get('generation_epoch')))
   value=[v for v in value if isinstance(v,dict) and isinstance(event_epoch(v),int) and (start is None or event_epoch(v)>=start) and (end is None or event_epoch(v)<=end)]
  count=len(value);offset=arguments.get('offset',0);limit=arguments.get('limit',20)
  if type(offset)is not int or type(limit)is not int or offset<0 or not 1<=limit<=100:raise ValueError('Invalid page; limit1..100, selection count remains uncapped')
  value=value[offset:offset+limit];page=dict(offset=offset,returned=len(value),total=count,next_offset=offset+len(value) if offset+len(value)<count else None)
 elif isinstance(value,str):
  if char_paging:
   offset=arguments.get('char_offset',0);limit=arguments.get('char_limit',2500)
   if type(offset)is not int or type(limit)is not int or offset<0 or not 1<=limit<=2500:raise ValueError('Invalid character page')
   total=len(value);value=value[offset:offset+limit];page=dict(unit='characters',offset=offset,returned=len(value),total=total,next_offset=offset+len(value) if offset+len(value)<total else None)
  else:
   lines=value.splitlines();offset=arguments.get('offset',0);limit=arguments.get('limit',40)
   if type(offset)is not int or type(limit)is not int or offset<0 or not 1<=limit<=100:raise ValueError('Invalid text page')
   value='\n'.join(lines[offset:offset+limit]);page=dict(unit='lines',offset=offset,returned=min(limit,max(0,len(lines)-offset)),total=len(lines),next_offset=offset+limit if offset+limit<len(lines) else None)
 else:page=None
 result=dict(topic=topic,epoch_cutoff=manifest['epoch'],source_sha256=record['sha256'],pointer=pointer,page=page,value=value)
 # Refuse accidental whole-history dumps. Narrow pointer/range or page; no
 # silent clipping, no hidden loss of numeric fields or old negative outcomes.
 if len(compact(result))>22000:
  help_text='Request a specific JSON pointer, epoch interval or smaller page. Full evidence remains available.'
  if topic=='negative_generations':help_text+=' For actual training use, query automatic_negative_usage /since_previous_review or /by_epoch. /records contains selected pools, not actual use.'
  # Give actionable paths without returning (or truncating) evidence values.
  # Pointer escaping matters for keys containing a slash or tilde.
  suggestions=[]
  if isinstance(value,dict):
   for key,child in list(value.items())[:30]:
    token=str(key).replace('~','~0').replace('/','~1')
    suggestions.append(dict(pointer=pointer+'/'+token,characters=len(compact(child)),kind=type(child).__name__))
  elif isinstance(value,list) and page and page['returned']:
   suggestions.append(dict(pointer=pointer,offset=page['offset'],limit=1,meaning='Request one item at a time; narrow its pointer further if needed.'))
   if start is None and end is None:
    suggestions.append(dict(pointer=pointer+'/'+str(page['offset']),meaning='Inspect this original list item by pointer, then narrow its fields.'))
  elif isinstance(value,str) and page:
   suggestions.append(dict(pointer=pointer,char_offset=0,char_limit=2500,meaning='Read the full original string in character pages; advance using next_offset.'))
  return dict(topic=topic,needs_narrower_query=True,characters=len(compact(result)),pointer=pointer,page=page,suggestions=suggestions,instruction=help_text,source_sha256=record['sha256'])
 return result

def brief(packet,manifest,negative_summary=None,usage=None):
 current=packet.get('current_review',packet.get('baseline_review',{}));job=packet.get('job',{});config=packet.get('current_config',packet.get('historical_anchor_config',{}))
 keep=['lambda_adv','main_lr','D_lr','D_steps_per_G_step','temperature','confidence_cap','main_weight_decay','D_weight_decay','supplementary_loss_weight','supplementary_positive_scale','supplementary_occurrence_cap','supplementary_interval_steps','supplementary_horizon_epochs','adversarial_projection_strength','main_lr_finish_epochs','adversarial_state','schedule_state','supplementary_default','_automatic_negative_state']
 return dict(decision_kind=packet['decision_kind'],epoch=packet['epoch'],owner=pick(job,['job_id','run_id','country','target_year','architecture','source_year','pretrained_epoch','minimum_stop_epoch','max_epoch','review_interval','per_epoch_policy','two_tier_review_policy','deep_review_interval','supplementary_default_policy','evidence_based_stop_policy']),early_stopping_policy=packet.get('early_stopping_policy'),allowed_actions=packet.get('allowed_actions'),current_metrics=metrics(current),automatic_negative_measured_usage=compact_usage(usage) if usage else None,current_controls=pick(config,keep),allowed_parameter_bounds=packet.get('allowed_parameter_bounds'),safety=packet.get('safety_envelope'),source_safety=packet.get('source_protection_policy'),information_boundary=packet.get('information_boundary'),history_catalog={k:dict(characters=v['characters'],items=v['items']) for k,v in manifest['documents'].items()},gradient_semantics='source_adversarial_cosine is source versus combined adversarial gradient; source_local_cosine and source_global_cosine are separate components. Never combine these fields into one numerical range. State sample count and avoid generalizing a few probes to all training.',memory_instruction='Use get_training_evidence for relevant full facts/history/prior/experiments/candidates. Not queried is not absent. Before attributing an action response, retrieve its original action and comparable observations. Before SELECT inspect the candidate set. All own-run history is preserved, not only recent4. Short thread, same logical group owner.',prior_summary='user_prior_observation_v1; WestAfrica_whole_map_reference. Dense Cashew mainly expected in strongest raw-score prior areas across WestAfrica; p90/p95/p99 about0.844/0.867/0.917. Missing context unknown; no hard mask/quota. Query prior for actual group evidence before regional claims.',negative_candidates=negative_summary)
