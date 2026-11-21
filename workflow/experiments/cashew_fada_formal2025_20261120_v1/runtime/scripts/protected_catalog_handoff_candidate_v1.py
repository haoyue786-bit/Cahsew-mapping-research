"""Hard-disabled, same-run private catalog -> reviewed mask handoff.

No production authorization: true source/catalog lineage is not yet
certified. Only tests may patch the private denial, in temporary synthetic
journals.  Agent input is a numeric summary; private IDs/caches never go to it.
"""
import io,json,math,re,tempfile
from pathlib import Path
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from affine import Affine
from decision_transaction import canonical,digest,immutable,read_json
from sealed_epoch_journal_v1 import EpochJournal
from sealed_worker_adapter_v1 import stage_for
from decision_transaction import Transaction
from decision_bridge import normalize_review,normalize_automatic_selection
from automatic_negative_selection_validation_v1 import validate_selection

NAME=re.compile(r'^[A-Za-z0-9_-]{1,96}$')
VERSION='protected_catalog_handoff_candidate_v1'

def _require_source_lineage():
 raise RuntimeError('trusted catalog/input lineage unavailable; production disabled')

def _guard(journal):
 _require_source_lineage()
 from sealed_safe_epoch_journal_v1 import SafeTrainingJournal
 from safe_pool_journal_router_sim_v1 import SafePoolJournal
 if type(journal) not in (EpochJournal,SafeTrainingJournal,SafePoolJournal):raise ValueError('owning journal required')
 journal._held()
 if journal.job.get('experimental_gate',{}).get('simulation_only') is not True:raise ValueError('synthetic-only job required')
 try:journal.root.resolve().relative_to(Path(tempfile.gettempdir()).resolve())
 except ValueError:raise ValueError('synthetic temporary journal required')
 state=journal.recover()
 if state['state']!='FULL_REVIEW':raise RuntimeError('current full review required')
 return state

def _directory(journal,epoch):return journal.root/'protected_catalog_handoff'/('epoch_%06d'%epoch)
def _budget(journal):
 b=journal.job.get('experimental_catalog_handoff_budget')
 if not isinstance(b,dict) or set(b)!={'max_binary_bytes'} or type(b['max_binary_bytes']) is not int or b['max_binary_bytes']<1:
  raise RuntimeError('explicit catalog/cache and mask binary budget required')
 return b['max_binary_bytes']
def _pid(name):return '_'.join(Path(name).stem.split('_')[-3:])
def _read(path):
 p=Path(path)
 if p.is_symlink() or not p.is_file():raise ValueError('unsafe private artifact')
 return p.read_bytes()

def _cache(raw,rising,valid):
 with np.load(io.BytesIO(raw),allow_pickle=False) as x:
  if set(x.files)!={'indices','distance','delta'}:raise ValueError('unexpected cache contents')
  ind=x['indices'];distance=x['distance'];delta=x['delta']
 if (ind.ndim!=1 or ind.dtype.kind not in 'iu' or len(ind)!=rising or len(ind)!=len(set(ind.tolist()))
     or len(ind)>valid or np.any(ind<0) or np.any(ind>=128*128) or distance.shape!=ind.shape or delta.shape!=ind.shape
     or not np.isfinite(distance).all() or not np.isfinite(delta).all() or np.any(distance< -1e-5) or np.any(distance>2.00001)
     or np.any(delta<=0) or np.any(delta>1)):raise ValueError('invalid candidate cache')
 return ind,distance,delta

def _summary(rows):
 result={'candidate_count':len(rows)}
 for prefix,key in (('source_distance','source_cashew_distance_mean'),('p2_increase','p2_increase_mean'),('regional_context','regional_context_rank')):
  values=np.asarray([r[key] for r in rows],dtype=float)
  if len(values):
   for tag,q in (('q25',.25),('q50',.5),('q75',.75)):result[prefix+'_'+tag]=float(np.quantile(values,q))
 return result

def stage_owned(journal,*,rows,identity,image_dir,filtered,protected,input_manifest_sha256,previous_epoch,producer_sha256):
 """Called only inside the protected producer while its private snapshot exists."""
 state=_guard(journal);limit_bytes=_budget(journal);epoch=state['epoch'];image_dir=Path(image_dir).resolve(strict=True)
 if not isinstance(rows,list) or not isinstance(identity,dict) or identity.get('source_checkpoint_sha256')!=digest(state['checkpoint']) or identity.get('previous_epoch')!=previous_epoch:
  raise ValueError('catalog checkpoint identity mismatch')
 if not re.fullmatch(r'[0-9a-f]{64}',str(input_manifest_sha256)) or not re.fullmatch(r'[0-9a-f]{64}',str(producer_sha256)):
  raise ValueError('missing producer or input identity')
 names={}
 for name in filtered:
  key=_pid(name)
  if not NAME.fullmatch(key) or key in names:raise ValueError('ambiguous training tile identity')
  names[key]=name
 root=_directory(journal,epoch);root.mkdir(parents=True,exist_ok=True);(root/'cache').mkdir(exist_ok=True)
 intent={'version':VERSION,'job_id':journal.job['job_id'],'run_id':journal.job['run_id'],'epoch':epoch,
  'checkpoint_sha256':digest(state['checkpoint']),'previous_epoch':previous_epoch,
  'previous_checkpoint_sha256':identity['previous_checkpoint_sha256'], 'input_manifest_sha256':input_manifest_sha256,'producer_sha256':producer_sha256}
 immutable(root/'stage_intent.json',canonical(intent));clean=[];seen=set();cache_bytes=0
 for row in rows:
  key=row.get('patch_id') if isinstance(row,dict) else None
  if not isinstance(key,str) or not NAME.fullmatch(key) or key in seen or key not in names:raise ValueError('invalid private catalog row')
  seen.add(key);expected=(image_dir/names[key]).resolve(strict=True)
  if Path(row.get('image_path','')).resolve(strict=True)!=expected:raise ValueError('catalog image differs from protected snapshot')
  with rasterio.open(expected) as ds:
   if ds.width!=128 or ds.height!=128 or ds.crs is None or ds.crs.to_epsg()!=4326 or protected.exclude_bounds(tuple(ds.bounds),ds.crs):
    raise ValueError('candidate image violates geometry protection')
   transform=list(ds.transform)[:6]
  valid=row.get('valid_pixels');rising=row.get('rising_pixels')
  if type(valid) is not int or not 1<=valid<=128*128 or type(rising) is not int or not 1<=rising<=valid:raise ValueError('invalid candidate coverage')
  cache_path=Path(row.get('cache',''));expected_cache=image_dir.parent/'output'/'cache'/(key+'.npz')
  if cache_path.resolve(strict=True)!=expected_cache.resolve(strict=True):raise ValueError('cache escapes owned catalog output')
  cache_raw=_read(cache_path)
  cache_bytes+=len(cache_raw)
  if cache_bytes>limit_bytes:raise RuntimeError('catalog/cache binary budget exceeded')
  if digest(cache_raw)!=row.get('cache_sha256'):raise ValueError('candidate cache changed')
  indices,distance,delta=_cache(cache_raw,rising,valid)
  numbers={k:row.get(k) for k in ('source_cashew_distance_mean','p2_increase_mean','regional_context_rank')}
  if any(type(x) not in (int,float) or not math.isfinite(x) for x in numbers.values()) or not 0<=numbers['regional_context_rank']<=1:
   raise ValueError('invalid candidate ranking evidence')
  if not math.isclose(numbers['source_cashew_distance_mean'],float(distance.mean()),rel_tol=1e-6,abs_tol=1e-6) or not math.isclose(numbers['p2_increase_mean'],float(delta.mean()),rel_tol=1e-6,abs_tol=1e-6):raise ValueError('catalog scores differ from cache')
  image_sha=digest(_read(expected));clean.append({'patch_id':key,'image_sha256':image_sha,'cache_sha256':digest(cache_raw),
   'valid_pixels':valid,'rising_pixels':rising,'transform':transform,**numbers})
  immutable(root/'cache'/(key+'.npz'),cache_raw)
 catalog=canonical(clean);immutable(root/'catalog.json',catalog)
 summary=_summary(clean);immutable(root/'summary.json',canonical(summary))
 ready={'intent_sha256':digest(canonical(intent)),'catalog_sha256':digest(catalog),'summary_sha256':digest(canonical(summary)),
        'cache_sha256':{r['patch_id']:r['cache_sha256'] for r in clean}}
 immutable(root/'stage_ready.json',canonical(ready));return summary

def _stage(journal,epoch):
 root=_directory(journal,epoch);intent,ir=read_json(root/'stage_intent.json');rows,rr=read_json(root/'catalog.json');summary,sr=read_json(root/'summary.json');ready,_=read_json(root/'stage_ready.json')
 if (intent.get('version')!=VERSION or intent.get('epoch')!=epoch or intent.get('job_id')!=journal.job['job_id'] or intent.get('run_id')!=journal.job['run_id']
     or ready!={'intent_sha256':digest(ir),'catalog_sha256':digest(rr),'summary_sha256':digest(sr),'cache_sha256':{r['patch_id']:r['cache_sha256'] for r in rows}}
     or summary!=_summary(rows)):raise ValueError('staged catalog binding mismatch')
 for row in rows:
  raw=_read(root/'cache'/(row['patch_id']+'.npz'))
  if digest(raw)!=row['cache_sha256']:raise ValueError('staged cache changed')
  _cache(raw,row['rising_pixels'],row['valid_pixels'])
 return root,intent,rows,summary,ready

def bind_request(journal,request,trusted,expected,*,previous=None):
 """Bind the exact candidate summary and request BEFORE network submission."""
 state=_guard(journal);_budget(journal);epoch=state['epoch'];root,intent,rows,summary,ready=_stage(journal,epoch)
 if intent.get('checkpoint_sha256')!=digest(state['checkpoint']):raise ValueError('staged catalog is not bound to active training checkpoint')
 evidence=((request.get('evidence') or {}).get('automatic_negative_candidates') or {})
 if (request.get('epoch')!=epoch or request.get('contract_version')!='safe_full_review_candidate_v1'
     or not request.get('automatic_selection_enabled') or request.get('automatic_candidate_limit')!=len(rows)
     or evidence.get('values')!=summary):raise ValueError('review request does not bind staged catalog')
 from request_reconstruction import verify_request
 verify_request(request,trusted,expected=expected,previous=previous)
 journal._verify_review_request(epoch,request,'FULL_REVIEW',expected,trusted,previous)
 # The preceding hash binds the trained checkpoint; the actual Agent input may
 # have a deterministic before-review default transition.
 pre=journal.prepare_full_review(epoch)['checkpoint']
 binding={'version':VERSION,'epoch':epoch,'stage_ready_sha256':digest(canonical(ready)),
          'request_sha256':digest(canonical(request)),'agent_pre_sha256':digest(pre),'trained_sha256':intent['checkpoint_sha256']}
 binding_path=root/'review_binding.json'
 submission=stage_for(journal.job,request)/'transaction'/'submission.json'
 if submission.exists() and not binding_path.is_file():raise RuntimeError('catalog must be bound before request submission')
 immutable(binding_path,canonical(binding));return binding

def materialize(journal,pre,decision):
 """Create the same pool_change expected by the single-authority bridge."""
 state=_guard(journal);limit_bytes=_budget(journal);epoch=state['epoch'];root,intent,rows,summary,ready=_stage(journal,epoch)
 if intent.get('checkpoint_sha256')!=digest(state['checkpoint']):raise ValueError('staged catalog is not bound to active training checkpoint')
 binding,_=read_json(root/'review_binding.json')
 if hasattr(journal,'_out_fs'):
  prefix='agent/sealed_v1/epoch_%03d/full/transaction/'%epoch;fs=journal._out_fs
  identity={'job_id':journal.job['job_id'],'run_id':journal.job['run_id'],'epoch':epoch,'kind':'full_review'}
  manifest=fs.json(prefix+'manifest.json');req=fs.read(prefix+'request.json');recorded_pre=fs.read(prefix+'pre_checkpoint.bin')
  submission=fs.json(prefix+'submission.json');terminal=fs.json(prefix+'terminal_identity.json');completed=fs.json(prefix+'completed.json')
  answer_raw=fs.read(prefix+'response.json');answer=json.loads(answer_raw);stored_raw=fs.read(prefix+'decision.json');stored=json.loads(stored_raw);receipt=fs.json(prefix+'decision_receipt.json')
  if (binding.get('stage_ready_sha256')!=digest(canonical(ready)) or binding.get('request_sha256')!=digest(req) or binding.get('agent_pre_sha256')!=digest(pre)
      or manifest.get('identity')!=identity or manifest.get('request_sha256')!=digest(req) or manifest.get('pre_checkpoint_sha256')!=digest(recorded_pre)
      or pre!=recorded_pre or stored!=decision or submission.get('identity')!=identity or terminal!=completed
      or submission.get('thread_id')!=completed.get('thread_id') or completed.get('identity')!=identity
      or completed.get('request_sha256')!=manifest['request_sha256'] or completed.get('response_sha256')!=digest(answer_raw)
      or receipt!={'identity':identity,'request_sha256':manifest['request_sha256'],'pre_checkpoint_sha256':digest(pre),'response_sha256':digest(answer_raw),'decision_sha256':digest(stored_raw)}):raise ValueError('review turn is unresolved or not bound to catalog')
 else:
  stage=stage_for(journal.job,{'contract_version':'safe_full_review_candidate_v1','epoch':epoch})
  txn=Transaction(stage/'transaction',journal.job['job_id'],journal.job['run_id'],epoch,'full_review')
  manifest,req,recorded_pre=txn._require();submission,_=read_json(txn.root/'submission.json');terminal,_=read_json(txn.root/'terminal_identity.json')
  completed,_=read_json(txn.root/'completed.json');answer,answer_raw=read_json(txn.root/'response.json');stored,stored_raw=read_json(txn.root/'decision.json');receipt,_=read_json(txn.root/'decision_receipt.json')
  if (binding.get('stage_ready_sha256')!=digest(canonical(ready)) or binding.get('request_sha256')!=digest(req) or binding.get('agent_pre_sha256')!=digest(pre)
      or pre!=recorded_pre or stored!=decision or submission.get('identity')!=txn.identity or terminal!=completed
      or submission.get('thread_id')!=completed.get('thread_id') or completed.get('identity')!=txn.identity
      or completed.get('request_sha256')!=manifest['request_sha256'] or completed.get('response_sha256')!=digest(answer_raw)
      or receipt!={'identity':txn.identity,'request_sha256':manifest['request_sha256'],'pre_checkpoint_sha256':digest(pre),'response_sha256':digest(answer_raw),'decision_sha256':digest(stored_raw)}):raise ValueError('review turn is unresolved or not bound to catalog')
 request=json.loads(req);selection=normalize_automatic_selection(request,answer,actual_candidate_count=len(rows),actual_max_weight=request.get('automatic_max_weight'))
 if selection!=decision.get('automatic_negative') or answer.get('action')!=decision.get('action') or decision.get('action') not in ('KEEP','ADJUST','CONTINUE','PAUSE_ADV','RESUME_ADV') or decision.get('continue_training') is not True:
  raise ValueError('pool selection is not an eligible completed decision')
 selection=validate_selection(selection,len(rows))
 if selection['action']=='ABSTAIN':return None
 weights=selection['ranking_weights'];scored=sorted(rows,key=lambda r:(-(weights['source_cashew_distance']*r['source_cashew_distance_mean']/2+weights['p2_increase']*r['p2_increase_mean']+weights['regional_prior']*r['regional_context_rank']),r['patch_id']))
 selected=scored[:selection['selected_count']];out=root/'masks';out.mkdir(exist_ok=True)
 identity={'version':VERSION,'stage_sha256':digest(canonical(ready)),'decision_sha256':digest(stored_raw),'checkpoint_sha256':digest(pre)}
 immutable(root/'mask_intent.json',canonical(identity));mask_bytes={};used=sum((root/'cache'/(r['patch_id']+'.npz')).stat().st_size for r in rows)
 for row in selected:
  idx,distance,delta=_cache(_read(root/'cache'/(row['patch_id']+'.npz')),row['rising_pixels'],row['valid_pixels'])
  score=weights['source_cashew_distance']*distance/2+weights['p2_increase']*delta+weights['regional_prior']*row['regional_context_rank']
  limit=min(256,int(.02*row['valid_pixels']));chosen=idx[np.argsort(-score,kind='stable')[:limit]]
  if len(chosen)!=limit or not limit:raise ValueError('empty selected mask')
  mask=np.full((128,128),10,np.uint8);mask.ravel()[chosen]=0
  with MemoryFile() as mem:
   with mem.open(driver='GTiff',width=128,height=128,count=1,dtype='uint8',crs='EPSG:4326',transform=Affine(*row['transform']),nodata=10,compress='deflate') as ds:ds.write(mask,1)
   data=mem.read()
  used+=len(data)
  if used>limit_bytes:raise RuntimeError('catalog/cache and mask binary budget exceeded')
  filename='automatic_noncashew_'+row['patch_id']+'.tif';immutable(out/filename,data);mask_bytes[filename]=data
 generation=digest(canonical(identity));pool={'generation':generation,'horizon':128,'weight':selection['automatic_weight'],'masks':{n:digest(b) for n,b in sorted(mask_bytes.items())}}
 result={'version':VERSION,'identity':identity,'pool_manifest':pool,'selected_count':len(mask_bytes)}
 immutable(root/'materialized.json',canonical(result))
 return {'action':'ACTIVATE','manifest':pool,'masks':mask_bytes}

def finish_review(bridge,request,trusted,expected,**router_options):
 """Resolve the pool only *inside* its same-run review transaction.

 Call bind_request before calling this method. The worker's normal request
 validation and exact turn receipt still apply. Source lineage remains hard
 blocked; this function is usable only in a monkeypatched synthetic test.
 """
 from unified_epoch_pool_bridge_sim_v1 import UnifiedEpochPool
 from safe_pool_journal_router_sim_v1 import SafePoolJournal,ordinary_post
 _require_source_lineage()
 if type(bridge) is not UnifiedEpochPool or type(bridge.journal) is not SafePoolJournal:
  raise ValueError('safe single-authority pool journal required')
 if 'transition' in router_options or 'pool_receipt_verify' in router_options:
  raise ValueError('handoff owns review transition')
 before=bridge.recover()
 if before['state']!='FULL_REVIEW' or request.get('epoch')!=before['epoch']:
  raise ValueError('current full review required')
 epoch=before['epoch'];prior=bridge._cp_state(before['checkpoint'])
 # No automatic late binding after a request may already have been submitted.
 root=_directory(bridge.journal,epoch)
 if not (root/'review_binding.json').is_file():raise RuntimeError('catalog request was not bound before submission')
 def transition(pre,decision):
  change=materialize(bridge.journal,pre,decision)
  if change is None:return ordinary_post(pre,decision)
  return bridge._pool_action(epoch,pre,decision,change,ordinary_post)
 def verify(checked,decision,post):
  if checked!=epoch:return False
  bridge._audit_pool_action(epoch,prior,post);return True
 bridge.journal.finish_agent(epoch,request,trusted,expected,transition=transition,pool_receipt_verify=verify,**router_options)
 return bridge.recover()
