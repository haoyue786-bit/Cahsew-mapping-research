"""Check live source/panel identities; this is not proof of blind training.

The registry trust anchor is a reviewed, fixed digest, not a hash from the job.
These checks do not authorize launch: previous model provenance, actual data
producer integrity, runtime code identity, and in-flight file changes remain
separate obligations. This version rejects unknown source profiles/roots.
"""
import hashlib,json,os
from pathlib import Path
from evaluation_sample_v1 import EvaluationNames
from train_tile import source_label_path

REGISTRY=Path(__file__).resolve().parents[1]/'references/model_registry.json'
REGISTRY_SHA256='e1ba568a1249336a28990f1f5b3705fe147e2b26c0c016b02dee4e102c088c93'
TARGET_ROOT=Path('@ASSETS@/alphaearth_data_split')
ALLOWED_PROFILES={'source2019_benin_e173','source2025_benin_e173'}

def digest_file(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def path(p):return Path(p).resolve(strict=True)
def under(root,relative):
 if not isinstance(relative,str) or not relative or Path(relative).is_absolute():raise ValueError('nonrelative image name')
 candidate=(root/relative).resolve(strict=True)
 try:candidate.relative_to(root)
 except ValueError:raise ValueError('image path escapes registered root')
 if not candidate.is_file() or candidate.suffix.lower() not in ('.tif','.tiff'):raise ValueError('invalid image file')
 return candidate
def lines(p):return [s.strip() for s in Path(p).read_text().splitlines() if s.strip()]
def input_snapshot(job,source_valid_names,panel_names):
 """Cheap before/after identity snapshot; not a content hash or race proof."""
 source=path(job['source']['source_root']);labels=path(job['source']['source_label_root']);target=path(job['target_root'])
 policy=job.get('target_evaluation') or {}
 entries=[('registry',str(path(REGISTRY)))]
 for role,p in (('source_train_list',job['source']['source_train']),('source_valid_list',job['source']['source_valid']),
                ('pretrained_checkpoint',job['source']['pretrained']),('target_full_list',job['target_full_list']),('target_train_list',job['target_train_list']),
                ('panel_names',policy['names_file']),('panel_locations',policy['locations_csv']),('panel_report',policy['report'])):
  entries.append((role,str(path(p))))
 for name in source_valid_names:
  image=under(source,name);label=source_label_path(labels,name).resolve(strict=True)
  try:label.relative_to(labels)
  except ValueError:raise ValueError('source label escapes registered root')
  entries.extend((('source_image',str(image)),('source_label',str(label))))
 for name in panel_names:entries.append(('target_image',str(under(target,name))))
 snapshot=[]
 for role,name in entries:
  stat=os.stat(name)
  if not Path(name).is_file():raise ValueError('sample path is not a regular file')
  snapshot.append((role,name,stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns))
 data=json.dumps(snapshot,separators=(',',':'),ensure_ascii=False).encode()
 return {'sha256':hashlib.sha256(data).hexdigest(),'file_count':len(snapshot),'scope':'stat-only; not content hashing or race proof'}
def assert_unchanged(before,after):
 if not isinstance(before,dict) or not isinstance(after,dict) or before!=after:raise RuntimeError('source or target input metadata changed during review')
def assert_runtime_paths(gan,common,expected):
 for module in (gan,common):
  for attr,key in (('SOURCE_VALID','source_valid'),('SOURCE_ROOT','source_root'),('SOURCE_LABEL_ROOT','source_label_root'),('TARGET_ROOT','target_root')):
   if path(getattr(module,attr))!=expected[key]:raise ValueError('runtime path mismatch: '+attr)
 diag=common.old_diag
 for attr,key in (('SOURCE_VALID','source_valid'),('SOURCE_ROOT','source_root'),('LABEL_ROOT','source_label_root'),('TARGET_ROOT','target_root')):
  if path(getattr(diag,attr))!=expected[key]:raise ValueError('diagnostic path mismatch: '+attr)
 if not callable(getattr(diag,'source_label_path',None)):raise ValueError('missing diagnostic label resolver')

def check(job,gan,common,source_valid_names,panel_names,*,epoch,expected_epoch,hash_pretrained=True):
 if type(epoch) is not int or epoch<0 or type(expected_epoch) is not int or epoch!=expected_epoch:raise ValueError('review epoch mismatch')
 registry_bytes=REGISTRY.read_bytes()
 if hashlib.sha256(registry_bytes).hexdigest()!=REGISTRY_SHA256:raise ValueError('reviewed model registry changed')
 registry=json.loads(registry_bytes);name=job.get('source_profile')
 if name not in ALLOWED_PROFILES:raise ValueError('source profile not reviewed')
 profile=registry['profiles'][name];source=job.get('source')
 if not isinstance(source,dict):raise ValueError('missing source identity')
 for key in ('source_country','source_year','source_root','source_label_root','source_train','source_valid','source_train_sha256','source_valid_sha256','pretrained','pretrained_sha256'):
  if source.get(key)!=profile.get(key):raise ValueError('job differs from reviewed source registry: '+key)
 if job.get('target_year') not in profile['compatible_target_years']:raise ValueError('unsupported source/target year')
 expected={k:path(profile[k]) for k in ('source_root','source_label_root','source_train','source_valid','pretrained')};expected['target_root']=path(TARGET_ROOT)
 if path(job['target_root'])!=expected['target_root']:raise ValueError('target imagery root differs from reviewed root')
 source_files={}
 for key in ('source_train','source_valid'):
  data=expected[key].read_bytes()
  if hashlib.sha256(data).hexdigest()!=profile[key+'_sha256']:raise ValueError('source file changed: '+key)
  source_files[key]=[x.strip() for x in data.decode('utf-8').splitlines() if x.strip()]
 if type(hash_pretrained) is not bool:raise ValueError('invalid pretrained verification option')
 if hash_pretrained and digest_file(expected['pretrained'])!=profile['pretrained_sha256']:raise ValueError('source pretrained file changed')
 assert_runtime_paths(gan,common,expected)
 source_names=source_files['source_valid'];train=source_files['source_train']
 if len(source_names)!=profile['source_valid_expected_count'] or len(train)!=profile['source_train_expected_count'] or len(source_names)!=len(set(source_names)) or set(source_names)&set(train):raise ValueError('source split invalid')
 if list(source_valid_names)!=source_names:raise ValueError('source names differ from reviewed split')
 for name in source_names:
  under(expected['source_root'],name)
  expected_label=source_label_path(expected['source_label_root'],name).resolve(strict=True)
  try:expected_label.relative_to(expected['source_label_root'])
  except ValueError:raise ValueError('source label escapes registered root')
  for resolver in (gan.source_label_path,common.source_label_path,common.old_diag.source_label_path):
   if path(resolver(name))!=expected_label:raise ValueError('runtime source label resolver differs')
  if not expected_label.is_file():raise ValueError('source label missing')
 policy=job.get('target_evaluation')
 if not isinstance(policy,dict) or policy.get('version')!='fixed_tile_stratified_evaluation_v1':raise ValueError('fixed unlabelled panel is required')
 actual=panel_names
 if not isinstance(actual,EvaluationNames) or not isinstance(panel_names,EvaluationNames) or list(actual)!=list(panel_names) or actual.policy!=panel_names.policy or actual.locations!=panel_names.locations or actual.report!=panel_names.report:raise ValueError('live panel differs from registered panel')
 for name in actual:under(expected['target_root'],name)
 return {'epoch':epoch,'registry_sha256':REGISTRY_SHA256,'source_profile':job['source_profile'],
  'source_validation_list_sha256':profile['source_valid_sha256'],'source_validation_count':len(source_names),
  'source_checkpoint_sha256':profile['pretrained_sha256'],'source_checkpoint_content_checked':hash_pretrained,'target_panel_names_sha256':policy['names_sha256'],
  'target_panel_locations_sha256':policy['locations_csv_sha256'],'target_panel_report_sha256':policy['report_sha256'],
  'target_panel_count':len(actual),'status':'identity_checked_not_lineage_attestation','training_authorized':False,
  'limitations':['source checkpoint point exposure not proved','panel selection independent of target point outcomes not proved',
   'image and label contents not hashed','in-flight input file replacement not prevented','runtime implementation integrity must be checked separately']}
