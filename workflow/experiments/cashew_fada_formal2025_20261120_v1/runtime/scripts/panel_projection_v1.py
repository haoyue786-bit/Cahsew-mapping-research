"""Typed, point-sealed source/target panel projection, not provenance attestation.

 Caller must independently authenticate the producer, input identities, epoch,
 checkpoint and panel before invoking. Hash fields bind supplied records only;
 they cannot establish clean training lineage, same pixel mask, or independently
 prove which tensors were actually evaluated. No target labels or points used.
"""
import math,re

HEX=re.compile(r'^[0-9a-f]{64}$');TILE=re.compile(r'^[0-9a-f]{20}$')

def identity(x):
 keys={'epoch','checkpoint_sha256','panel_sha256','panel_content_manifest_sha256','valid_mask_manifest_sha256'}
 if not isinstance(x,dict) or set(x)!=keys:raise ValueError('invalid review identity')
 if type(x['epoch']) is not int or x['epoch']<0 or any(not isinstance(x[k],str) or not HEX.fullmatch(x[k]) for k in keys-{'epoch'}):raise ValueError('invalid review identity fields')
 return dict(x)
def value(x,*,low=None,high=None):
 if x is None:return None
 if type(x) not in (int,float) or not math.isfinite(x) or (low is not None and x<low) or (high is not None and x>high):raise ValueError('nonfinite or out-of-range numeric fact')
 return x

def vector(raw,*,normalized=False):
 if not isinstance(raw,(list,tuple)) or len(raw)!=4:raise ValueError('expected four-class vector')
 out=[value(x,low=0,high=1) for x in raw]
 if normalized and (None in out or abs(sum(out)-1)>1e-3):raise ValueError('invalid probability distribution')
 return out

def count(x):
 if x is None:return None
 if type(x) is not int or x<0:raise ValueError('invalid sample support count')
 return x

def project(source,panel,current,*,roster,expected_current,transition=None,previous=None,transition_identity=None,probe_identity=None,expected_probe_identity=None):
 current=identity(current);expected_current=identity(expected_current)
 if current!=expected_current:raise ValueError('current checkpoint, panel or epoch mismatch')
 if not isinstance(source,dict) or not isinstance(panel,dict):raise ValueError('missing source or target producer')
 if not isinstance(roster,(list,tuple)) or not roster or any(not isinstance(t,str) or not TILE.fullmatch(t) for t in roster) or len(set(roster))!=len(roster):raise ValueError('invalid independently registered tile roster')
 for key in ('class_distribution','f1_per_class'):
  if key not in (panel if key=='class_distribution' else source):raise ValueError('missing four-class metric')
 src={'loss':value(source.get('loss'),low=0),'class_f1':vector(source['f1_per_class']),
      'validation_pixels':count(source.get('pixels'))}
 trg={'class_fraction':vector(panel['class_distribution'],normalized=True),
      'entropy':value(panel.get('entropy'),low=0,high=math.log(4)+1e-6),
      'normalized_entropy':value(panel.get('normalized_entropy'),low=0,high=1),
      'sample_patch_count':count(panel.get('patch_count')),
      'sample_valid_pixel_count':count(panel.get('pixel_count')),
      'probe_snd':None,'tiles':[]}
 # SND uses a separately selected target probe subset, not the full fixed
 # panel. Source/target subset and mask policies each require independent
 # identities, and must match the controller's independently verified record.
 if probe_identity is not None or expected_probe_identity is not None:
  required=set(current)|{'target_patch_count','source_patch_count','target_subset_sha256','source_subset_sha256','target_mask_policy_sha256','source_mask_policy_sha256'}
  if not isinstance(probe_identity,dict) or set(probe_identity)!=required or not isinstance(expected_probe_identity,dict) or set(expected_probe_identity)!=required:raise ValueError('invalid probe identity')
  if probe_identity!=expected_probe_identity:raise ValueError('probe subset or mask policy mismatch')
  probe_base={k:probe_identity[k] for k in current}
  if identity(probe_base)!=current:raise ValueError('unverified probe checkpoint or parent panel')
  for k in ('target_patch_count','source_patch_count'):
   if type(probe_identity[k]) is not int or probe_identity[k]<=0:raise ValueError('invalid probe subset support')
  for k in ('target_subset_sha256','source_subset_sha256','target_mask_policy_sha256','source_mask_policy_sha256'):
   if not isinstance(probe_identity[k],str) or not HEX.fullmatch(probe_identity[k]):raise ValueError('invalid probe subset or mask policy identity')
  trg['probe_snd']=value(panel.get('snd'),low=0)
  trg['probe_support']={k:probe_identity[k] for k in ('target_patch_count','source_patch_count','target_subset_sha256','source_subset_sha256','target_mask_policy_sha256','source_mask_policy_sha256')}
 # Order is fixed externally: tile IDs are not copied into the model packet.
 tiles=panel.get('tile_estimates')
 if tiles is not None:
  if not isinstance(tiles,list):raise ValueError('invalid tile list')
  by={}
  for t in tiles:
   if not isinstance(t,dict) or t.get('tile_id') not in roster or t['tile_id'] in by:raise ValueError('unknown or duplicated tile')
   fraction=value(t.get('class2_fraction'),low=0,high=1);patches=t.get('sample_patch_count');pixels=t.get('sampled_valid_pixels')
   if type(patches) is not int or patches<0 or type(pixels) is not int or pixels<0:raise ValueError('invalid tile support')
   if pixels==0 and fraction is not None:raise ValueError('class fraction undefined without pixels')
   by[t['tile_id']]={'tile_index':roster.index(t['tile_id']),'class2_fraction':fraction,'sample_patch_count':patches,'sampled_valid_pixels':pixels}
  trg['tiles']=[by[t] for t in roster if t in by]
  trg['missing_tile_count']=len(roster)-len(by)
 else:trg['missing_tile_count']=len(roster)
 comparison=None
 if any(x is not None for x in (previous,transition,transition_identity)):
  if previous is None or transition is None or transition_identity is None:raise ValueError('incomplete comparison provenance')
  previous=identity(previous);tr=transition_identity
  if not isinstance(tr,dict) or set(tr)!={'previous','current'} or identity(tr['previous'])!=previous or identity(tr['current'])!=current:raise ValueError('comparison checkpoint identity mismatch')
  if any(previous[k]!=current[k] for k in ('panel_sha256','panel_content_manifest_sha256','valid_mask_manifest_sha256')) or previous['epoch']>=current['epoch']:raise ValueError('different panel, imagery, valid support or non-forward comparison')
  if not isinstance(transition,dict) or transition.get('from_epoch')!=previous['epoch'] or transition.get('to_epoch')!=current['epoch']:raise ValueError('stale or mismatched comparison')
  comparison={'from_epoch':previous['epoch'],'to_epoch':current['epoch'],
    'all_class_pixel_agreement':value(transition.get('all_class_pixel_agreement'),low=0,high=1),
    'class2_iou':value(transition.get('class2_iou'),low=0,high=1)}
  for field in ('class2_pixels_previous','class2_pixels_current'):
   n=transition.get(field)
   if n is not None and (type(n) is not int or n<0):raise ValueError('invalid class pixel support')
   comparison[field]=n
 return {'identity':current,'source_validation':src,'unlabelled_target_panel':trg,'adjacent_prediction_comparison':comparison}
