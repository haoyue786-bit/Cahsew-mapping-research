"""Strict candidate packet. Caller must authenticate worker-owned artifacts and hashes.
No raw history, narrative, external tools, or unverified previous Agent reasoning.
"""
import hashlib,json,math,re
TRAIN={'cls','seg_cls','adv','D_loss','local_D_acc','global_D_acc','cls_grad','adv_grad','effective_gradient_ratio','steps','supp_scheduled_samples','supp_unique_samples','supp_selected_pixels'}
TRAIN |= {f'{opt}_steps' for opt in ('G','D_local','D_global')}
TRAIN |= {f'{opt}_{field}' for opt in ('G','D_local','D_global') for field in ('attempted_steps','steps_with_gradient')}
TRAIN |= {'domain_'+field for field in ('planned_steps','loaded_steps','completed_g_steps',
 'target_loss_active_completed_steps','source_planned_occurrences','source_planned_unique',
 'target_planned_occurrences','target_planned_unique','source_loaded_occurrences','source_loaded_unique',
 'target_loaded_occurrences','target_loaded_unique','source_completed_occurrences','source_completed_unique',
 'target_completed_occurrences','target_completed_unique','target_loss_active_occurrences','target_loss_active_unique',
 'target_generator_nonzero_weight_steps','target_generator_nonzero_weight_occurrences','target_generator_nonzero_weight_unique')}
TRAIN |= {'supp_'+field for field in ('planned_steps','planned_occurrences','planned_unique_patches',
 'attempted_steps','attempted_occurrences','attempted_unique_patches','loaded_steps','loaded_occurrences','loaded_unique_patches',
 'forward_steps','loss_steps','completed_optimizer_steps','completed_occurrences','completed_unique_patches',
 'human_negative_effective_steps','human_positive_effective_steps','human_negative_effective_pixel_occurrences','human_positive_effective_pixel_occurrences',
 'human_negative_effective_unique_patches','human_positive_effective_unique_patches','unknown_sign_completed_steps')}
TRAIN |= {f'{domain}_{field}' for domain in ('source','target_unlabeled') for field in (
 'sampled_pixels','current_cap_pixel_fraction','current_cap_element_fraction',
 'temperature_down_mean_abs_change','temperature_up_mean_abs_change',
 'temperature_down_pixel_fraction_changed','temperature_up_pixel_fraction_changed',
 'cap_down_mean_abs_change','cap_up_mean_abs_change',
 'cap_down_pixel_fraction_changed','cap_up_pixel_fraction_changed')}
TRAIN |= {f'{opt}_group_{index}_{field}' for opt in ('G','D_local','D_global') for index in range(32) for field in (
 'lr','lr_multiplier','weight_decay','beta1','beta2','eps','sample_delta_l2','sample_before_l2','sample_count',
 'coverage_sampled_steps','coverage_sampled_tensors','coverage_eligible_tensors')}
TRAIN |= {'sampled_temperature','sampled_cap','temperature_probe_lower','temperature_probe_upper','cap_probe_lower','cap_probe_upper'}
TRAIN |= {f'{prefix}_{field}' for prefix,fields in (
 ('human_polygon',('negative_pixel_occurrences','positive_pixel_occurrences','positive_scale','effective_negative_weight_min','effective_negative_weight_max','effective_positive_weight_min','effective_positive_weight_max')),
 ('automatic_negative',('effective_weight','configured_weight','updates','unique_patches','patch_occurrences','pixel_occurrences','selected_patch_count','pool_age_epochs')),
 ('gradient_direction',('measured_steps','completed_steps','opposing_steps','source_adversarial_cosine_median','source_local_cosine_median','source_global_cosine_median','local_global_cosine_median','weighted_opposition_relative_to_source_median')),
 ) for field in fields}
CONTROL={'main_lr','D_lr','lambda_adv','temperature','confidence_cap','main_weight_decay','D_weight_decay','D_steps_per_G_step','supplementary_loss_weight','supplementary_positive_scale','supplementary_occurrence_cap','supplementary_interval_steps','supplementary_horizon_epochs','adversarial_projection_strength','main_lr_finish_epochs'}
CONTROL.add('refiner_lr_multiplier')
from module_controls_v2 import BOUNDS as MODULE_BOUNDS
CONTROL |= set(MODULE_BOUNDS)
from transfer_regularization_v3 import BOUNDS as REGULARIZATION_BOUNDS
CONTROL |= set(REGULARIZATION_BOUNDS)
from hard_sample_training import BOUNDS as ANNOTATION_BOUNDS
CONTROL |= set(ANNOTATION_BOUNDS)
TRAIN.add('regularization_loss')
from module_dynamics_contract import COMPONENTS as MODULE_COMPONENTS, FIELDS as MODULE_FIELDS
TRAIN |= {'module_'+c+'_'+f for c in MODULE_COMPONENTS for f in MODULE_FIELDS}
from module_observer_v2 import KEYS as DETAIL_KEYS
TRAIN |= DETAIL_KEYS
PANEL={'predicted_class2_fraction','mean_entropy','pixel_agreement','class2_iou'}
PANEL |= {'class0_fraction','class1_fraction','class3_fraction','normalized_entropy','sample_patch_count','sample_valid_pixel_count','probe_snd','probe_target_patch_count','probe_source_patch_count','missing_tile_count','comparison_from_epoch','comparison_to_epoch','comparison_class2_pixels_previous','comparison_class2_pixels_current'}
PANEL |= {f'tile_{i}_{field}' for i in range(64) for field in ('class2_fraction','sample_patch_count','sampled_valid_pixels')}
SOURCE={'class2_f1','source_loss','validation_pixels','class0_f1','class1_f1','class3_f1'}
from module_functional_v2 import KEYS as FUNCTIONAL_KEYS
SOURCE |= FUNCTIONAL_KEYS
PANEL |= FUNCTIONAL_KEYS
AUTOMATIC_SUMMARY={'candidate_count'}|{p+'_'+q for p in ('source_distance','p2_increase','regional_context') for q in ('q25','q50','q75')}
AUTOMATIC_POOL={'weight','effective_weight','updates','cursor','horizon'}
BRANCH={'status_code','completed_steps','control_value','treatment_value'}|{role+'_'+field for role in ('control','treatment') for field in ('start_source_class2_f1','end_source_class2_f1','start_target_class2_fraction','end_target_class2_fraction','end_effective_gradient_ratio')}
HASH=re.compile(r'^[0-9a-f]{64}$')

def values_digest(kind,epoch,values):
 """Digest the exact projected artifact, not just an unrelated source path."""
 payload={'kind':kind,'epoch':epoch,'values':values}
 return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def obj(value,keys):
 if not isinstance(value,dict) or set(value)!=set(keys): raise ValueError('unexpected or missing fields')
def nums(d,allowed):
 if not isinstance(d,dict) or set(d)-allowed: raise ValueError('unknown observation')
 if any(v is not None and (isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v)) for v in d.values()): raise ValueError('nonfinite or nonnumeric observation')
 return dict(d)
def evidence(a,kind,allowed,cutoff, trusted, *, min_epoch=1):
 obj(a,['kind','epoch','sha256','values'])
 if a['kind']!=kind or type(a['epoch']) is not int or not min_epoch<=a['epoch']<=cutoff:raise ValueError('wrong scope or future epoch')
 if not isinstance(a['sha256'],str) or not HASH.fullmatch(a['sha256']) or (kind,a['epoch'],a['sha256']) not in trusted:raise ValueError('untrusted artifact')
 values=nums(a['values'],allowed)
 if values_digest(kind,a['epoch'],values)!=a['sha256']:raise ValueError('artifact contents do not match registered hash')
 return {'epoch':a['epoch'],'sha256':a['sha256'],'values':values}
def training_window(epoch,rows,trusted):
 if type(epoch) is not int or epoch<1:raise ValueError('invalid epoch')
 if not isinstance(rows,list) or not 1<=len(rows)<=5:raise ValueError('window must contain 1..5 rows')
 clean=[]
 for row in rows:
  obj(row,['training','controls']);t=evidence(row['training'],'training',TRAIN,epoch,trusted);c=evidence(row['controls'],'controls',CONTROL,epoch,trusted)
  if t['epoch']!=c['epoch']:raise ValueError('mismatched control epoch')
  clean.append({'training':t,'controls':c})
 epochs=[x['training']['epoch'] for x in clean]
 if epochs!=list(range(epoch-len(epochs)+1,epoch+1)):raise ValueError('stale, missing or duplicated window')
 return clean
def packet(epoch,rows,source,panel,trusted):
 clean=training_window(epoch,rows,trusted)
 s=evidence(source,'source',SOURCE,epoch,trusted);p=evidence(panel,'panel',PANEL,epoch,trusted)
 if s['epoch']!=epoch or p['epoch']!=epoch:raise ValueError('source/panel not measured at this review')
 return {'epoch':epoch,'training_window':clean,'source_validation':s,'unlabelled_panel':p}

def tool_query(*args,**kwargs):
 raise RuntimeError('fail closed: no arbitrary historical retrieval is allowed by this isolated candidate')
