"""Fail closed before online publication if forbidden point-evaluation fields appear."""
import hashlib,json,re
BLOCK={'target_point_metrics','target_validation','target_validation_metrics','development_points','point_predictions','optimal_f1','best_threshold','validation_point_predictions','point_f1','target_f1','target_precision','target_recall','target_accuracy','threshold_search','pointwise_correctness'}
BLOCK.update({'target_points','target_point_labels','validation_point_labels','point_labels',
 'point_precision','point_recall','point_accuracy','point_confusion','point_confusion_matrix',
 'point_fp','point_fn','point_fpr','point_false_positive_rate','point_false_negatives',
 'point_true_positives','point_true_negatives','point_threshold','point_fitted_threshold',
 'fitted_threshold','point_rankings','point_checkpoint_ranking','target_point_results',
 'target_point_f1','target_point_precision','target_point_recall','target_point_accuracy',
 'target_point_fp','target_point_fn','target_point_fpr','target_confusion_matrix',
 'target_checkpoint_ranking','optimal_threshold','pointwise_predictions'})
TARGET_CONTEXT={'target','target_validation','target_point','target_points','target_evaluation',
                'target_metrics','evaluation_points','validation_points','point_evaluation',
                'point_metrics','point_results','point_sample','point_samples',
                'target_unsupervised','target_results','target_point_evaluation'}
POINT_RESULTS={'label','labels','f1','precision','recall','accuracy','confusion',
               'confusion_matrix','fp','fn','tp','tn','fpr','false_positive_rate',
               'false_positive','false_negative','false_positives','false_negatives',
               'threshold','fitted_threshold','rankings','checkpoint_ranking','correctness'}
def verify_online(value,path='root',target_context=False):
 if isinstance(value,dict):
  for k,v in value.items():
   name=str(k).lower()
   if name in BLOCK or (target_context and name in POINT_RESULTS):
    raise ValueError('Sealed point evidence at '+path+'.'+str(k))
   verify_online(v,path+'.'+str(k),target_context or name in TARGET_CONTEXT)
 elif isinstance(value,list):
  for i,v in enumerate(value):verify_online(v,path+'['+str(i)+']',target_context)


def isolate_history_text(result):
 """Validate a source-owned reply without deleting or rewriting its prose.

 The history service reads committed current-run training producers only.
 Final point evaluation is a separate offline application, not a history source.
 """
 if not isinstance(result,dict):raise ValueError('history result must be a mapping')
 verify_online(result)
 return result
