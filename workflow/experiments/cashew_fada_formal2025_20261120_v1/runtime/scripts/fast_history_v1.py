"""Use same-run point-isolated worker snapshots for historical facts."""
import copy
from pathlib import Path
from complete_history_store import _project,load_epoch

_STATIC_PATHS=('image_root','label_root','names_file')

def committed(journal,epoch):
    # load_epoch's crash fallback deliberately reads an old full checkpoint.
    # Fast mode instead refuses missing snapshots and leaves reconciliation to
    # the owner, rather than silently reintroducing the expensive replay.
    path=Path(journal.job['output_root'])/'evidence_store/snapshots'/('epoch_%06d.json'%epoch)
    if path.is_symlink() or not path.is_file():raise ValueError('missing fast history snapshot')
    return load_epoch(journal.job['output_root'],journal.job['job_id'],journal.job['run_id'],epoch)[0]

def restore_config(snapshot_controls,job):
    """Restore the three job-pinned paths intentionally omitted from snapshots."""
    config=copy.deepcopy(snapshot_controls)
    saved=config.get('supplementary_training');spec=job.get('supplementary')
    if isinstance(saved,dict):
        if not isinstance(spec,dict):raise ValueError('missing static supplementary specification')
        for key in _STATIC_PATHS:
            if key in spec:saved[key]=spec[key]
    return config

def verify_current_config(config,job):
    # Unknown omitted keys fail closed instead of subtly changing gate inputs.
    if restore_config(_project(config,('controls',)),job)!=config:
        raise ValueError('fast history cannot reconstruct current gate config exactly')
