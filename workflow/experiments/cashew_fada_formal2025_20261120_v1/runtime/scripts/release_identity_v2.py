"""Verify the pinned cohort code once per process before training."""
import hashlib,json
from pathlib import Path

def verify(root):
    root=Path(root)
    identity=json.loads((root/'runtime_identity.json').read_text())
    if identity.get('version')!='cashew_fada_formal2025_v1':raise ValueError('unexpected runtime release')
    for name,digest in identity['files'].items():
        p=root/name
        if p.is_symlink() or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
            raise ValueError('runtime release changed: '+name)
    return identity['version']
