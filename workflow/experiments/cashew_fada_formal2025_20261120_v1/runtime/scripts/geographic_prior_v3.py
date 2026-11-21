"""Pinned multi-country2019 prior context; context alone never assigns pixel labels."""
import hashlib,json
from pathlib import Path
VERSION='WestAfrica_approved_regional_prior_target2019_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def packet_evidence(job,compact=False):
 pinned=job['geographic_prior'];p=Path(pinned['report'])
 if sha(p)!=pinned['sha256']:raise ValueError('prior context changed')
 result=json.loads(p.read_text())
 if result['version']!=VERSION or pinned['version']!=VERSION or job['geographic_prior_policy']['version']!=VERSION:raise ValueError('prior version mismatch')
 iso={'Ivoire':'CIV','Nigeria':'NGA','Senegal':'SEN'}.get(job['country'])
 if not iso or result['country']!=iso or job['target_year']!=2019 or result['target_year']!=2019:raise ValueError('prior country/year mismatch')
 if result['member_tile_ids']!=job['target_domain']['member_tile_ids']:raise ValueError('prior membership mismatch')
 for path,expected in result['artifact_hashes'].items():
  if sha(path)!=expected:raise ValueError('prior artifact changed: '+path)
 if sha(pinned['card'])!=pinned['card_sha256']:raise ValueError('prior card changed')
 return {k:result[k] for k in ['version','target_year','group','member_tile_ids','patch_summaries','guidance','limitations','source_class_weights_changed','tile_fraction_estimate']} if compact else result
