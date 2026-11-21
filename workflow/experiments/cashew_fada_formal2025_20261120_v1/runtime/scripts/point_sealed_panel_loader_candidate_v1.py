"""Candidate unlabelled-panel manifest reader, not an authorization.

Reads only five fixed role paths under a declared run root.  Before consuming
bytes it walks every path with directory descriptors + O_NOFOLLOW and requires
a regular single-link file with the pre-registered device/inode/size.  It
then verifies the complete SHA-256 and strict panel schema.  It does not open
target imagery or any point/shape file, or emit arbitrary CSV/report text.

The caller must independently authenticate the registration, run lineage,
fixed image frame, producer and absence of target-point-derived selection.
File hashes or a caller-provided registration do not establish those facts.
Filesystem isolation is not a sandbox against an owner who can rewrite an
entire unsigned registration and replace file contents in place.  This module
is experimental, not wired to a worker; production authorization always fails.
"""
import csv
import hashlib
import io
import json
import math
import os
import re
import stat
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType

VERSION = 'point_sealed_panel_loader_candidate_v1'
PANEL_VERSION = 'fixed_tile_stratified_evaluation_v1'
ROLE_PATHS = {
    'names': 'monitor/target_evaluation_fixed.txt',
    'locations': 'monitor/evaluation_samples_absolute.csv',
    'report': 'monitor/evaluation_sampling_report.json',
    'full': 'target_full.txt',
}
ROLES = frozenset((*ROLE_PATHS, 'training'))
HEX = re.compile(r'^[0-9a-f]{64}$')
IDENTITY = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$')
RUN_ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_./-]{0,255}$')
TRAINING = re.compile(r'^target_train_seed[0-9]{1,10}\.txt$')
IMAGE = re.compile(r'(?P<year>20[0-9]{2})/(?P<country>[A-Za-z0-9][A-Za-z0-9._-]{0,95})/'
                   r'(?P<tile>(?P=country)(?P<image_year>20[0-9]{2})_(?P<id>[A-Za-z0-9-]+))/'
                   r'(?P=tile)_(?P<col>[0-9]+)_(?P<row>[0-9]+)\.tif')
CSV_COLUMNS = ('adm1_id','adm1_name','year','tile_id','patch_id','tif_path','center_lon','center_lat',
               'west','south','east','north','population_tile_patches','sampled_tile_patches',
               'expansion_weight','overlaps_training','source_csv_path','prior_card_path')
REPORT_KEYS = frozenset(('version','seed','requested_ratio','training_count','population_count','sample_count',
    'tile_count','overlap_training_count','tile_allocation','allocation','estimate','estimate_limits',
    'boundary_policy','population_list_sha256','training_list_sha256','names_file','names_sha256',
    'locations_csv','locations_csv_sha256'))
ALLOCATION_KEYS = frozenset(('tile_id','population_count','eligible_count','sample_count','expansion_weight'))
REPORT_TEXT = {
    'allocation':'Proportional tile quotas with minimum 8 where available; uniform random without replacement within tile, no prior-based weighting',
    'estimate':'sum_h (N_h/n_h)*Class2_valid_pixels_h / sum_h (N_h/n_h)*valid_pixels_h',
    'estimate_limits':'Ratio estimate of prediction distribution, not target accuracy or a full census; training exclusion and spatial dependence limit representativeness.',
    'boundary_policy':'Same complete boundary-crossing patches as the declared target footprint',
}


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum: raise ValueError('invalid '+name)
    return value


def _root(path):
    if not hasattr(os,'O_NOFOLLOW') or not hasattr(os,'O_DIRECTORY'):
        raise RuntimeError('descriptor-safe reads unavailable')
    if not isinstance(path,(str,Path)) or not os.path.isabs(path): raise ValueError('absolute root required')
    parts=PurePosixPath(str(path)).parts
    if not parts or parts[0]!='/' or any(x in ('.','..') for x in parts): raise ValueError('invalid root')
    fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
    try:
        for name in parts[1:]:
            nxt=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
            os.close(fd);fd=nxt
        return fd
    except BaseException:
        os.close(fd);raise


def _record(record, role):
    keys={'path','sha256','dev','ino','size','max_bytes'}
    if not isinstance(record,dict) or set(record)!=keys:raise ValueError('invalid artifact registration')
    name=record['path']
    if not isinstance(name,str) or (role=='training' and not TRAINING.fullmatch(name)) or (role!='training' and name!=ROLE_PATHS[role]):
        raise ValueError('unexpected artifact role path')
    if not isinstance(record['sha256'],str) or not HEX.fullmatch(record['sha256']):raise ValueError('invalid artifact digest')
    for key in ('dev','ino','size','max_bytes'):_integer(record[key],key,1 if key in ('ino','max_bytes') else 0)
    if record['size']>record['max_bytes']:raise ValueError('artifact exceeds declared budget')
    return record


def _read(rootfd,record):
    """Never read a symlink/hardlink or an inode not registered for this role."""
    parts=PurePosixPath(record['path']).parts;parent=os.dup(rootfd)
    try:
        for part in parts[:-1]:
            nxt=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent)
            os.close(parent);parent=nxt
        fd=os.open(parts[-1],os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=parent)
        try:
            first=os.fstat(fd)
            if not stat.S_ISREG(first.st_mode) or first.st_nlink!=1:
                raise ValueError('artifact is not a private regular file')
            if (first.st_dev,first.st_ino,first.st_size)!=(record['dev'],record['ino'],record['size']):
                raise ValueError('artifact inode does not match registration')
            data=bytearray()
            while len(data)<=record['max_bytes']:
                chunk=os.read(fd,min(1024*1024,record['max_bytes']+1-len(data)))
                if not chunk:break
                data.extend(chunk)
            if len(data)!=record['size'] or len(data)>record['max_bytes']:raise ValueError('artifact length changed')
            last=os.fstat(fd);linked=os.stat(parts[-1],dir_fd=parent,follow_symlinks=False)
            fields=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')
            if any(getattr(first,k)!=getattr(last,k) or getattr(last,k)!=getattr(linked,k) for k in fields):
                raise ValueError('artifact changed while reading')
            result=bytes(data)
            if hashlib.sha256(result).hexdigest()!=record['sha256']:raise ValueError('artifact digest changed')
            return result
        finally:os.close(fd)
    finally:os.close(parent)


def _names(raw):
    try:text=raw.decode('utf-8')
    except UnicodeDecodeError as exc:raise ValueError('invalid name encoding') from exc
    values=text.splitlines()
    if not values or any(not x or x!=x.strip() or not IMAGE.fullmatch(x) for x in values) or len(values)!=len(set(values)):
        raise ValueError('invalid or duplicate image name')
    return values


def _number(value, name):
    try:number=float(value)
    except (ValueError,TypeError) as exc:raise ValueError('invalid '+name) from exc
    if not math.isfinite(number):raise ValueError('nonfinite '+name)
    return number


def _count_string(value, name):
    if not isinstance(value,str) or not re.fullmatch(r'(0|[1-9][0-9]*)',value):raise ValueError('invalid '+name)
    return int(value)


@dataclass(frozen=True)
class UnauthorisedPanel:
    """Worker-only geometry/names; never send paths or coordinates to Agent."""
    names: tuple
    locations: tuple
    policy: object
    report: object
    audit: object


def load_for_candidate(*,run_root,image_root,job_id,run_id,domain_id,domain_name,target_year,
                       source_csv_path,prior_card_path,registration):
    """Check registered panel files; returned artifact is explicitly unapproved.

    Registration is exactly {'version','job_id','run_id','target_year','seed',
    'sample_count','population_count','tile_count','artifacts'}.  Each artifact
    is the expected role path, content hash, inode identity, size and budget.
    Obtaining this object does not establish trusted source/provenance.  The
    root descriptor protects against symlink ancestor replacement while open.
    """
    keys={'version','job_id','run_id','target_year','seed','sample_count','population_count','tile_count','artifacts'}
    if not isinstance(registration,dict) or set(registration)!=keys or registration.get('version')!=VERSION:
        raise ValueError('invalid panel registration')
    if (any(not isinstance(v,str) or not IDENTITY.fullmatch(v) for v in (job_id,domain_id))
            or not isinstance(run_id,str) or not RUN_ID.fullmatch(run_id)
            or any(part in ('','.','..') for part in run_id.split('/'))):
        raise ValueError('invalid run identity')
    if not isinstance(domain_name,str) or not domain_name or len(domain_name)>200 or '\n' in domain_name:
        raise ValueError('invalid domain name')
    if registration['job_id']!=job_id or registration['run_id']!=run_id or registration['target_year']!=target_year:
        raise ValueError('run or target year identity mismatch')
    _integer(target_year,'target year',1900);_integer(registration['seed'],'seed')
    for key in ('sample_count','population_count','tile_count'):_integer(registration[key],key,1)
    if not isinstance(image_root,(str,Path)) or not os.path.isabs(image_root) or '..' in PurePosixPath(str(image_root)).parts:
        raise ValueError('absolute image root required')
    for value in (source_csv_path,prior_card_path):
        if not isinstance(value,str) or not value.startswith('/') or '\n' in value or '..' in PurePosixPath(value).parts:
            raise ValueError('invalid registered provenance path')
    artifacts=registration['artifacts']
    if not isinstance(artifacts,dict) or set(artifacts)!=ROLES:raise ValueError('incomplete artifact roles')
    records={k:_record(artifacts[k],k) for k in ROLES}
    # A file cannot serve multiple roles, even if its content happens to match.
    if len({(r['dev'],r['ino']) for r in records.values()})!=len(records):raise ValueError('shared artifact inode')
    fd=_root(run_root)
    try:blobs={k:_read(fd,records[k]) for k in ('names','locations','report','full','training')}
    finally:os.close(fd)
    names=_names(blobs['names']);full=_names(blobs['full']);training=_names(blobs['training'])
    periods={2018:'2018-2019',2019:'2018-2019',2024:'2024-2025',2025:'2024-2025'}
    if target_year not in periods:raise ValueError('unsupported target period')
    for name in full:
        match=IMAGE.fullmatch(name);folder=int(match['year']);actual=int(match['image_year'])
        if folder!=target_year or periods.get(actual)!=periods[target_year]:
            raise ValueError('target image period mismatch')
    if not set(training).issubset(full) or not set(names).issubset(full):raise ValueError('panel outside registered frame')
    if len(names)!=registration['sample_count'] or len(full)!=registration['population_count']:
        raise ValueError('panel registration counts mismatch')
    try:report=json.loads(blobs['report'])
    except (ValueError,UnicodeDecodeError) as exc:raise ValueError('invalid report JSON') from exc
    if not isinstance(report,dict) or set(report)!=REPORT_KEYS or report.get('version')!=PANEL_VERSION:
        raise ValueError('unexpected report schema')
    if any(report.get(k)!=v for k,v in REPORT_TEXT.items()):raise ValueError('unrecognised report prose')
    if type(report.get('requested_ratio')) not in (int,float) or not math.isfinite(report['requested_ratio']) or report['requested_ratio']<=0:
        raise ValueError('invalid sampling ratio')
    for key,value in (('seed',registration['seed']),('sample_count',len(names)),('population_count',len(full)),
                      ('training_count',len(training)),('tile_count',registration['tile_count'])):
        if type(report.get(key)) is not int or report[key]!=value:raise ValueError('report identity/count mismatch')
    for field,role in (('names_sha256','names'),('locations_csv_sha256','locations'),
                       ('population_list_sha256','full'),('training_list_sha256','training')):
        if report[field]!=records[role]['sha256']:raise ValueError('report artifact digest mismatch')
    # Referenced names/locations must be the same registered run-owned files.
    if report['names_file']!=str(Path(run_root)/records['names']['path']) or report['locations_csv']!=str(Path(run_root)/records['locations']['path']):
        raise ValueError('report refers to different artifacts')
    try:
        reader=csv.DictReader(io.StringIO(blobs['locations'].decode('utf-8-sig'),newline=''))
        if tuple(reader.fieldnames or ())!=CSV_COLUMNS:raise ValueError('unexpected locations columns')
        rows=list(reader)
    except (UnicodeDecodeError,csv.Error) as exc:raise ValueError('invalid locations CSV') from exc
    if len(rows)!=len(names):raise ValueError('panel names/locations count mismatch')
    frame=Counter(IMAGE.fullmatch(n)['id'] for n in full)
    counts=Counter(IMAGE.fullmatch(n)['id'] for n in names)
    if len(frame)!=registration['tile_count'] or set(counts)!=set(frame):raise ValueError('incomplete tile coverage')
    reduced=[];overlap=0
    for name,row in zip(names,rows):
        if None in row or any(value is None for value in row.values()):raise ValueError('malformed locations row')
        match=IMAGE.fullmatch(name);tile=match['id'];patch=tile+'_'+match['col']+'_'+match['row']
        if (row['adm1_id']!=domain_id or row['adm1_name']!=domain_name or row['year']!=str(target_year)
                or row['tile_id']!=tile or row['patch_id']!=patch or row['tif_path']!=str(Path(image_root)/name)
                or row['source_csv_path']!=source_csv_path or row['prior_card_path']!=prior_card_path):
            raise ValueError('locations identity mismatch')
        if row['overlaps_training'] not in ('True','False'):raise ValueError('invalid training overlap')
        expected_overlap=name in training
        if (row['overlaps_training']=='True')!=expected_overlap:raise ValueError('training overlap mismatch')
        overlap+=int(expected_overlap)
        n=_count_string(row['population_tile_patches'],'population count')
        q=_count_string(row['sampled_tile_patches'],'sample count')
        w=_number(row['expansion_weight'],'expansion weight')
        if n!=frame[tile] or q!=counts[tile] or not q or not math.isclose(w,n/q,rel_tol=1e-6,abs_tol=1e-12):
            raise ValueError('stratum population/weight mismatch')
        coords={k:_number(row[k],k) for k in ('west','south','east','north','center_lon','center_lat')}
        if not (-180<=coords['west']<coords['east']<=180 and -90<=coords['south']<coords['north']<=90):
            raise ValueError('unsupported geographic bounds')
        if not math.isclose(coords['center_lon'],(coords['west']+coords['east'])/2,abs_tol=1e-9) or not math.isclose(coords['center_lat'],(coords['south']+coords['north'])/2,abs_tol=1e-9):
            raise ValueError('location center mismatch')
        # Only these fixed fields are passed to the worker; the text columns,
        # original report prose and any unverified external data are discarded.
        reduced.append(MappingProxyType({'tile_id':tile,'relative_tif':name,'tif_path':row['tif_path'],
            **{k:coords[k] for k in ('west','south','east','north')},'population_tile_patches':n,
            'sampled_tile_patches':q,'expansion_weight':w}))
    if type(report.get('overlap_training_count')) is not int or report['overlap_training_count']!=overlap:
        raise ValueError('report training overlap mismatch')
    alloc=report.get('tile_allocation')
    if not isinstance(alloc,list) or len(alloc)!=len(frame):raise ValueError('invalid tile allocation')
    seen=set()
    for item in alloc:
        if not isinstance(item,dict) or set(item)!=ALLOCATION_KEYS:raise ValueError('invalid allocation schema')
        tile=item['tile_id']
        if not isinstance(tile,str) or tile in seen or tile not in frame:raise ValueError('invalid allocation tile')
        seen.add(tile)
        for field in ('population_count','eligible_count','sample_count'):_integer(item[field],field)
        if item['population_count']!=frame[tile] or item['sample_count']!=counts[tile] or item['eligible_count']<item['sample_count']:
            raise ValueError('allocation count mismatch')
        if type(item['expansion_weight']) not in (int,float) or not math.isfinite(item['expansion_weight']) or not math.isclose(item['expansion_weight'],frame[tile]/counts[tile],rel_tol=1e-6,abs_tol=1e-12):
            raise ValueError('allocation weight mismatch')
    policy=MappingProxyType({'version':PANEL_VERSION,'sample_count':len(names),'population_count':len(full),
                             'tile_count':len(frame),'seed':registration['seed'],
                             'scope':'fixed_tile_stratified_sample_valid_pixels','estimated_population_distribution':True})
    audit=MappingProxyType({'version':VERSION,'job_id':job_id,'run_id':run_id,
          'training_authorized':False,'role_and_lineage_authenticated':False,
          'image_contents_authenticated':False,'artifact_count':len(blobs)})
    report_out=MappingProxyType({'estimate_limits':'Unlabelled panel describes sampled predictions, not target accuracy.'})
    return UnauthorisedPanel(tuple(names),tuple(reduced),policy,report_out,audit)


def authorize_training(*_args,**_kwargs):
    raise RuntimeError('no authenticated panel producer/lineage or production release')
