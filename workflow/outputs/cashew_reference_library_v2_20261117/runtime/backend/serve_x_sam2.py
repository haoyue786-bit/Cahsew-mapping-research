"""Official X-AnyLabeling SAM2 plugin with configuration routing and observation only."""
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OFFICIAL = Path('@ASSETS@/outputs/cashew_sample_assist_on_demand_context_20261116/_tools/remote_model_screen/X-AnyLabeling-Server-main')
WEIGHTS = Path('@ASSETS@/outputs/tgo_visual_annotation_v4_20261115/_tools/sam2_weights/sam2.1_hiera_large.pt')
os.environ['XANYLABELING_SERVER_CONFIG'] = str(HERE / 'configs/server.yaml')
os.environ['XANYLABELING_MODELS_CONFIG'] = str(HERE / 'configs/models.yaml')
sys.path.insert(0, str(OFFICIAL))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def safe(value):
    value = str(value)
    if not value or value in ['.', '..'] or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.' for c in value):
        raise ValueError('Use simple case/unit/revision IDs for recording')
    return value

from app.core.registry import ModelRegistry
from app.models.segment_anything_2 import SegmentAnything2

PLUGIN_FILE = Path(inspect.getfile(SegmentAnything2))
PLUGIN_SHA = sha(PLUGIN_FILE)
ORIGINAL_LOAD = SegmentAnything2.load
ORIGINAL_MARKS = SegmentAnything2._predict_with_marks
LOCAL = threading.local()

def observed_load(self):
    started = time.monotonic()
    ORIGINAL_LOAD(self)
    original_predict = self.predictor.predict
    original_set_image = self.predictor.set_image
    self.observed_predict_calls = 0
    self.observed_set_image_calls = 0

    def observed_set_image(image, *args, **kwargs):
        assert image.shape == (1024, 1024, 3), 'Reference curation requires the complete1024 AEF footprint'
        self.observed_set_image_calls += 1
        return original_set_image(image, *args, **kwargs)

    def observed_predict(*args, **kwargs):
        result = original_predict(*args, **kwargs)
        masks, scores, low_res_logits = result
        self.observed_predict_calls += 1
        record = LOCAL.record
        record['predict_calls_total'] = self.observed_predict_calls
        record['set_image_calls_total'] = self.observed_set_image_calls
        record['mask_shape'] = list(masks.shape)
        record['predicted_geometry_IoU'] = scores.tolist()
        record['multimask_output'] = bool(kwargs.get('multimask_output', False))
        record['mask_input_used'] = kwargs.get('mask_input') is not None
        record['point_coords'] = kwargs.get('point_coords').tolist() if kwargs.get('point_coords') is not None else None
        record['point_labels'] = kwargs.get('point_labels').tolist() if kwargs.get('point_labels') is not None else None
        record['box'] = kwargs.get('box').tolist() if kwargs.get('box') is not None else None
        folder = LOCAL.folder
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / 'predictor_raw.npz', masks=masks, scores=scores, low_res_logits=low_res_logits)
        batches = masks[None] if masks.ndim == 3 else masks
        score_batches = scores[None] if scores.ndim == 1 else scores
        for batch, planes in enumerate(batches):
            best = int(np.argmax(score_batches[batch])) if record['multimask_output'] else 0
            for index, mask in enumerate(planes):
                Image.fromarray(mask.astype('uint8') * 255).save(folder / f'batch{batch}_mask{index}.png')
            Image.fromarray(planes[best].astype('uint8') * 255).save(folder / f'batch{batch}_selected_raw.png')
        record['raw_npz_sha256'] = sha(folder / 'predictor_raw.npz')
        write(folder / 'PREDICTOR_RECEIPT.json', record)
        return result

    self.predictor.set_image = observed_set_image
    self.predictor.predict = observed_predict
    write(HERE / 'MODEL_LOAD_RECEIPT.json', {
        'status': 'loaded', 'pid': os.getpid(), 'model_id': 'segment_anything_2',
        'implementation_class': SegmentAnything2.__module__ + '.' + SegmentAnything2.__name__,
        'official_plugin_path': str(PLUGIN_FILE), 'official_plugin_sha256': PLUGIN_SHA,
        'predictor_source_path': inspect.getfile(type(self.predictor)),
        'predictor_source_sha256': sha(inspect.getfile(type(self.predictor))),
        'weights': str(WEIGHTS), 'weights_sha256': sha(WEIGHTS),
        'torch_device': str(self.device), 'CUDA_VISIBLE_DEVICES': os.getenv('CUDA_VISIBLE_DEVICES'),
        'load_seconds': time.monotonic() - started, 'config': self.config if hasattr(self, 'config') else self.params,
        'GT_read': False, 'official_sources_modified': False,
        'adaptation': 'Isolated configuration directory and observation-only raw predictor recording; official load/predict/shape conversion unchanged',
    })

def observed_marks(self, image, marks, params):
    assert image.shape == (1024, 1024, 3), 'Reference curation requires the complete1024 AEF footprint'
    folder = ROOT / 'raw_sam2' / safe(params.get('record_case_id', 'UNSPECIFIED')) / safe(params.get('record_unit_id', 'UNSPECIFIED')) / safe(params.get('record_revision', '0'))
    if (folder / 'OFFICIAL_RESPONSE.json').exists():
        raise ValueError('Preserve completed revision; use a new record_revision')
    image_key = self._get_image_key(image, params)
    LOCAL.folder = folder
    LOCAL.record = {'case_id': params.get('record_case_id'), 'unit_id': params.get('record_unit_id'),
        'revision': params.get('record_revision', '0'), 'image_id': params.get('image_id'),
        'marks': marks, 'image_shape': list(image.shape), 'RGB_pixel_sha256': hashlib.sha256(image[:, :, ::-1].tobytes()).hexdigest(),
        'embedding_cache_hit': self.image_embedding_cache.get(image_key) is not None,
        'official_plugin_sha256': PLUGIN_SHA, 'GT_read': False, 'recording_changes_predictions': False,
        'scores_are_geometric_not_semantic': True, 'query_tiles': False}
    started = time.monotonic()
    result = ORIGINAL_MARKS(self, image, marks, params)
    serial = {**result, 'shapes': [s.model_dump() if hasattr(s, 'model_dump') else s for s in result['shapes']]}
    write(folder / 'OFFICIAL_RESPONSE.json', serial)
    LOCAL.record['wall_seconds'] = time.monotonic() - started
    LOCAL.record['official_shape_count'] = len(result['shapes'])
    LOCAL.record['official_polygon_conversion'] = 'RETR_EXTERNAL plus official contour filtering/approximation; raw masks retained independently with holes'
    write(folder / 'PREDICTOR_RECEIPT.json', LOCAL.record)
    with (HERE / 'PREDICTION_EVENTS.jsonl').open('a') as f:
        f.write(json.dumps({'epoch': time.time(), 'folder': str(folder), **LOCAL.record}) + '\n')
    return result

SegmentAnything2.load = observed_load
SegmentAnything2._predict_with_marks = observed_marks

import app.main as main

# Redirect only configuration-file location; the official registry, executor,
# HTTP routes and registered SegmentAnything2 implementation remain in use.
main.ModelRegistry = lambda config_dir, models_config_path=None: ModelRegistry(HERE / 'configs', models_config_path=models_config_path)

if __name__ == '__main__':
    import torch
    import uvicorn
    torch.set_num_threads(4)
    uvicorn.run(main.app, host='127.0.0.1', port=19574, workers=1)
