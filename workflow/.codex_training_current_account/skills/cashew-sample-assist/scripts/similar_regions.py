#!/usr/bin/env python3
"""Match land-unit masks to representative masks from the same registered RGB."""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from time import monotonic

import numpy as np
from PIL import Image

from _runtime.common import SKILL_ROOT, digest, read, write
from annotation_geometry import SIZE

PATCH_SIZE = 14
CELLS = SIZE // PATCH_SIZE
ENCODER_SIZE = CELLS * PATCH_SIZE
MIN_CORE_PATCHES = 3
MIN_SIMILARITY = .80
MIN_MARGIN = .03
MIN_AGREEMENT = .60
DINO_REPO = Path.home()/'.cache/torch/hub/facebookresearch_dinov2_main'
WEIGHTS = Path.home()/'.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth'
WEIGHTS_SHA256 = 'b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'


def local_model_runtime():
    default_python = (SKILL_ROOT.parents[2] /
                      'outputs/cashew_sample_assist_civ_a05_candidate_boxes_20261119/detector_env/bin/python')
    python = os.environ.get('CASHEW_FIELDS_PYTHON', str(default_python))
    gpu = os.environ.get('CASHEW_FIELDS_GPU', '2')
    return python, dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)


def encode(image):
    import torch
    import torch.nn.functional as F

    if digest(WEIGHTS) != WEIGHTS_SHA256:
        raise ValueError('DINOv2 ViT-S/14 checkpoint checksum differs')
    sys.path.insert(0, str(DINO_REPO))
    from dinov2.hub.backbones import dinov2_vits14

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = dinov2_vits14(pretrained=False)
    model.load_state_dict(torch.load(WEIGHTS, map_location='cpu', weights_only=True))
    model.eval().to(device)
    image = image.resize((ENCODER_SIZE, ENCODER_SIZE), Image.Resampling.BILINEAR)
    rgb = np.asarray(image, dtype=np.float32) / 255
    rgb = (rgb - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
    pixels = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.float16, enabled=device == 'cuda'):
        patches = model.forward_features(pixels)['x_norm_patchtokens'][0]
    return F.normalize(patches.float(), dim=-1).cpu().numpy()


def core_patches(mask):
    raster = Image.fromarray(mask.astype(np.uint8)).resize((ENCODER_SIZE, ENCODER_SIZE), Image.Resampling.NEAREST)
    coverage = np.asarray(raster).reshape(CELLS, PATCH_SIZE, CELLS, PATCH_SIZE).mean(axis=(1,3))
    return coverage.ravel() >= .8


def unit_vectors(features, mask):
    return features[core_patches(mask)]


def match_regions(features, anchors, candidates, *, permit_woody):
    codes = []
    prototypes = []
    for code, mask in sorted(anchors.items()):
        vectors = unit_vectors(features, mask)
        if len(vectors) >= MIN_CORE_PATCHES:
            mean = vectors.mean(axis=0)
            prototypes.append(mean / np.linalg.norm(mean))
            codes.append(code)

    rows = []
    for identity, mask in candidates.items():
        result = dict(id=identity, class_id=None)
        rows.append(result)
        if len(codes) < 2:
            result['reason'] = 'need_two_comparable_classes'
            continue
        if any(np.count_nonzero(mask & anchor) > .30 * min(mask.sum(), anchor.sum()) for anchor in anchors.values()):
            result['reason'] = 'overlapping_anchor'
            continue
        vectors = unit_vectors(features, mask)
        if len(vectors) < MIN_CORE_PATCHES:
            result['reason'] = 'insufficient_interior'
            continue
        prototype_matrix = np.asarray(prototypes)
        mean = vectors.mean(axis=0)
        coherence = np.linalg.norm(mean)
        if coherence == 0:
            result['reason'] = 'no_coherent_interior'
            continue
        scores = (mean / coherence) @ prototype_matrix.T
        order = np.argsort(scores)
        best, second = order[-1], order[-2]
        winner = codes[best]
        margin = scores[best] - scores[second]
        agreement = np.mean((vectors @ prototype_matrix.T).argmax(axis=1) == best)
        result.update(best_class=winner, similarity=round(float(scores[best]),5),
                      margin=round(float(margin),5), agreement=round(float(agreement),5))
        if winner in (1,2) and (not permit_woody or not {1,2}.issubset(codes)):
            result['reason'] = 'need_well_supported_cashew_and_mix_tree_in_dry_imagery'
        elif scores[best] < MIN_SIMILARITY or margin < MIN_MARGIN or agreement < MIN_AGREEMENT:
            result['reason'] = 'ambiguous_similarity'
        else:
            result.update(class_id=winner, reason='high_similarity')
    return rows


def read_mask(path):
    with Image.open(path) as image:
        mask = np.asarray(image.convert('L')) != 0
    if mask.shape != (SIZE, SIZE):
        raise ValueError('Matching masks must use the complete RGB dimensions')
    return mask


def match_detected_regions(context, anchor_masks, candidates, valid, *, permit_woody):
    """Run the local encoder once on the image behind the saved SAM2 masks."""
    folder = context['case_dir'] / 'region_matching'
    folder.mkdir(exist_ok=True)
    output = folder / 'REGION_MATCHES.json'
    if len(anchor_masks) < 2:
        report = dict(status='insufficient_representatives', matches=[])
        write(output, report)
        return report
    scene = context['scene']
    anchors = []
    for code, mask in sorted(anchor_masks.items()):
        path = folder / f'anchor_{code}.png'
        Image.fromarray((mask & valid).astype(np.uint8) * 255).save(path)
        anchors.append(dict(class_id=code, mask_path=str(path)))
    footprint = folder / 'valid.png'
    Image.fromarray(valid.astype(np.uint8) * 255).save(footprint)
    manifest = folder / 'INPUTS.json'
    write(manifest, dict(rgb=str(scene), rgb_sha256=digest(scene), anchors=anchors,
                         candidates=[dict(id=key, mask_path=str(path)) for key, path in candidates.items()],
                         valid_mask_path=str(footprint), permit_woody=permit_woody))
    python, env = local_model_runtime()
    log = folder / 'dinov2.log'
    try:
        with log.open('w') as stream:
            subprocess.run([python, str(Path(__file__).resolve()), str(manifest), str(output)],
                           env=env, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=180)
    except (OSError, subprocess.SubprocessError) as error:
        report = dict(status='local_matching_unavailable', reason=type(error).__name__, log=str(log), matches=[])
        write(output, report)
        return report
    return read(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    start = monotonic()
    manifest = read(args.manifest)
    with Image.open(manifest['rgb']) as source:
        if source.size != (SIZE, SIZE):
            raise ValueError('Matching expects the complete registered RGB')
        features = encode(source.convert('RGB'))
    valid = read_mask(manifest['valid_mask_path']) if manifest.get('valid_mask_path') else np.ones((SIZE,SIZE), bool)
    anchors = {a['class_id']: read_mask(a['mask_path']) & valid for a in manifest['anchors']}
    candidates = {a['id']: read_mask(a['mask_path']) & valid for a in manifest['candidates']}
    matches = match_regions(features, anchors, candidates, permit_woody=manifest['permit_woody'])
    write(args.output, dict(status='completed', model='DINOv2 ViT-S/14', weights_sha256=WEIGHTS_SHA256,
                            seconds=round(monotonic()-start,2), matches=matches))


if __name__ == '__main__':
    main()
