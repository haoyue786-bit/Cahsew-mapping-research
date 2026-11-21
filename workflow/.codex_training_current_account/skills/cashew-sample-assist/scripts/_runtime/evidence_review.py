"""Repair missed visual inputs on the same annotation thread before publishing."""
from collections import Counter
from pathlib import Path

from . import transport
from .common import read, write
from .vision_audit import image_delivery_report


def reconcile_annotation_images(session, context, initial_dir, result, registry, schema, review_dir=None):
    directory = Path(initial_dir)
    root = Path(review_dir) if review_dir else directory / 'evidence_review'
    validation = read(directory / 'final_validation.json')
    all_images = dict(registry)
    requested = set()
    attempts = Counter()
    sequence = 0
    last_turn = read(directory / 'status.json')['turn_id']
    while validation['disposition'] == 'publish':
        comp = read(Path(validation['composition_folder']) / 'COMPOSITION.json')
        required = set(comp['required_image_ids']) | requested
        if required - all_images.keys():
            raise ValueError('Selected composition cites an unregistered image')
        # Host delivery receipts can include images that the calling bridge did not forward.
        seen = image_delivery_report(session.directory, all_images, required, through_turn=last_turn)
        missing = seen['missing_image_ids']
        sequence += 1
        turn_dir = root / f'{sequence:03d}'
        manifest = turn_dir / 'REDELIVERY.json'
        if not missing and not manifest.exists():
            break
        if manifest.exists():
            previous = read(manifest)
            if previous['from_turn'] != last_turn:
                raise RuntimeError('Evidence review belongs to a different decision turn')
            batch = previous['image_ids']
        else:
            if any(attempts[i] >= 2 for i in missing):
                raise RuntimeError('Direct image input still unverified after individual redelivery: ' + ', '.join(missing))
            batch = missing[:1] if any(attempts[i] for i in missing) else missing[:4]
            write(manifest, dict(from_turn=last_turn, image_ids=batch, composition_folder=validation['composition_folder']))
        attempts.update(batch)
        requested.update(batch)
        state_path = turn_dir / 'status.json'
        state = read(state_path) if state_path.exists() else {}
        if state and state.get('status') not in ('completed', 'prepared'):
            raise RuntimeError(f'Existing evidence turn at {turn_dir} requires reconciliation before retry')
        if not state:
            choices = {f"{context['case']}_{u['unit_id']}_r{u['revision']}_sam": u for u in comp['accepted_units']}
            info = dict(case_id=context['case'], evidence_redelivery=True, prior_decision_turn=last_turn,
                        current_submission_id=comp.get('annotation_id', comp.get('preview_image_id')),
                        image_ids_for_review=batch, mask_choices={i: choices[i] for i in batch if i in choices})
            prompt = ("Continue the same sample. The attached original images now enter your visual input directly. "
                      "Inspect their actual contents and check the referenced mask choices against the current RGB. "
                      "Keep the saved submission ID if its classes and geometry are supported. Use the original "
                      "annotation tools to correct any real discrepancy and return the resulting ID. "
                      "Use the same final schema and spatial review policy as the previous turn.")
            transport.prepare_turn(session.thread, turn_dir, prompt, [all_images[i] for i in batch],
                                   info, schema, prior_images=[all_images[i] for i in seen['verified_image_ids']])
        if state.get('status') == 'completed':
            if state['thread_id'] != session.thread_id:
                raise RuntimeError('The saved evidence decision belongs to another thread')
            result, received = read(turn_dir / 'result.json'), read(turn_dir / 'image_registry.json')
        else:
            result, received = session.run_turn(turn_dir, context, timeout_seconds=1800)
        all_images.update(received)
        last_turn = read(state_path)['turn_id']
        proof = image_delivery_report(session.directory, all_images, batch, turn_id=last_turn, through_turn=last_turn)
        write(turn_dir / 'DIRECT_IMAGE_AUDIT.json', proof)
        validation = read(turn_dir / 'final_validation.json')
        if proof['missing_image_ids']:
            # Only a turn that actually received the missed images may confirm an annotation.
            if validation['disposition'] != 'publish':
                raise RuntimeError('The evidence turn changed image disposition before receiving the required images')
            if any(attempts[i] >= 2 for i in proof['missing_image_ids']):
                raise RuntimeError('Required image unavailable as direct visual input')
    validation = dict(validation, decision_turn_id=last_turn)
    if requested:
        write(root / 'DECISION.json', dict(decision_turn_id=last_turn, decision=result, validation=validation,
                                           direct_image_ids=sorted(requested)))
    return result, all_images, validation
