"""Intervention response views are immutable at their own observation cutoff."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import training_recall_v3


class CutoffViews(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'journal/epoch_000005').mkdir(parents=True)
        (self.root / 'journal/epoch_000006').mkdir()
        (self.root / 'evidence_store/snapshots').mkdir(parents=True)
        self.commit = {'decision_kind': 'agent', 'route': 'FULL_REVIEW'}
        for e in (5, 6):
            (self.root / 'journal' / ('epoch_%06d' % e) / 'commit.json').write_text(json.dumps(self.commit))
        self.snapshot = {'training': {'cls': .4}, 'controls': {'main_lr': .000002}}
        (self.root / 'evidence_store/snapshots/epoch_000006.json').write_text(json.dumps(self.snapshot))
        self.values = {'response': {'diagnosis': 'Original history', 'intervention_review': []},
                       'decision': {'action': 'KEEP', 'intervention': {}}}
        for target, kwargs in [('complete_history_store._identity', {'return_value': {}}),
                               ('complete_history_store._transaction', {'return_value': (self.values, {'decision_sha256': 'a'*64})})]:
            p = patch(target, **kwargs)
            p.start()
            self.addCleanup(p.stop)

    def read(self, cutoff):
        return training_recall_v3.ledger(self.root, 'fixture', 'fixture/run', 5, cutoff)

    def test_later_response_uses_a_distinct_csv_without_changing_old_view(self):
        first = self.read(6)
        path = self.root / 'evidence_store/interventions/epoch_000005_before_000006.csv'
        original = path.read_bytes()
        second = self.read(7)
        self.assertEqual(first['subsequent_observations'], [])
        self.assertEqual([x['epoch'] for x in second['subsequent_observations']], [6])
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(self.read(6), first)
        self.assertEqual(self.read(7), second)

    def test_legacy_view_is_retained_and_does_not_block_new_cutoffs(self):
        path = self.root / 'evidence_store/interventions/epoch_000005.csv'
        path.parent.mkdir()
        path.write_bytes(b'original legacy view\n')
        self.read(6)
        self.read(7)
        self.assertEqual(path.read_bytes(), b'original legacy view\n')

    def test_same_cutoff_still_rejects_changed_source_data(self):
        self.read(7)
        self.snapshot['training']['cls'] = .9
        (self.root / 'evidence_store/snapshots/epoch_000006.json').write_text(json.dumps(self.snapshot))
        with self.assertRaisesRegex(ValueError, 'immutable evidence record differs'):
            self.read(7)


if __name__ == '__main__':
    unittest.main()
