"""Partial history reads must not create or overwrite complete-query views."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import complete_history_store as store


class PartialQueryCSV(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / 'evidence_store/agent_queries/epoch_000005.csv'
        self.reasoning = {'action': 'KEEP', 'observations': 'Full original reasoning.'}
        self.values = {'request': {'evidence': {}}, 'response': self.reasoning,
                       'decision': {'action': 'KEEP', 'intervention': {}}}
        self.commit = {'route': 'FULL_REVIEW', 'action': 'KEEP', 'decision_kind': 'agent'}
        self.patcher = patch.object(store, '_transaction', return_value=(self.values, {'decision_sha256': 'a'*64}))
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def read(self, include):
        return store.review_data(self.root, 'fixture', 'fixture/run', 5, self.commit, include_queries=include)

    def test_partial_read_keeps_existing_complete_query_csv_byte_identical(self):
        original = [{'arguments': {'kinds': ['initialization']}, 'recorded_epochs': [0]}]
        store._csv_view(self.path, original)
        before = self.path.read_bytes()
        self.read(False)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertTrue((self.root / 'evidence_store/agent_reasoning/epoch_000005.csv').is_file())

    def test_partial_read_does_not_poison_future_complete_view(self):
        self.read(False)
        self.assertFalse(self.path.exists())
        store._csv_view(self.path, [{'tool': 'query_training_history'}])
        self.read(False)
        self.assertFalse((self.root / 'evidence_store/reviews/epoch_000005.csv').exists())

    def test_complete_read_still_publishes_and_verifies_query_view(self):
        self.read(True)
        before = self.path.read_bytes()
        self.read(False)
        self.read(True)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertTrue((self.root / 'evidence_store/reviews/epoch_000005.csv').is_file())

    def test_immutable_csv_mismatch_is_still_rejected(self):
        store._csv_view(self.path, [{'arguments': {'start_epoch': 0}}])
        with self.assertRaisesRegex(ValueError, 'immutable evidence record differs'):
            store._csv_view(self.path, [{'arguments': {'start_epoch': 1}}])


if __name__ == '__main__':
    unittest.main()
