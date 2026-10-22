import unittest
import json
import numpy as np
import pandas as pd
from core import ROOT, Predictor, LABELS, clean_text, routed_predictions

class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = Predictor()
        cls.data = pd.read_csv(ROOT / 'data/campus_queries.csv')

    def test_disjoint_splits_and_minimum_examples(self):
        self.assertFalse(self.data.clean_text.duplicated().any())
        self.assertEqual(set(self.data.label), set(LABELS))
        self.assertTrue((self.data.groupby('label').size() >= 10).all())
        self.assertEqual(set(self.data.split), {'train', 'validation', 'test'})
        for split in self.data.split.unique():
            self.assertEqual(set(self.data[self.data.split == split].label), set(LABELS))

    def test_no_test_only_token_in_training_vocabulary(self):
        vec = self.predictor.pipeline.named_steps['tfidf']
        analyzer = vec.build_analyzer()
        train_terms = set(t for text in self.data[self.data.split == 'train'].text for t in analyzer(text))
        self.assertEqual(set(vec.vocabulary_), train_terms)

    def test_invalid_inputs(self):
        for value in ['', '   ', '1234!!!', None, 'a' * 1001]:
            with self.assertRaises(ValueError):
                self.predictor.predict(value)

    def test_unknown_vocabulary_fallback(self):
        result = self.predictor.predict('zzzxqv pppqqzz')
        self.assertEqual(result['category'], 'other')
        self.assertEqual(result['reason'], 'unseen_vocabulary')

    def test_serving_matches_evaluation(self):
        data = self.data[self.data.split == 'test']
        actual = [self.predictor.predict(t)['category'] for t in data.text]
        expected = routed_predictions(self.predictor.pipeline, data.text, self.predictor.bundle['threshold'])
        self.assertEqual(actual, list(expected))
        saved = pd.read_csv(ROOT / f"reports/{self.predictor.bundle['model']}_predictions.csv")
        self.assertEqual(actual, saved.prediction.tolist())

    def test_probability_contract(self):
        result = self.predictor.predict('Where is the library?')
        self.assertEqual(set(result['scores']), set(LABELS))
        self.assertAlmostEqual(sum(result['scores'].values()), 1.0)
        self.assertEqual(result, self.predictor.predict('Where is the library?'))

    def test_cleaning(self):
        self.assertEqual(clean_text('  CAMPUS!!  WiFi? '), 'campus wifi')

if __name__ == '__main__':
    unittest.main()
