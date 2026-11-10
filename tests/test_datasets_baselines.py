"""Tests for datasets.py (MONK's loader) and baselines.py."""

import itertools
import random
import unittest

from baselines import DecisionTree, MajorityLabel, NearestNeighbour
from bool_net import all_inputs, evaluate
from datasets import MONKS_ARITIES, MONKS_CONCEPTS, feature_names, load_monks, one_hot


def _domain():
    return {one_hot(a): a for a in
            itertools.product(*[range(1, k + 1) for k in MONKS_ARITIES])}


class TestMonks(unittest.TestCase):

    def test_sizes_and_encoding(self):
        sizes = {1: 124, 2: 169, 3: 122}
        for p in (1, 2, 3):
            tr, te = load_monks(p)
            self.assertEqual(len(tr), sizes[p])
            self.assertEqual(len(te), 432)
            for x in list(tr) + list(te):
                self.assertEqual(len(x), 17)
                # exactly one bit set per attribute block
                i = 0
                for k in MONKS_ARITIES:
                    self.assertEqual(sum(x[i:i + k]), 1)
                    i += k
        self.assertEqual(len(feature_names()), 17)

    def test_labels_match_documented_concepts(self):
        dom = _domain()
        self.assertEqual(len(dom), 432)
        for p in (1, 2, 3):
            tr, te = load_monks(p)
            c = MONKS_CONCEPTS[p]
            self.assertEqual(set(te), set(dom))          # test = full domain
            self.assertTrue(all(c(dom[x]) == y for x, y in te.items()))
            noisy = sum(c(dom[x]) != y for x, y in tr.items())
            self.assertEqual(noisy, 6 if p == 3 else 0)  # MONK-3: 5% noise


class TestBaselines(unittest.TestCase):

    def setUp(self):
        rng = random.Random(0)
        rows = all_inputs(6)
        self.data = {x: rng.randint(0, 1) for x in rng.sample(rows, 40)}

    def test_majority_label(self):
        m = MajorityLabel({(0,): 1, (1,): 1, (2,): 0})
        self.assertEqual(m((5,)), 1)

    def test_nn_and_unpruned_tree_fit_consistent_training_data(self):
        for model in (NearestNeighbour(self.data), DecisionTree(self.data)):
            tr, _ = evaluate(model, self.data, {}, verbose=False)
            self.assertEqual(tr, 1.0, type(model).__name__)

    def test_tree_learns_simple_concept_exactly(self):
        fn = lambda x: int(x[0] and not x[3])
        full = {x: fn(x) for x in all_inputs(6)}
        tree = DecisionTree(full)
        self.assertTrue(all(tree(x) == y for x, y in full.items()))
        self.assertLessEqual(tree.n_leaves(), 3)

    def test_depth_limit(self):
        stump = DecisionTree(self.data, max_depth=1)
        self.assertLessEqual(stump.n_leaves(), 2)


if __name__ == "__main__":
    unittest.main()
