"""
Regression tests for bool_net.py.

Run from the project root:
    python3 -m unittest discover -s tests -v
(pytest also works: `pytest tests/`.)

These are correctness guards, kept fast (well under a minute). The
experiment harness that produces the README's numbers lives in test.py.
"""

import hashlib
import json
import random
import unittest

from bool_net import (
    BooleanNetwork, all_inputs, compute_loss, compute_objective, make_dataset,
    majority, parity, prune, train, train_with_restarts, xor2, custom,
)


def brute_force_derivs(net, x):
    """∂f/∂w for every weight by literally flipping it and re-evaluating."""
    base = net(x)
    out = {}
    for wid in net.all_weight_ids():
        net.flip_weight(wid)
        out[wid] = int(net(x) != base)
        net.flip_weight(wid)
    return out


def weights_fingerprint(net):
    return hashlib.sha256(json.dumps(
        [[l.W for l in net.layers], net.out_w]).encode()).hexdigest()[:16]


def random_net(rng, n_inputs, widths):
    """A random net, then a few random flips so we also cover states that
    init alone never produces (e.g. all-zero columns, contradictions,
    disconnected outputs)."""
    net = BooleanNetwork(n_inputs, widths, seed=rng.randint(0, 2**31 - 1))
    wids = list(net.all_weight_ids())
    for w in rng.sample(wids, min(len(wids), rng.randint(0, 6))):
        net.flip_weight(w)
    return net


class TestBackwardExactness(unittest.TestCase):
    """backward() must equal the brute-force Boolean derivative exactly."""

    def test_matches_brute_force(self):
        rng = random.Random(1234)
        shapes = [(2, [3]), (2, [4, 4]), (3, [4]), (3, [5, 3]),
                  (3, [3, 3, 3]), (4, [4, 4]), (4, [6, 2, 3])]
        checked = 0
        for n, widths in shapes:
            for _ in range(15):
                net = random_net(rng, n, widths)
                for x in all_inputs(n):
                    expected = brute_force_derivs(net, x)
                    got = net.backward(x)
                    # backward() may omit zero-derivative layer weights.
                    for wid, d in expected.items():
                        self.assertEqual(
                            got.get(wid, 0), d,
                            f"shape={n},{widths} x={x} wid={wid}")
                    checked += len(expected)
        self.assertGreater(checked, 10_000)

    def test_backward_does_not_mutate_net(self):
        net = BooleanNetwork(3, [4, 4], seed=7)
        before = weights_fingerprint(net)
        for x in all_inputs(3):
            net.backward(x)
        self.assertEqual(weights_fingerprint(net), before)


class TestInitialization(unittest.TestCase):

    def test_no_dead_or_trivial_neurons_at_init(self):
        for seed in range(50):
            net = BooleanNetwork(4, [5, 4, 3], seed=seed)
            self.assertFalse(net.has_contradiction(), f"seed={seed}")
            for layer in net.layers:
                for j in range(layer.output_width):
                    self.assertTrue(any(layer.W[i][j] for i in range(layer.input_width)),
                                    f"seed={seed}: all-zero column")

    def test_same_seed_same_network(self):
        a = BooleanNetwork(4, [4, 4], seed=99)
        b = BooleanNetwork(4, [4, 4], seed=99)
        self.assertEqual(weights_fingerprint(a), weights_fingerprint(b))


class TestTraining(unittest.TestCase):

    def test_xor_and_parity3_converge(self):
        cases = [(2, xor2, [4, 4]), (3, parity, [4]), (3, parity, [6, 4])]
        for n, fn, widths in cases:
            data, _ = make_dataset(n, fn, observed_fraction=1.0, seed=0)
            for seed in range(5):
                net = train_with_restarts(n, widths, data, n_restarts=10,
                                          verbose=False, seed=seed)
                self.assertEqual(compute_loss(net, data), 0,
                                 f"{fn.__name__} {widths} seed={seed}")

    def test_training_is_reproducible(self):
        data, _ = make_dataset(4, custom, observed_fraction=0.75, seed=1)
        fps = set()
        for _ in range(2):
            net = train_with_restarts(4, [4, 4], data, n_restarts=3,
                                      verbose=False, seed=42)
            fps.add(weights_fingerprint(net))
        self.assertEqual(len(fps), 1)

    def test_no_contradictions_after_training(self):
        data, _ = make_dataset(3, parity, observed_fraction=1.0, seed=0)
        for seed in range(10):
            net = BooleanNetwork(3, [4, 4], seed=seed)
            train(net, data, verbose=False, seed=seed)
            self.assertFalse(net.has_contradiction(), f"seed={seed}")

    def test_unregularized_behavior_is_frozen(self):
        """Golden fingerprints of lam=0 training. The README's convergence
        numbers depend on this exact behavior; if a change is *meant* to
        alter training, re-run test.py, update the README, then update
        these values."""
        data, _ = make_dataset(3, parity, observed_fraction=1.0, seed=0)
        net = train_with_restarts(3, [6, 4], data, n_restarts=5,
                                  verbose=False, seed=0)
        explicit = train_with_restarts(3, [6, 4], data, n_restarts=5,
                                       verbose=False, seed=0, lam=0.0)
        self.assertEqual(weights_fingerprint(net), weights_fingerprint(explicit))
        self.assertEqual(weights_fingerprint(net), GOLDEN_PARITY3)


# Filled from the pre-regularization code (commit dbf2545) — see the test above.
GOLDEN_PARITY3 = "ae9a144d52673805"


class TestRegularization(unittest.TestCase):

    def test_objective(self):
        data, _ = make_dataset(3, majority, observed_fraction=1.0, seed=0)
        net = BooleanNetwork(3, [4, 4], seed=3)
        self.assertEqual(compute_objective(net, data, 0.0), compute_loss(net, data))
        self.assertAlmostEqual(compute_objective(net, data, 0.25),
                               compute_loss(net, data) + 0.25 * net.complexity())

    def test_state_roundtrip(self):
        net = BooleanNetwork(3, [4, 4], seed=3)
        state = net.get_state()
        fp = weights_fingerprint(net)
        for w in list(net.all_weight_ids())[:10]:
            net.flip_weight(w)
        net.set_state(state)
        self.assertEqual(weights_fingerprint(net), fp)

    def test_prune_keeps_fit_and_is_locally_minimal(self):
        for n, fn, widths in [(3, majority, [4, 4]), (3, parity, [6, 4])]:
            data, _ = make_dataset(n, fn, observed_fraction=1.0, seed=0)
            net = train_with_restarts(n, widths, data, verbose=False, seed=1)
            self.assertEqual(compute_loss(net, data), 0)
            before = net.complexity()
            removed = prune(net, data, lam=0.1)
            self.assertEqual(compute_loss(net, data), 0)
            self.assertEqual(net.complexity(), before - removed)
            # Locally minimal: no remaining active weight can be dropped
            # without costing a training error.
            for wid in list(net.all_weight_ids()):
                net.flip_weight(wid)
                removed_one = net.complexity() < before - removed
                loss = compute_loss(net, data)
                net.flip_weight(wid)
                if removed_one:
                    self.assertGreater(loss, 0, f"{wid} was removable")

    def test_regularized_training_converges_and_is_smaller(self):
        data, _ = make_dataset(3, parity, observed_fraction=1.0, seed=0)
        for seed in range(3):
            plain = train_with_restarts(3, [6, 4], data, n_restarts=5,
                                        verbose=False, seed=seed)
            reg = train_with_restarts(3, [6, 4], data, n_restarts=5,
                                      verbose=False, seed=seed, lam=0.1)
            self.assertEqual(compute_loss(reg, data), 0, f"seed={seed}")
            self.assertFalse(reg.has_contradiction())
            self.assertLess(reg.complexity(), plain.complexity(), f"seed={seed}")

    def test_scoring_restores_state(self):
        data, _ = make_dataset(3, majority, observed_fraction=1.0, seed=0)
        net = BooleanNetwork(3, [4, 4], seed=5)
        fp = weights_fingerprint(net)
        prune(net, data, lam=0.0)          # lam=0 must be a no-op
        self.assertEqual(weights_fingerprint(net), fp)


class TestDataset(unittest.TestCase):

    def test_split_is_disjoint_and_complete(self):
        tr, te = make_dataset(5, majority, observed_fraction=0.75, seed=3)
        self.assertEqual(len(tr), 24)
        self.assertEqual(len(te), 8)
        self.assertFalse(set(tr) & set(te))
        self.assertEqual(set(tr) | set(te), set(all_inputs(5)))
        for x, y in {**tr, **te}.items():
            self.assertEqual(y, majority(x))


if __name__ == "__main__":
    unittest.main()
