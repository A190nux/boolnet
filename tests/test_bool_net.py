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
    BooleanNetwork, Ensemble, Objective, PackedData, all_inputs, compute_loss,
    compute_objective, error_mask, influence_rows, make_dataset, polish,
    total_influence, train_ensemble,
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

    def test_packed_backward_matches_rowwise_and_brute_force(self):
        rng = random.Random(4321)
        shapes = [(2, [3]), (2, [4, 4]), (3, [5, 3]), (3, [3, 3, 3]),
                  (4, [6, 2, 3]), (7, [5, 4])]          # n=7: 128-bit masks
        for n, widths in shapes:
            rows = all_inputs(n)
            for _ in range(10):
                net = random_net(rng, n, widths)
                data = {x: 0 for x in rng.sample(rows, rng.randint(1, len(rows)))}
                pd = PackedData(data)
                subset = rng.getrandbits(pd.n_rows)      # random row selection
                masks = net.backward_packed(pd, subset)
                for r, x in enumerate(pd.rows):
                    brute = brute_force_derivs(net, x)
                    rowwise = net.backward(x)
                    for wid in net.all_weight_ids():
                        got = (masks.get(wid, 0) >> r) & 1
                        want = brute[wid] if (subset >> r) & 1 else 0
                        self.assertEqual(got, want, f"n={n} {widths} x={x} {wid}")
                        if (subset >> r) & 1:
                            self.assertEqual(rowwise.get(wid, 0), want)

    def test_fast_pair_prefilter_matches_reference(self):
        from bool_net import _pair_prefilter, _pair_prefilter_reference
        rng = random.Random(606)
        checked = 0
        for n, widths in [(2, [3]), (3, [4, 4]), (4, [5, 3]), (5, [4, 3, 2])]:
            for _ in range(12):
                net = random_net(rng, n, widths)
                # Sometimes start from a state that already has a contradiction.
                if rng.random() < 0.3:
                    layer = net.layers[0]
                    m = layer.input_width // 2
                    j = rng.randrange(layer.output_width)
                    p = rng.randrange(m)
                    layer.W[p][j] = layer.W[p + m][j] = 1
                wids = list(net.all_weight_ids())
                active = sorted(rng.sample(wids, min(len(wids), rng.randint(2, 14))))
                fp = weights_fingerprint(net)
                for x in rng.sample(all_inputs(n), 3):
                    fast = _pair_prefilter(net, active, x)
                    ref = _pair_prefilter_reference(net, active, x)
                    self.assertEqual(fast, ref, f"{n} {widths} x={x}")
                    checked += len(active) * (len(active) - 1) // 2
                self.assertEqual(weights_fingerprint(net), fp)   # net restored
        self.assertGreater(checked, 2000)

    def test_backward_does_not_mutate_net(self):
        net = BooleanNetwork(3, [4, 4], seed=7)
        before = weights_fingerprint(net)
        for x in all_inputs(3):
            net.backward(x)
        self.assertEqual(weights_fingerprint(net), before)


class TestBitPacking(unittest.TestCase):
    """forward_packed / compute_loss must agree with per-row forward()."""

    def test_packed_forward_matches_rowwise(self):
        rng = random.Random(77)
        # n=7 gives 128 rows — wider than a machine word, so big-int
        # masks are exercised, not just small ones.
        for n, widths in [(2, [3]), (3, [4, 4]), (4, [5, 3, 2]), (7, [6, 4])]:
            rows = all_inputs(n)
            for _ in range(20):
                net = random_net(rng, n, widths)
                sub = rng.sample(rows, rng.randint(1, len(rows)))
                data = {x: rng.randint(0, 1) for x in sub}
                pd = PackedData(data)
                out = net.forward_packed(pd)
                for r, x in enumerate(pd.rows):
                    self.assertEqual((out >> r) & 1, net(x), f"n={n} x={x}")
                self.assertEqual(compute_loss(net, pd),
                                 sum(net(x) != y for x, y in data.items()))
                self.assertEqual(compute_loss(net, data), compute_loss(net, pd))
                # error rows come back in the dict's own order
                self.assertEqual(pd.rows_in(error_mask(net, pd)),
                                 [x for x, y in data.items() if net(x) != y])

    def test_empty_dataset(self):
        net = BooleanNetwork(3, [4], seed=0)
        self.assertEqual(compute_loss(net, {}), 0)


class TestInitialization(unittest.TestCase):

    def test_no_dead_or_trivial_neurons_at_init(self):
        for seed in range(50):
            net = BooleanNetwork(4, [5, 4, 3], seed=seed)
            self.assertFalse(net.has_contradiction(), f"seed={seed}")
            for layer in net.layers:
                for j in range(layer.output_width):
                    self.assertTrue(any(layer.W[i][j] for i in range(layer.input_width)),
                                    f"seed={seed}: all-zero column")

    def test_density(self):
        # None must reproduce the original init exactly (same RNG sequence):
        # fingerprint taken from the original code, commit dbf2545.
        self.assertEqual(
            weights_fingerprint(BooleanNetwork(5, [6, 3], seed=11, density=None)),
            "efcbc6fb3ee77804")
        # The default is sparse (Step 20).
        self.assertEqual(
            weights_fingerprint(BooleanNetwork(5, [6, 3], seed=11)),
            weights_fingerprint(BooleanNetwork(5, [6, 3], seed=11, density=0.05)))
        sizes = {}
        for d in (0.05, 0.3, 1.0):
            total = 0
            for seed in range(30):
                net = BooleanNetwork(10, [8, 4], seed=seed, density=d)
                self.assertFalse(net.has_contradiction())
                for layer in net.layers:
                    m = layer.input_width // 2
                    for j in range(layer.output_width):
                        used = sum(layer.W[i][j] for i in range(layer.input_width))
                        self.assertGreaterEqual(used, 1)          # never all-zero
                        if d == 1.0:
                            self.assertEqual(used, m)             # every signal used
                total += net.complexity()
            sizes[d] = total
        self.assertLess(sizes[0.05], sizes[0.3])
        self.assertLess(sizes[0.3], sizes[1.0])

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

    def test_unconverged_training_returns_best_state(self):
        """A run that never reaches zero loss must return the best state it
        committed to, not whatever state the last random perturbation left
        behind. train() calls error_mask() exactly once per step, on the
        committed state (never on probe flips), so wrapping it records the
        loss trajectory."""
        import bool_net
        rng = random.Random(3)
        data = {x: rng.randint(0, 1) for x in all_inputs(6)}   # random labels
        original = bool_net.error_mask
        caught_a_late_regression = False
        try:
            for seed in range(20):
                seen = []
                def recording(net, pd, _seen=seen):
                    m = original(net, pd)
                    _seen.append(m.bit_count())
                    return m
                bool_net.error_mask = recording
                net = BooleanNetwork(6, [3], seed=seed)          # too small to fit
                train(net, data, max_steps=200, verbose=False, seed=seed)
                bool_net.error_mask = original
                final = compute_loss(net, data)
                self.assertEqual(final, min(seen), f"seed={seed}")
                caught_a_late_regression |= seen[-1] > min(seen)
        finally:
            bool_net.error_mask = original
        # Make sure the scenario actually exercises the fix.
        self.assertTrue(caught_a_late_regression)

    def test_neutral_moves(self):
        rng = random.Random(8)
        data = {x: rng.randint(0, 1) for x in rng.sample(all_inputs(6), 40)}
        with self.assertRaises(ValueError):
            train(BooleanNetwork(6, [4], seed=0), data, verbose=False, neutral="bogus")
        for mode in ("remove", "any"):
            fps = set()
            for _ in range(2):
                net = BooleanNetwork(6, [6, 3], seed=1)
                start = compute_loss(net, data)
                train(net, data, max_steps=150, verbose=False, seed=1, neutral=mode)
                self.assertFalse(net.has_contradiction(), mode)
                self.assertLessEqual(compute_loss(net, data), start, mode)
                fps.add(weights_fingerprint(net))
            self.assertEqual(len(fps), 1, f"{mode} not reproducible")

    def test_neutral_removals_only_ever_remove(self):
        """In "remove" mode a neutral move must never add a connection."""
        import bool_net
        rng = random.Random(2)
        data = {x: rng.randint(0, 1) for x in all_inputs(5)}
        net = BooleanNetwork(5, [5, 3], seed=4)
        flips = []
        original = bool_net._log_flip
        bool_net._log_flip = lambda n, wid, score: flips.append(
            (wid, bool_net._weight_value(n, wid), score))
        try:
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()):
                train(net, data, max_steps=150, verbose=True, seed=4, neutral="remove")
        finally:
            bool_net._log_flip = original
        neutral = [(w, v) for w, v, score in flips if score == 0]
        self.assertTrue(neutral, "scenario never made a neutral move")
        self.assertTrue(all(v == 0 for _, v in neutral), neutral)

    def test_no_contradictions_after_training(self):
        data, _ = make_dataset(3, parity, observed_fraction=1.0, seed=0)
        for seed in range(10):
            net = BooleanNetwork(3, [4, 4], seed=seed)
            train(net, data, verbose=False, seed=seed)
            self.assertFalse(net.has_contradiction(), f"seed={seed}")

    def test_unregularized_behavior_is_frozen(self):
        """Golden fingerprints of lam=0 training. The README's numbers
        depend on this exact behavior; if a change is *meant* to alter
        training, re-run test.py, update the README, then update these.

        Two are pinned: the original dense init (density=None), which every
        README number up to Step 19 was measured with, and the current
        default (sparse, density=0.05, since Step 20)."""
        data, _ = make_dataset(3, parity, observed_fraction=1.0, seed=0)
        dense = train_with_restarts(3, [6, 4], data, n_restarts=5,
                                    verbose=False, seed=0, density=None)
        explicit = train_with_restarts(3, [6, 4], data, n_restarts=5,
                                       verbose=False, seed=0, lam=0.0, density=None)
        self.assertEqual(weights_fingerprint(dense), weights_fingerprint(explicit))
        self.assertEqual(weights_fingerprint(dense), GOLDEN_PARITY3_DENSE)
        default = train_with_restarts(3, [6, 4], data, n_restarts=5,
                                      verbose=False, seed=0)
        self.assertEqual(weights_fingerprint(default), GOLDEN_PARITY3_DEFAULT)


# Dense init: from the pre-regularization code (commit dbf2545).
GOLDEN_PARITY3_DENSE = "ae9a144d52673805"
# Sparse default (density=0.05): recorded when it became the default (Step 20).
GOLDEN_PARITY3_DEFAULT = "9dd58b20208fc8ed"


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


def brute_influence(net, n):
    rows = all_inputs(n)
    flips = sum(net(x) != net(x[:i] + (1 - x[i],) + x[i+1:])
                for x in rows for i in range(n))
    return flips / len(rows)


class TestInfluence(unittest.TestCase):

    def test_packed_influence_matches_brute_force(self):
        rng = random.Random(55)
        for n, widths in [(2, [3]), (3, [4, 4]), (4, [5, 3, 2]), (7, [6, 4])]:
            pd = influence_rows(n)
            self.assertEqual(pd.n_rows, 2 ** n)
            for _ in range(15):
                net = random_net(rng, n, widths)
                self.assertAlmostEqual(total_influence(net, pd),
                                       brute_influence(net, n))

    def test_constant_function_has_zero_influence(self):
        net = BooleanNetwork(4, [4], seed=0)
        net.out_w = [0] * 4                     # output is constantly 0
        self.assertEqual(total_influence(net, influence_rows(4)), 0.0)

    def test_sampled_rows_for_large_n(self):
        pd = influence_rows(14, max_rows=500, seed=1)
        self.assertLessEqual(pd.n_rows, 500)
        self.assertEqual(pd.rows, influence_rows(14, max_rows=500, seed=1).rows)

    def test_objective_terms(self):
        data, _ = make_dataset(4, majority, observed_fraction=0.75, seed=0)
        net = BooleanNetwork(4, [4, 4], seed=3)
        loss = compute_loss(net, data)
        self.assertEqual(Objective(data)(net), loss)
        self.assertIsInstance(Objective(data)(net), int)
        self.assertAlmostEqual(
            compute_objective(net, data, lam=0.1, mu=0.5),
            loss + 0.1 * net.complexity() + 0.5 * brute_influence(net, 4))

    def test_polish_is_monotone_safe_and_locally_optimal(self):
        data, _ = make_dataset(4, majority, observed_fraction=0.5, seed=2)
        for seed in range(3):
            net = BooleanNetwork(4, [4, 4], seed=seed)
            obj = Objective(data, lam=0.05, mu=0.5)
            before = obj(net)
            polish(net, obj)
            after = obj(net)
            self.assertLessEqual(after, before)
            self.assertFalse(net.has_contradiction())
            for wid in list(net.all_weight_ids()):
                if net.is_safe_flip(wid):
                    net.flip_weight(wid)
                    self.assertGreaterEqual(obj(net), after - 1e-12, wid)
                    net.flip_weight(wid)

    def test_influence_training_never_worse_than_init(self):
        data, _ = make_dataset(5, majority, observed_fraction=0.5, seed=4)
        for seed in range(3):
            net = BooleanNetwork(5, [8, 4], seed=seed)
            obj = Objective(data, mu=0.5)
            start = obj(net)
            train(net, data, verbose=False, seed=seed, mu=0.5)
            self.assertLessEqual(obj(net), start)
            self.assertFalse(net.has_contradiction())


class TestEnsemble(unittest.TestCase):

    def test_vote_and_packed_vote_agree(self):
        rng = random.Random(9)
        members = [random_net(rng, 4, [4, 4]) for _ in range(5)]
        ens = Ensemble(members)
        pd = influence_rows(4)
        out = ens.forward_packed(pd)
        for r, x in enumerate(pd.rows):
            votes = sum(m(x) for m in members)
            self.assertEqual(ens(x), int(votes >= 3))
            self.assertEqual((out >> r) & 1, ens(x))

    def test_train_ensemble_is_odd_and_reproducible(self):
        data, _ = make_dataset(4, custom, observed_fraction=0.75, seed=0)
        a = train_ensemble(4, [4, 4], data, n_members=5, seed=3)
        b = train_ensemble(4, [4, 4], data, n_members=5, seed=3)
        self.assertEqual(len(a.members) % 2, 1)
        self.assertEqual([weights_fingerprint(m) for m in a.members],
                         [weights_fingerprint(m) for m in b.members])


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
