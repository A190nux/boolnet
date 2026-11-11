"""
test.py — Eight investigations, each producing numbers to fold back into the README:

  1. Rerun XOR(n=2) and Parity(n=3) convergence stats with real, reproducible seeds
     (replaces old convergence claims with numbers from THIS run).
  2. Empirically measure how backward() actually scales with depth and width,
     to check the documented complexity claim against reality.
  3. Probe whether layer_widths=[4,4] can represent 4-input parity exactly, or
     whether it needs more restarts / a wider config.
  4. Measure whether Boolean regularization (lam > 0) improves test accuracy.
  5. Compare regularizers: weight count, total influence, ensembles.
  6. A real dataset: MONK's problems vs decision tree / nearest-neighbour baselines.
  7. Neutral moves: isolating why λ > 0 helps the search.
  8. Sparse initialization: are sparse networks easier to search?

Fast correctness tests live in tests/ (python3 -m unittest discover -s tests).

Run: python3 test.py
"""

import math
import random
import time

from bool_net import *
from baselines import DecisionTree, MajorityLabel, NearestNeighbour
from datasets import load_monks


# ─────────────────────────────────────────────────────────────────────────────
# 1. Convergence reruns: XOR(2), Parity(3)
# ─────────────────────────────────────────────────────────────────────────────

def convergence_stats(desc, n, fn, widths, n_seeds=30, max_steps=500,
                       n_restarts=10, base_seed=1000):
    """
    Runs train_with_restarts across n_seeds independent seeds and reports how
    many reach zero training loss.

    Uses observed_fraction=1.0 (the full truth table) deliberately: this is a
    question about whether the architecture + training algorithm can find an
    exact fit at all, not about generalization to held-out rows. This matches
    the convention already used for these functions in bool_net.py's __main__.
    """
    train_data, _ = make_dataset(n, fn, observed_fraction=1.0, seed=0)
    successes = 0
    losses = []
    t0 = time.perf_counter()
    for s in range(n_seeds):
        seed = base_seed + s
        net = train_with_restarts(
            n_inputs=n, layer_widths=widths, train_data=train_data,
            max_steps=max_steps, n_restarts=n_restarts, verbose=False, seed=seed,
        )
        loss = compute_loss(net, train_data)
        losses.append(loss)
        if loss == 0:
            successes += 1
    elapsed = time.perf_counter() - t0
    print(f"  {desc:32s} widths={str(widths):10s} "
          f"{successes}/{n_seeds} seeds -> zero loss  "
          f"({elapsed:.2f}s total)")
    if successes < n_seeds:
        print(f"      non-zero losses seen: {[l for l in losses if l > 0]}")
    return successes, n_seeds


def run_convergence_section():
    print("=" * 78)
    print("1. CONVERGENCE RERUNS (XOR n=2, Parity n=3)")
    print("=" * 78)
    convergence_stats("XOR (n=2)", 2, xor2, [4, 4])
    convergence_stats("Parity (n=3)", 3, parity, [4])
    convergence_stats("Parity (n=3)", 3, parity, [6, 4])
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Complexity: measure how backward() actually scales
# ─────────────────────────────────────────────────────────────────────────────

def time_backward(depth, width, n_inputs=4, n_samples=30, seed=1):
    """Average wall-clock time of a single net.backward(x) call."""
    widths = [width] * depth
    net = BooleanNetwork(n_inputs=n_inputs, layer_widths=widths, seed=seed)
    xs = all_inputs(n_inputs)
    rng = random.Random(seed)
    samples = [rng.choice(xs) for _ in range(n_samples)]
    for x in samples[:3]:          # warm-up, not timed
        net.backward(x)
    t0 = time.perf_counter()
    for x in samples:
        net.backward(x)
    t1 = time.perf_counter()
    return (t1 - t0) / n_samples


def loglog_slope(xs, ys):
    """Least-squares slope of log(y) vs log(x) -> empirical exponent."""
    lx = [math.log(v) for v in xs]
    ly = [math.log(v) for v in ys]
    n = len(xs)
    mx, my = sum(lx) / n, sum(ly) / n
    num = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    den = sum((a - mx) ** 2 for a in lx)
    return num / den


def run_complexity_section():
    print("=" * 78)
    print("2. COMPLEXITY: measured scaling of backward()")
    print("=" * 78)

    widths = [8, 16, 24, 32, 48, 64]
    depth = 3
    wid_times = [time_backward(depth, w) for w in widths]
    print(f"  Width sweep (depth fixed at {depth}):")
    for w, t in zip(widths, wid_times):
        print(f"    width={w:3d}  avg backward() = {t * 1e6:9.1f} us")
    exp_all = loglog_slope(widths, wid_times)
    exp_tail = loglog_slope(widths[2:], wid_times[2:])
    print(f"  measured width exponent (all points):    {exp_all:.2f}")
    print(f"  measured width exponent (largest half):  {exp_tail:.2f}  "
          f"(closer to the asymptotic value)")

    print()
    depths = [4, 8, 12, 16, 20, 24]
    width = 6
    dep_times = [time_backward(d, width) for d in depths]
    print(f"  Depth sweep (width fixed at {width}):")
    for d, t in zip(depths, dep_times):
        print(f"    depth={d:3d}  avg backward() = {t * 1e6:9.1f} us")
    exp_all = loglog_slope(depths, dep_times)
    exp_tail = loglog_slope(depths[2:], dep_times[2:])
    print(f"  measured depth exponent (all points):    {exp_all:.2f}")
    print(f"  measured depth exponent (largest half):  {exp_tail:.2f}  "
          f"(closer to the asymptotic value)")
    print()
    print("  Note: exponents are expected to CLIMB toward their asymptotic value")
    print("  as depth/width grow — sum_{k=1}^{D-1}(D-k) ~ D^2/2 - D/2, so the")
    print("  linear term still matters at small D and pulls the fitted slope down.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Is [4,4] genuinely too small for exact 4-input parity?
# ─────────────────────────────────────────────────────────────────────────────

def representability_probe(desc, n, fn, widths, n_trials=20, max_steps=500,
                            base_seed=5000):
    """
    Unlike train_with_restarts (which stops at the first success), this runs
    every trial to completion so we get an actual success RATE, not just a
    yes/no. Low-but-nonzero suggests "hard to find, but representable."
    Zero across many independent trials, especially next to configs that
    succeed easily, is evidence (not proof) of a genuine capacity limit.
    """
    train_data, _ = make_dataset(n, fn, observed_fraction=1.0, seed=0)
    rng = random.Random(base_seed)
    successes = 0
    t0 = time.perf_counter()
    for _ in range(n_trials):
        net_seed = rng.randint(0, 2**31 - 1)
        train_seed = rng.randint(0, 2**31 - 1)
        net = BooleanNetwork(n_inputs=n, layer_widths=widths, seed=net_seed)
        train(net, train_data, max_steps=max_steps, verbose=False, seed=train_seed)
        if compute_loss(net, train_data) == 0:
            successes += 1
    elapsed = time.perf_counter() - t0
    print(f"  {desc:28s} widths={str(widths):10s} "
          f"{successes:2d}/{n_trials} independent restarts -> zero loss  "
          f"({elapsed:.1f}s)")
    return successes, n_trials


def run_representability_section():
    print("=" * 78)
    print("3. Is layer_widths=[4,4] too small for exact 4-input parity?")
    print("=" * 78)
    print("  Step A: per-restart success rate (each trial run to completion,")
    print("  no early stop) -- tells us whether [4,4] can represent the")
    print("  function at all, and how that compares to giving it more width")
    print("  or more depth:")
    configs = [
        ("baseline (in question)", [4, 4]),
        ("wider layer 0",          [6, 4]),
        ("much wider, same depth", [8, 8]),
        ("extra depth, same width",[4, 4, 4]),
    ]
    for desc, widths in configs:
        representability_probe(desc, 4, parity, widths, n_trials=20)

    print()
    print("  Step B: if [4,4] is representationally sufficient, does simply")
    print("  raising train_with_restarts' n_restarts close the gap? Reusing")
    print("  the same convergence_stats() harness from Section 1:")
    convergence_stats("Parity (n=4) [4,4], n_restarts=10 (current default)",
                       4, parity, [4, 4], n_seeds=15, n_restarts=10, base_seed=7000)
    convergence_stats("Parity (n=4) [4,4], n_restarts=20",
                       4, parity, [4, 4], n_seeds=15, n_restarts=20, base_seed=8000)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Does Boolean regularization improve generalization?
# ─────────────────────────────────────────────────────────────────────────────

def run_regularization_section(n_seeds=8, lams=(0.0, 0.1, 0.5)):
    """
    Partial-data generalization with and without the complexity penalty
    (objective = errors + lam × active weights). Each seed is a different
    train/test split AND a different training seed; numbers are averaged.
    Parity is deliberately absent: every Hamming neighbour of a held-out
    parity row has the opposite label, so any simplicity bias predicts it
    wrong — it measures nothing about regularization.
    """
    print("=" * 78)
    print("4. REGULARIZATION: test accuracy vs lam (partial truth tables)")
    print("=" * 78)
    cases = [
        ("Majority n=5, 75%",   5, majority,     [8, 4], 0.75),
        ("Majority n=5, 50%",   5, majority,     [8, 4], 0.50),
        ("AtLeastTwo n=5, 50%", 5, at_least_two, [8, 4], 0.50),
        ("custom n=4, 50%",     4, custom,       [4, 4], 0.50),
        ("custom n=4, 75%",     4, custom,       [4, 4], 0.75),
    ]
    for desc, n, fn, widths, frac in cases:
        print(f"  {desc}  widths={widths}")
        for lam in lams:
            tr_acc = te_acc = size = 0.0
            t0 = time.perf_counter()
            for s in range(n_seeds):
                train_data, test_data = make_dataset(n, fn, observed_fraction=frac,
                                                     seed=100 + s)
                net = train_with_restarts(n, widths, train_data, n_restarts=5,
                                          verbose=False, seed=s, lam=lam)
                a, b = evaluate(net, train_data, test_data, verbose=False)
                tr_acc += a; te_acc += b; size += net.complexity()
            print(f"    lam={lam:<4}  train {100*tr_acc/n_seeds:5.1f}%   "
                  f"test {100*te_acc/n_seeds:5.1f}%   "
                  f"active weights {size/n_seeds:5.1f}   "
                  f"({time.perf_counter() - t0:.0f}s)")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Regularizer comparison: weights vs influence vs ensemble
# ─────────────────────────────────────────────────────────────────────────────

def at_least_three(x): return int(sum(x) >= 3)

REG_CASES = [
    ("Majority n=5, 75%",    5, majority,       [8, 4], 0.75),
    ("Majority n=5, 50%",    5, majority,       [8, 4], 0.50),
    ("AtLeastTwo n=5, 50%",  5, at_least_two,   [8, 4], 0.50),
    ("AtLeastThree n=6, 50%", 6, at_least_three, [8, 4], 0.50),
    ("custom n=4, 50%",      4, custom,         [4, 4], 0.50),
    ("custom n=4, 75%",      4, custom,         [4, 4], 0.75),
]


def _reg_job(job):
    """One (case, config, seed) run — top-level so multiprocessing can pickle it."""
    case_i, cfg, seed = job
    _, n, fn, widths, frac = REG_CASES[case_i]
    kind, lam, mu = cfg
    train_data, test_data = make_dataset(n, fn, observed_fraction=frac, seed=seed)
    if kind in BASELINES:
        net = BASELINES[kind](train_data)
        tr, te = evaluate(net, train_data, test_data, verbose=False)
        return case_i, cfg, tr, te, None
    if kind == "ensemble":
        net = train_ensemble(n, widths, train_data, n_members=9, seed=seed,
                             lam=lam, mu=mu)
        infl = None
    else:
        net = train_with_restarts(n, widths, train_data, n_restarts=5,
                                  verbose=False, seed=seed, lam=lam, mu=mu)
        infl = total_influence(net, influence_rows(n))
    tr, te = evaluate(net, train_data, test_data, verbose=False)
    return case_i, cfg, tr, te, infl


def _parallel_map(fn, jobs, processes=None, label="runs"):
    """Pool map that reports progress on stderr as runs finish (results are
    returned in job order, so output is still deterministic)."""
    import sys
    from multiprocessing import Pool
    out = [None] * len(jobs)
    t0 = time.perf_counter()
    with Pool(processes) as pool:
        for done, (i, r) in enumerate(pool.imap_unordered(_indexed, [(fn, i, j) for i, j in enumerate(jobs)]), 1):
            out[i] = r
            print(f"\r    [{done}/{len(jobs)} {label}, {time.perf_counter() - t0:5.0f}s]",
                  end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return out


def _indexed(arg):
    fn, i, job = arg
    return i, fn(job)


def regularizer_comparison(configs, seeds, processes=None):
    """Mean train/test accuracy (± standard error over seeds) per case and config."""
    jobs = [(c, cfg, s) for c in range(len(REG_CASES)) for cfg in configs for s in seeds]
    results = _parallel_map(_reg_job, jobs, processes)
    table = {}
    for c, cfg, tr, te, infl in results:
        table.setdefault((c, cfg), []).append((tr, te, infl))
    for c, (desc, n, fn, _, _) in enumerate(REG_CASES):
        true_infl = total_influence_of(fn, n)
        print(f"  {desc}   (target influence {true_infl:.2f})")
        for cfg in configs:
            rows = table[(c, cfg)]
            tes = [r[1] for r in rows]
            mean = sum(tes) / len(tes)
            se = (sum((t - mean) ** 2 for t in tes) / (len(tes) - 1)) ** 0.5 / len(tes) ** 0.5
            tr = sum(r[0] for r in rows) / len(rows)
            infls = [r[2] for r in rows if r[2] is not None]
            infl = f"{sum(infls)/len(infls):.2f}" if infls else "  - "
            print(f"    {cfg_name(cfg):22s} train {100*tr:5.1f}%   "
                  f"test {100*mean:5.1f}% ± {100*se:4.1f}   influence {infl}")
    return table


BASELINES = {"const": MajorityLabel, "1nn": NearestNeighbour}


def cfg_name(cfg):
    kind, lam, mu = cfg
    if kind == "const":
        return "baseline: majority label"
    if kind == "1nn":
        return "baseline: nearest nbr"
    parts = [p for p in (f"λ={lam}" if lam else "", f"μ={mu}" if mu else "") if p]
    label = " ".join(parts) or "none"
    return f"ensemble(9) {label}" if kind == "ensemble" else label


def total_influence_of(fn, n):
    """Influence of the true target function (reference point)."""
    rows = all_inputs(n)
    flips = sum(fn(x) != fn(x[:i] + (1 - x[i],) + x[i+1:]) for x in rows for i in range(n))
    return flips / len(rows)


def run_regularizer_section():
    print("=" * 78)
    print("5. REGULARIZERS: weight count vs total influence vs ensemble")
    print("=" * 78)
    configs = [("single", 0.0, 0.0), ("single", 0.1, 0.0),
               ("single", 0.0, 0.25), ("single", 0.0, 0.5), ("single", 0.0, 1.0),
               ("single", 0.0, 2.0), ("single", 0.1, 0.5),
               ("ensemble", 0.1, 0.0)]
    print("  Tuning seeds 100-119:")
    regularizer_comparison(configs, range(100, 120))
    print()
    # Configs chosen from the tuning run (mu values 0.25-1.0 were identical
    # there, so one stands in for all), plus two trivial baselines, on
    # FRESH seeds — these are the numbers to quote.
    confirm = [("const", 0, 0), ("1nn", 0, 0),
               ("single", 0.0, 0.0), ("single", 0.1, 0.0),
               ("single", 0.0, 0.5), ("single", 0.1, 0.5),
               ("ensemble", 0.1, 0.0)]
    print("  Confirmation seeds 200-219 (fresh splits and training seeds):")
    regularizer_comparison(confirm, range(200, 220))
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 6. A real dataset: the MONK's problems
# ─────────────────────────────────────────────────────────────────────────────

MONKS_METHODS = {
    "λ=0":                     {},
    "λ=0.1":                   {"lam": 0.1},
    "λ=1e-9":                  {"lam": 1e-9},
    "λ=0 + neutral removals":  {"neutral": "remove"},
    "λ=0 + any neutral move":  {"neutral": "any"},
    "λ=0.1 + any neutral move": {"lam": 0.1, "neutral": "any"},
    "λ=0, density 0.3":        {"density": 0.3},
    "λ=0, density 0.15":       {"density": 0.15},
    "λ=0, density 0.05":       {"density": 0.05},
    "λ=1e-9, density 0.3":     {"lam": 1e-9, "density": 0.3},
    "λ=1e-9, density 0.15":    {"lam": 1e-9, "density": 0.15},
    "λ=1e-9, density 0.05":    {"lam": 1e-9, "density": 0.05},
}


def _monks_job(job):
    """One (problem, widths, method, seed) BoolNet run on the official split."""
    problem, widths, method, seed = job
    train_data, test_data = load_monks(problem)
    t0 = time.perf_counter()
    net = train_with_restarts(17, widths, train_data, n_restarts=5,
                              verbose=False, seed=seed, **MONKS_METHODS[method])
    tr, te = evaluate(net, train_data, test_data, verbose=False)
    return problem, tuple(widths), method, tr, te, net.complexity(), time.perf_counter() - t0


def monks_comparison(widths_list, methods, seeds, processes=None):
    jobs = [(p, w, m, s) for p in (1, 2, 3) for w in widths_list
            for m in methods for s in seeds]
    results = _parallel_map(_monks_job, jobs, processes)
    table = {}
    for p, w, m, tr, te, cx, dt in results:
        table.setdefault((p, w, m), []).append((tr, te, cx, dt))
    for p in (1, 2, 3):
        train_data, test_data = load_monks(p)
        print(f"  MONK-{p}  ({len(train_data)} train / {len(test_data)} test"
              f"{', 5% train label noise' if p == 3 else ''})")
        for name, model in [("majority label", MajorityLabel(train_data)),
                            ("nearest neighbour", NearestNeighbour(train_data)),
                            ("decision tree (ID3)", DecisionTree(train_data))]:
            tr, te = evaluate(model, train_data, test_data, verbose=False)
            print(f"    {name:34s} train {100*tr:5.1f}%   test {100*te:5.1f}%")
        for w in widths_list:
            for m in methods:
                rows = table[(p, tuple(w), m)]
                tes = [r[1] for r in rows]
                mean = sum(tes) / len(tes)
                se = ((sum((t - mean) ** 2 for t in tes) / (len(tes) - 1)) ** 0.5
                      / len(tes) ** 0.5) if len(tes) > 1 else 0.0
                tr = sum(r[0] for r in rows) / len(rows)
                cx = sum(r[2] for r in rows) / len(rows)
                dt = sum(r[3] for r in rows) / len(rows)
                best = max(tes)
                print(f"    BoolNet {str(w):8s} {m:26s} train {100*tr:5.1f}%   "
                      f"test {100*mean:5.1f}% ± {100*se:4.1f} (best {100*best:5.1f})   "
                      f"|W| {cx:5.1f}   {dt:5.1f}s/run")
    return table


def run_monks_section():
    print("=" * 78)
    print("6. MONK'S PROBLEMS (official train/test splits, 17 one-hot inputs)")
    print("=" * 78)
    monks_comparison([[8, 4], [12, 6]], ["λ=0", "λ=0.1"], range(10))
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 7. Why does λ help the search? Neutral moves, isolated
# ─────────────────────────────────────────────────────────────────────────────

def run_neutral_section():
    """
    Step 16 found that λ=1e-9 trains exactly as well as λ=0.1 on MONK's, and
    concluded by elimination that the gain comes from λ>0 turning
    error-neutral, weight-removing flips into accepted moves. This section
    tests that directly: λ=0 plus ONLY those moves ("neutral removals")
    should match λ=1e-9. "any neutral move" (with a tabu list) asks whether
    a freer plateau walk helps more.
    """
    print("=" * 78)
    print("7. NEUTRAL MOVES: isolating why λ > 0 helps the search (MONK's, [8,4])")
    print("=" * 78)
    monks_comparison([[8, 4]], ["λ=0", "λ=1e-9", "λ=0 + neutral removals",
                                "λ=0 + any neutral move", "λ=0.1",
                                "λ=0.1 + any neutral move"], range(10))
    print()


# ─────────────────────────────────────────────────────────────────────────────
# 8. Sparse initialization: are sparse networks easier to search?
# ─────────────────────────────────────────────────────────────────────────────

def run_sparse_init_section():
    """
    Step 17's hypothesis: λ > 0 trains better mainly because it keeps the
    network sparse throughout. If so, λ=0 started from a sparse network
    should close much of the gap to λ=1e-9. The default init uses each
    input signal with probability 2/3.
    """
    print("=" * 78)
    print("8. SPARSE INIT: initial connection density (MONK's, [8,4])")
    print("=" * 78)
    for density in (None, 0.3, 0.15, 0.05):
        sizes = [BooleanNetwork(17, [8, 4], seed=s, density=density).complexity()
                 for s in range(50)]
        print(f"  density {str(density or '2/3 (default)'):14s} → "
              f"{sum(sizes)/len(sizes):5.1f} active weights at init")
    monks_comparison([[8, 4]], ["λ=0", "λ=0, density 0.3", "λ=0, density 0.15",
                                "λ=0, density 0.05", "λ=1e-9",
                                "λ=1e-9, density 0.3", "λ=1e-9, density 0.15",
                                "λ=1e-9, density 0.05"], range(10))
    print()


if __name__ == "__main__":
    run_convergence_section()
    run_complexity_section()
    run_representability_section()
    run_regularization_section()
    run_regularizer_section()
    run_monks_section()
    run_neutral_section()
    run_sparse_init_section()
