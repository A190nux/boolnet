"""
test.py — Three investigations, each producing numbers to fold back into the README:

  1. Rerun XOR(n=2) and Parity(n=3) convergence stats with real, reproducible seeds
     (replaces old convergence claims with numbers from THIS run).
  2. Empirically measure how backward() actually scales with depth and width,
     to check the documented complexity claim against reality.
  3. Probe whether layer_widths=[4,4] can represent 4-input parity exactly, or
     whether it needs more restarts / a wider config.

Run: python3 test.py
"""

import math
import random
import time

from bool_net import *


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


if __name__ == "__main__":
    run_convergence_section()
    run_complexity_section()
    run_representability_section()