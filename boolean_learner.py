"""
Boolean Function Learner
========================
Learns an unknown Boolean function from its truth table,
using the same loop as a 1-layer neural network:

    forward pass  →  compute loss  →  Boolean derivative  →  update weights  →  repeat

Representation
--------------
The hypothesis is a Sum of Products (SOP):
    f(x) = w_0·T_0(x)  OR  w_1·T_1(x)  OR  ...  OR  w_k·T_k(x)

where each T_i is one candidate AND-term (e.g. x0∧x2, x1, x0∧x1∧x3, ...)
and each weight w_i ∈ {0, 1} controls whether that term is included.

This is exactly a 1-layer network with:
    - Binary weights  (vs. real-valued in a NN)
    - OR as the aggregation  (vs. weighted sum)
    - Identity threshold at 0  (vs. sigmoid/ReLU)

Training
--------
At each step:
    1. Forward pass   – evaluate f on all 2^n inputs.
    2. Find errors    – rows where f(x) ≠ target(x).
    3. Boolean deriv  – for each term T_i, compute
                            ∂f/∂w_i = f(w_i=0, x) XOR f(w_i=1, x)
                        at the first error point. This tells us
                        whether toggling T_i changes the output there.
    4. Score          – simulate flipping each w_i and count how many
                        total errors it fixes (greedy, like gradient magnitude).
    5. Update         – flip the best w_i  (the Boolean "gradient step").
"""

from itertools import product, combinations


# ── helpers ──────────────────────────────────────────────────────────────────

def all_inputs(n: int) -> list[tuple]:
    """All 2^n binary input vectors."""
    return list(product([0, 1], repeat=n))


def generate_terms(n: int) -> list[tuple]:
    """
    All non-empty subsets of {0,...,n-1} as AND-terms.
    Term (0, 2) means  x0 AND x2.
    Ordered by size so single literals come first.
    Total: 2^n - 1 terms.
    """
    terms = []
    for size in range(1, n + 1):
        for combo in combinations(range(n), size):
            terms.append(combo)
    return terms


def eval_term(term: tuple, x: tuple) -> int:
    """1 iff all variables in term are 1 in x."""
    return int(all(x[i] for i in term))


def eval_f(weights: list[int], terms: list[tuple], x: tuple) -> int:
    """Forward pass: OR over all active AND-terms."""
    for w, t in zip(weights, terms):
        if w and eval_term(t, x):
            return 1
    return 0


def compute_loss(weights, terms, inputs, targets) -> int:
    """Number of misclassified rows (our 'loss')."""
    return sum(eval_f(weights, terms, x) != targets[x] for x in inputs)


def term_label(term: tuple, n: int) -> str:
    """Human-readable label, e.g. (0,2) → 'x0∧x2'."""
    return "∧".join(f"x{i}" for i in term)


def current_fn_str(weights, terms, n) -> str:
    active = [term_label(t, n) for w, t in zip(weights, terms) if w]
    return "f = " + " OR ".join(active) if active else "f = 0"


# ── Boolean derivative ────────────────────────────────────────────────────────

def bool_derivative(weights: list[int], terms: list[tuple], x: tuple, term_idx: int) -> int:
    """
    ∂f/∂w_i at input x.

    = f(w_i=0, x)  XOR  f(w_i=1, x)

    1 → flipping w_i changes the output at x  (term 'matters' here)
    0 → flipping w_i does nothing at x
    """
    w0 = weights.copy(); w0[term_idx] = 0
    w1 = weights.copy(); w1[term_idx] = 1
    return eval_f(w0, terms, x) ^ eval_f(w1, terms, x)


def score_flip(weights, terms, inputs, targets, term_idx) -> int:
    """
    Net errors fixed by flipping w_i.
    Analogous to the gradient magnitude — how much does this update help?
    """
    new_w = weights.copy()
    new_w[term_idx] ^= 1
    old_loss = compute_loss(weights, terms, inputs, targets)
    new_loss = compute_loss(new_w, terms, inputs, targets)
    return old_loss - new_loss          # positive = improvement


# ── training loop ─────────────────────────────────────────────────────────────

def train(
    n: int,
    target_fn,
    max_steps: int = 200,
    verbose: bool = True,
) -> list[int]:
    """
    Learn target_fn : {0,1}^n → {0,1} from its truth table.

    Parameters
    ----------
    n          : number of input variables
    target_fn  : callable (tuple of 0/1) → 0 or 1
    max_steps  : stop after this many steps even if not converged
    verbose    : print training log

    Returns
    -------
    weights : learned weight vector over all AND-terms
    """
    inputs  = all_inputs(n)
    targets = {x: target_fn(x) for x in inputs}
    terms   = generate_terms(n)
    k       = len(terms)

    # Start with all weights = 0  (f = 0 everywhere, like zero-init)
    weights = [0] * k

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Boolean Function Learner  |  n={n} inputs, {k} candidate terms")
        print(f"  Truth table: {len(inputs)} rows")
        print(f"{'='*60}")

    for step in range(1, max_steps + 1):
        loss = compute_loss(weights, terms, inputs, targets)

        if verbose:
            print(f"\nStep {step:>3}  |  Loss: {loss}/{len(inputs)}  |  {current_fn_str(weights, terms, n)}")

        if loss == 0:
            if verbose:
                print("\n✓ Converged!")
            break

        # ── pick the first error row (like picking a training sample) ─────────
        error_x = next(x for x in inputs if eval_f(weights, terms, x) != targets[x])

        if verbose:
            got      = eval_f(weights, terms, error_x)
            expected = targets[error_x]
            xstr     = "(" + ", ".join(f"x{i}={v}" for i,v in enumerate(error_x)) + ")"
            print(f"         Error at {xstr}: got {got}, expected {expected}")

        # ── Boolean derivatives at the error point ────────────────────────────
        derivs = [bool_derivative(weights, terms, error_x, i) for i in range(k)]

        if verbose:
            active_derivs = [(term_label(terms[i], n), derivs[i])
                             for i in range(k) if derivs[i] == 1]
            labels = ", ".join(f"{l}→1" for l,_ in active_derivs) or "none"
            print(f"         ∂f/∂w  non-zero: [{labels}]")

        # ── score every possible flip (like computing gradient for each weight)
        scores = [score_flip(weights, terms, inputs, targets, i) for i in range(k)]

        best_i = max(range(k), key=lambda i: scores[i])
        best_score = scores[best_i]

        if best_score <= 0:
            if verbose:
                print("         No single flip improves loss — stopping early.")
            break

        # ── update (the Boolean gradient step) ───────────────────────────────
        weights[best_i] ^= 1
        action = "ADD " if weights[best_i] else "REMOVE"

        if verbose:
            print(f"         → {action} [{term_label(terms[best_i], n)}]  "
                  f"(fixes {best_score} error{'s' if best_score!=1 else ''})")

    else:
        if verbose:
            print(f"\n✗ Did not converge in {max_steps} steps.")

    return weights


# ── verification ──────────────────────────────────────────────────────────────

def verify(weights, terms, n, target_fn, verbose=True):
    inputs  = all_inputs(n)
    targets = {x: target_fn(x) for x in inputs}
    errors  = 0

    if verbose:
        header = "  " + "  ".join(f"x{i}" for i in range(n)) + "  target  learned"
        print(f"\n{'='*60}")
        print("  Final truth table")
        print(f"{'='*60}")
        print(header)
        print("  " + "-"*(len(header)-2))

    for x in inputs:
        got = eval_f(weights, terms, x)
        tgt = targets[x]
        ok  = got == tgt
        if not ok:
            errors += 1
        if verbose:
            bits = "  ".join(str(v) for v in x)
            mark = "✓" if ok else "✗ ← error"
            print(f"  {bits}     {tgt}        {got}    {mark}")

    if verbose:
        print(f"\n  Accuracy: {len(inputs)-errors}/{len(inputs)} rows correct")
        print(f"  Learned function: {current_fn_str(weights, terms, n)}")

    return errors == 0


# ── example functions to learn ────────────────────────────────────────────────

def majority(x: tuple) -> int:
    """Output 1 if more than half the inputs are 1."""
    return int(sum(x) > len(x) / 2)


def parity(x: tuple) -> int:
    """Output 1 if an odd number of inputs are 1 (XOR for n inputs)."""
    return sum(x) % 2


def at_least_two(x: tuple) -> int:
    """Output 1 if at least 2 inputs are 1."""
    return int(sum(x) >= 2)


def custom_fn(x: tuple) -> int:
    """
    Example: define your own function here.
    x is a tuple of 0/1 values, one per input variable.
    """
    x0, x1, x2, x3 = x
    return int((x0 and not x1) or (x2 and x3))


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    examples = [
        ("Majority (n=3)",    3, majority),
        ("Majority (n=5)",    5, majority),
        ("Parity/XOR (n=3)",  3, parity),
        ("At-least-two (n=4)",4, at_least_two),
        ("Custom (n=4)",      4, custom_fn),
    ]

    for name, n, fn in examples:
        print(f"\n\n{'#'*60}")
        print(f"  EXAMPLE: {name}")
        print(f"{'#'*60}")
        weights = train(n, fn, verbose=True)
        terms   = generate_terms(n)
        verify(weights, terms, n, fn, verbose=True)
        input("\n  Press Enter for next example...")
