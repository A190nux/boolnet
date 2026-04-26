"""
Boolean Function Learner  —  partial data edition
==================================================
Learns a Boolean function from a *subset* of its truth table,
exactly like a neural network trained on a limited dataset.

Key difference from full-table version
---------------------------------------
  Before : loss computed over all 2^n rows  (cheating — perfect knowledge)
  Now    : loss computed only over observed rows  (realistic)

Unseen inputs are unknown. The learned SOP will generalize to them
based purely on what it inferred from the training data — same as a NN.

Pipeline (identical to a 1-layer neural network)
-------------------------------------------------
  1. Forward pass   — evaluate f on training rows only
  2. Loss           — errors on training rows only
  3. Bool deriv     — ∂f/∂w_i at the first wrong training row
  4. Score          — which flip fixes the most training errors
  5. Update         — flip the best weight
  6. Repeat until training loss = 0 or no improvement possible
  7. Evaluate       — check accuracy on held-out test rows
"""

from itertools import combinations, product
import random


# ── core helpers ──────────────────────────────────────────────────────────────

def all_inputs(n):
    return list(product([0, 1], repeat=n))

def generate_terms(n):
    terms = []
    for size in range(1, n + 1):
        for combo in combinations(range(n), size):
            terms.append(combo)
    return terms

def eval_term(term, x):
    return int(all(x[i] for i in term))

def eval_f(weights, terms, x):
    for w, t in zip(weights, terms):
        if w and eval_term(t, x):
            return 1
    return 0

def term_label(term):
    return "∧".join(f"x{i}" for i in term)

def current_fn_str(weights, terms):
    active = [term_label(t) for w, t in zip(weights, terms) if w]
    return "f = " + " OR ".join(active) if active else "f = 0"


# ── data splitting ────────────────────────────────────────────────────────────

def make_dataset(n, target_fn, observed_fraction=0.5, seed=42):
    """
    Split the full truth table into train and test sets.

    Parameters
    ----------
    n                  : number of input variables
    target_fn          : the true function  (tuple -> 0 or 1)
    observed_fraction  : fraction of 2^n rows used for training
    seed               : random seed for reproducibility

    Returns
    -------
    train : dict  {input_tuple: label}   -- what the learner sees
    test  : dict  {input_tuple: label}   -- held-out rows for evaluation
    """
    rng = random.Random(seed)
    all_rows = all_inputs(n)
    rng.shuffle(all_rows)

    n_train = max(1, int(len(all_rows) * observed_fraction))
    train_rows = all_rows[:n_train]
    test_rows  = all_rows[n_train:]

    train = {x: target_fn(x) for x in train_rows}
    test  = {x: target_fn(x) for x in test_rows}
    return train, test


# ── Boolean derivative & scoring ──────────────────────────────────────────────

def bool_derivative(weights, terms, x, term_idx):
    """df/dw_i at input x  =  f(w_i=0, x) XOR f(w_i=1, x)"""
    w0 = weights.copy(); w0[term_idx] = 0
    w1 = weights.copy(); w1[term_idx] = 1
    return eval_f(w0, terms, x) ^ eval_f(w1, terms, x)

def compute_loss(weights, terms, data):
    """Errors over the provided data rows only."""
    return sum(eval_f(weights, terms, x) != y for x, y in data.items())

def score_flip(weights, terms, data, term_idx):
    """Net training errors fixed by flipping w_i."""
    new_w = weights.copy(); new_w[term_idx] ^= 1
    return compute_loss(weights, terms, data) - compute_loss(new_w, terms, data)


# ── training loop ─────────────────────────────────────────────────────────────

def train(n, train_data, max_steps=300, verbose=True):
    """
    Learn from train_data only -- a subset of the truth table.
    """
    terms   = generate_terms(n)
    k       = len(terms)
    weights = [0] * k
    n_train = len(train_data)
    n_total = 2 ** n

    if verbose:
        print(f"\n{'='*62}")
        print(f"  Boolean Learner  |  n={n}  |  "
              f"training on {n_train}/{n_total} rows ({100*n_train/n_total:.0f}%)")
        print(f"  Candidate terms : {k}")
        print(f"{'='*62}")

    for step in range(1, max_steps + 1):
        loss = compute_loss(weights, terms, train_data)

        if verbose:
            print(f"\nStep {step:>3}  |  Train loss: {loss}/{n_train}"
                  f"  |  {current_fn_str(weights, terms)}")

        if loss == 0:
            if verbose:
                print("  ✓ Zero training loss.")
            break

        # first wrong training row
        error_x = next(x for x, y in train_data.items()
                       if eval_f(weights, terms, x) != y)

        derivs = [bool_derivative(weights, terms, error_x, i) for i in range(k)]
        scores = [score_flip(weights, terms, train_data, i) for i in range(k)]

        best_i     = max(range(k), key=lambda i: scores[i])
        best_score = scores[best_i]

        if best_score <= 0:
            if verbose:
                print("  ✗ No single flip improves training loss — stopping.")
            break

        weights[best_i] ^= 1
        action = "ADD   " if weights[best_i] else "REMOVE"

        if verbose:
            active_d = [term_label(terms[i]) for i in range(k) if derivs[i]]
            xstr = "(" + " ".join(f"x{i}={v}" for i, v in enumerate(error_x)) + ")"
            print(f"  Error  : {xstr}  expected={train_data[error_x]}")
            print(f"  df/dw  : [{', '.join(active_d) or 'none'}]")
            print(f"  Update : {action} [{term_label(terms[best_i])}]"
                  f"  (fixes {best_score} training error{'s' if best_score != 1 else ''})")
    else:
        if verbose:
            print(f"\n  ✗ Did not converge within {max_steps} steps.")

    return weights


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(weights, terms, train_data, test_data, verbose=True):
    """
    Report accuracy on both train (seen) and test (unseen) rows.
    This is the generalization check.
    """
    def accuracy(data):
        if not data:
            return 0, 0
        correct = sum(eval_f(weights, terms, x) == y for x, y in data.items())
        return correct, len(data)

    tr_ok, tr_n = accuracy(train_data)
    te_ok, te_n = accuracy(test_data)

    if verbose:
        print(f"\n{'='*62}")
        print(f"  Results")
        print(f"{'='*62}")
        print(f"  Learned  : {current_fn_str(weights, terms)}")
        print(f"\n  Train accuracy :  {tr_ok}/{tr_n}  ({100*tr_ok/tr_n:.0f}%)  <- rows it trained on")
        if te_n > 0:
            print(f"  Test  accuracy :  {te_ok}/{te_n}  ({100*te_ok/te_n:.0f}%)  <- rows it never saw")
        else:
            print(f"  Test  accuracy :  N/A  (trained on full dataset)")

        if te_n > 0 and te_ok < te_n:
            missed = [(x, eval_f(weights, terms, x), y)
                      for x, y in test_data.items()
                      if eval_f(weights, terms, x) != y]
            print(f"\n  Generalization misses ({len(missed)}):")
            for x, got, expected in missed:
                xstr = "(" + " ".join(f"x{i}={v}" for i, v in enumerate(x)) + ")"
                print(f"    {xstr}  predicted={got}  true={expected}")

    tr_acc = tr_ok / tr_n if tr_n else 0
    te_acc = te_ok / te_n if te_n else None
    return tr_acc, te_acc


# ── example functions ─────────────────────────────────────────────────────────

def majority(x):      return int(sum(x) > len(x) / 2)
def parity(x):        return sum(x) % 2
def at_least_two(x):  return int(sum(x) >= 2)


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    runs = [
        # (description,              n,  target_fn,    frac)
        ("Majority n=3, 100% data",  3,  majority,     1.00),  # baseline
        ("Majority n=3,  75% data",  3,  majority,     0.75),
        ("Majority n=3,  50% data",  3,  majority,     0.50),
        ("Majority n=3,  25% data",  3,  majority,     0.25),
        ("At-least-two n=4, 50%",    4,  at_least_two, 0.50),
        ("At-least-two n=4, 25%",    4,  at_least_two, 0.25),
    ]

    for desc, n, fn, frac in runs:
        print(f"\n\n{'#'*62}")
        print(f"  {desc}")
        print(f"{'#'*62}")
        train_data, test_data = make_dataset(n, fn, observed_fraction=frac)
        weights = train(n, train_data, verbose=True)
        terms   = generate_terms(n)
        evaluate(weights, terms, train_data, test_data, verbose=True)
        input("\n  Press Enter for next run...")