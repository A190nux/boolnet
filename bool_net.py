"""
Boolean Neural Network — Full Multi-Layer Implementation
=========================================================

This module implements a neural network where every component is Boolean:

    Real NN concept          →    Boolean equivalent
    ─────────────────────────────────────────────────────────────
    Real weight  w ∈ ℝ       →    Binary weight  w ∈ {0, 1}
    Weighted sum of inputs   →    AND of selected inputs
    ReLU / sigmoid           →    Already binary — no activation needed
    Dense layer (n → L)      →    AND-layer: W matrix of shape (n × L)
    Negative weights         →    Complement inputs  [x, ¬x]
    Gradient  ∂L/∂w          →    Boolean derivative  ∂f/∂w ∈ {0, 1}
    SGD weight update        →    Flip the best weight
    Layer stacking           →    Same — output of one layer feeds the next

Key ideas
─────────
1.  WEIGHT MATRIX instead of term enumeration.
    Each layer has a matrix W of shape (input_width × L).
    Column j = the AND-mask for neuron j.
    W[i][j] = 1 means "neuron j requires input i to be 1 to fire."
    This keeps parameters at O(n × L) instead of O(2^n).

2.  COMPLEMENT AUGMENTATION replaces negative weights.
    Before feeding data into any layer we append the bitwise NOT:
        x = (1, 0, 1)  →  augment(x) = (1, 0, 1, 0, 1, 0)
    Now a neuron can enforce "x1 must be 0" by connecting to ¬x1
    instead of to x1. This gives full expressive power.

3.  BOOLEAN DERIVATIVE drives learning.
    For any weight W[i][j] and any input x:
        ∂f/∂W[i][j]  =  f(W[i][j]=0, x)  XOR  f(W[i][j]=1, x)
    Result is 1 if flipping that weight changes the output, else 0.
    No chain rule — just two forward passes. Then we score the flip
    globally (how many training errors does it fix?) and take the best.

4.  MULTI-LAYER stacking lets us represent non-linear functions.
    XOR and parity are impossible for a single AND-layer (same reason
    a single-layer perceptron cannot learn XOR). Two layers fix this.
"""

import random
from itertools import product as iter_product


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def augment(x: tuple) -> tuple:
    """
    Append the bitwise complement of every element.

        (1, 0, 1)  →  (1, 0, 1,  0, 1, 0)
         originals        complements

    Why: AND-neurons can only enforce "this bit must be 1."
    By including ¬xᵢ as an explicit input, a neuron can enforce
    "xᵢ must be 0" by connecting to the complement instead.
    This is equivalent to allowing negative weights in a real NN.
    """
    return x + tuple(1 - v for v in x)


def all_inputs(n: int) -> list:
    """All 2^n binary input tuples for n variables."""
    return list(iter_product([0, 1], repeat=n))


# ─────────────────────────────────────────────────────────────────────────────
# Single layer
# ─────────────────────────────────────────────────────────────────────────────

class BooleanLayer:
    """
    A layer of L independent AND-neurons.

    Weight matrix W has shape  (input_width × output_width).

        W[i][j] = 1  →  neuron j requires input i to be 1
        W[i][j] = 0  →  neuron j ignores input i

    Forward pass for neuron j:
        h[j] = AND of { x[i]  for all i where W[i][j] == 1 }

    If a column is all-zero, the AND is vacuously true (always fires).
    If a column is all-one,  the AND fires only when every input is 1.
    Both extremes are useless, so initialization avoids them.

    Neural network analogy
    ──────────────────────
    This is a dense layer where:
      - weights are binary (0 or 1) not real-valued
      - the aggregation is AND not weighted-sum
      - the activation is identity (output is already 0 or 1)
    """

    def __init__(self, input_width: int, output_width: int, rng: random.Random):
        self.input_width  = input_width
        self.output_width = output_width
        # input_width is always even (augmented signals)
        m = input_width // 2

        self.W = [[0] * output_width for _ in range(input_width)]

        for j in range(output_width):
            # Build a single column, pair by pair
            col = [0] * input_width
            while True:
                # For each original signal i, randomly choose one of the
                # three allowed (non-contradictory) patterns.
                for i in range(m):
                    choice = rng.randint(0, 2)   # 0, 1, or 2
                    if choice == 0:
                        # (0,0) – no requirement on x_i
                        col[i]     = 0
                        col[i + m] = 0
                    elif choice == 1:
                        # (1,0) – require x_i = 1
                        col[i]     = 1
                        col[i + m] = 0
                    else:
                        # (0,1) – require x_i = 0 (i.e., require ¬x_i = 1)
                        col[i]     = 0
                        col[i + m] = 1

                # Reject the all‑zero column (neuron would always fire)
                if any(col[i] == 1 for i in range(input_width)):
                    break
                # Else (very rare): retry. We can also just flip one pair:
                # but the while loop is fine because probability of all-zero
                # is tiny: (1/3)^m, so it will rarely loop more than once.

            # Copy the valid column into W
            for i in range(input_width):
                self.W[i][j] = col[i]

    def forward(self, x: tuple) -> tuple:
        """
        Evaluate all AND-neurons on input x.

        x       : tuple of bits, length == input_width
        returns : tuple of bits, length == output_width

        For each neuron j:
            Collect all i where W[i][j] = 1 (the "required" inputs).
            Fire (output 1) iff every required input is 1.
        """
        h = []
        for j in range(self.output_width):
            required = [i for i in range(self.input_width) if self.W[i][j] == 1]
            h.append(int(all(x[i] for i in required)))
        return tuple(h)

    def flip(self, i: int, j: int):
        """Flip weight W[i][j] between 0 and 1."""
        self.W[i][j] ^= 1

    def mask_str(self, j: int, n_inputs: int) -> str:
        """
        Human-readable description of what neuron j computes.
        We use the augmented input layout: first n_inputs are originals,
        next n_inputs are complements.
        """
        parts = []
        for i in range(self.input_width):
            if self.W[i][j] == 1:
                if i < n_inputs:
                    parts.append(f"x{i}")
                else:
                    parts.append(f"¬x{i - n_inputs}")
        return " ∧ ".join(parts) if parts else "1 (always)"


# ─────────────────────────────────────────────────────────────────────────────
# Full network
# ─────────────────────────────────────────────────────────────────────────────

class BooleanNetwork:
    """
    Multi-layer Boolean neural network.

    Architecture
    ────────────
    Input  x  (n_inputs bits)
      │
      ▼  augment → [x, ¬x]  (2·n_inputs bits)
      │
      ▼  Layer 0 : W0  (2·n_inputs  ×  L0)   → h0  (L0 bits)
      │
      ▼  augment → [h0, ¬h0]  (2·L0 bits)
      │
      ▼  Layer 1 : W1  (2·L0  ×  L1)         → h1  (L1 bits)
      │
      ▼  augment → [h1, ¬h1]  ...
      │
      ▼  Layer k : Wk  (2·L_{k-1}  ×  Lk)   → hk  (Lk bits)
      │
      ▼  Output weight vector  out_w  (Lk bits, binary)
      │
      ▼  OR of { hk[j]  where  out_w[j] = 1 }
      │
    output  (1 bit)

    The output weight vector is a learnable final layer of width 1:
    it selects which last-layer neurons vote on the final answer.

    Parameters
    ──────────
    n_inputs     : number of raw Boolean input variables
    layer_widths : list of ints — width of each hidden layer
                   e.g. [4, 4] = two layers of 4 neurons each
    seed         : random seed for reproducible initialization
    """

    def __init__(self, n_inputs: int, layer_widths: list, seed: int = 42):
        self.n_inputs     = n_inputs
        self.layer_widths = layer_widths
        rng               = random.Random(seed)

        # ── Build layers ──────────────────────────────────────────────────
        self.layers = []
        prev_width  = 2 * n_inputs        # first input is augmented

        for L in layer_widths:
            self.layers.append(BooleanLayer(prev_width, L, rng))
            prev_width = 2 * L            # output is augmented before next layer

        # ── Output weights ────────────────────────────────────────────────
        # One binary weight per neuron in the last layer.
        # out_w[j] = 1 means neuron j feeds into the final OR.
        # Initialized to all-1: every neuron votes. Training may turn some off.
        self.out_w = [1] * layer_widths[-1]

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, x: tuple) -> int:
        """
        Full forward pass for a single input x.

        1. Augment x with its complement.
        2. Pass through each layer (AND-neurons).
        3. Augment between layers (gives next layer ability to negate).
        4. Apply output weights and OR.

        Returns a single bit: 0 or 1.
        """
        h = augment(x)                    # step 1: [x, ¬x]

        for k, layer in enumerate(self.layers):
            h = layer.forward(h)          # step 2: AND-neurons
            if k < len(self.layers) - 1:
                h = augment(h)            # step 3: augment between layers

        # Step 4: output OR with learned gate
        return int(any(self.out_w[j] and h[j] for j in range(len(h))))

    def __call__(self, x: tuple) -> int:
        return self.forward(x)

    # ── Boolean derivatives ───────────────────────────────────────────────────

    def derivative_layer(self, x: tuple, layer_idx: int, i: int, j: int) -> int:
        """
        Boolean derivative of the network output w.r.t. W[layer_idx][i][j] at x.

            ∂f/∂W[k][i][j]  =  f(W[k][i][j]=0, x)  XOR  f(W[k][i][j]=1, x)

        = 1  if flipping this weight changes the output at x
        = 0  if this weight is irrelevant at x

        Implementation: flip the weight, evaluate twice, flip back.
        This is two forward passes — no chain rule, no backprop algebra.

        Neural network analogy: ∂Loss/∂w tells you the gradient magnitude
        and sign for weight w. Here, the derivative tells you only whether
        flipping matters (1) or not (0). Direction is implicit: we always
        want the flip that reduces loss, so we score both outcomes globally.
        """
        layer = self.layers[layer_idx]
        orig  = layer.W[i][j]

        layer.W[i][j] = 0;  f0 = self.forward(x)
        layer.W[i][j] = 1;  f1 = self.forward(x)
        layer.W[i][j] = orig                      # always restore

        return f0 ^ f1

    def derivative_out(self, x: tuple, j: int) -> int:
        """
        Boolean derivative w.r.t. out_w[j] at x.
        Same idea: does toggling this output gate change the answer?
        """
        orig = self.out_w[j]

        self.out_w[j] = 0;  f0 = self.forward(x)
        self.out_w[j] = 1;  f1 = self.forward(x)
        self.out_w[j] = orig

        return f0 ^ f1

    # ── Weight enumeration (used by training loop) ────────────────────────────

    def all_weight_ids(self):
        """
        Yield an identifier for every learnable weight in the network.

        ('layer', k, i, j)  →  W[k][i][j]  in layer k
        ('out',   j)         →  out_w[j]

        Iterating this gives us all candidates for the update step,
        analogous to iterating over all parameters in a real optimizer.
        """
        for k, layer in enumerate(self.layers):
            for i in range(layer.input_width):
                for j in range(layer.output_width):
                    yield ('layer', k, i, j)
        for j in range(len(self.out_w)):
            yield ('out', j)

    def get_weight(self, wid) -> int:
        if wid[0] == 'layer':
            _, k, i, j = wid
            return self.layers[k].W[i][j]
        else:
            _, j = wid
            return self.out_w[j]

    def flip_weight(self, wid):
        """Apply a single weight flip (the Boolean 'gradient step')."""
        if wid[0] == 'layer':
            _, k, i, j = wid
            self.layers[k].flip(i, j)
        else:
            _, j = wid
            self.out_w[j] ^= 1

    def is_safe_flip(self, wid) -> bool:
        """
        Return True if flipping this weight would NOT create a contradiction
        (i.e., a neuron requiring both a signal and its complement).
        """
        if wid[0] == 'out':
            return True   # output weights are independent

        _, k, i, j = wid
        layer = self.layers[k]
        current = layer.W[i][j]

        # Flipping 1 → 0 is always safe (removing a requirement never
        # introduces a contradiction).
        if current == 1:
            return True

        # Flipping 0 → 1: must ensure the complement in the same column is 0.
        m = layer.input_width // 2          # number of original signals
        i_comp = i + m if i < m else i - m
        return layer.W[i_comp][j] == 0

    def compute_derivative(self, x: tuple, wid) -> int:
        """Dispatch derivative computation based on weight type."""
        if wid[0] == 'layer':
            _, k, i, j = wid
            return self.derivative_layer(x, k, i, j)
        else:
            _, j = wid
            return self.derivative_out(x, j)

    # ── Description ───────────────────────────────────────────────────────────

    def describe(self):
        n = self.n_inputs
        total_w = sum(l.input_width * l.output_width for l in self.layers) + len(self.out_w)
        print(f"  Inputs   : {n} variables → augmented to {2*n} bits")
        for k, layer in enumerate(self.layers):
            aug_note = f" → augmented to {2*layer.output_width}" if k < len(self.layers)-1 else ""
            print(f"  Layer {k}  : {layer.input_width} → {layer.output_width} AND-neurons{aug_note}")
        print(f"  Output   : OR over {sum(self.out_w)}/{len(self.out_w)} last-layer neurons")
        print(f"  Total W  : {total_w} binary weights")

    def describe_learned(self):
        """Show what each neuron currently computes, in human-readable form."""
        n = self.n_inputs
        print("  Learned function:")
        for k, layer in enumerate(self.layers):
            # Track input width for the previous-layer decoding
            prev_n = n if k == 0 else self.layer_widths[k-1]
            print(f"  Layer {k}:")
            for j in range(layer.output_width):
                label = layer.mask_str(j, prev_n)
                active = "●" if k == len(self.layers)-1 and self.out_w[j] else " "
                print(f"    {active} neuron {j}: {label}")
        active_out = [j for j in range(len(self.out_w)) if self.out_w[j]]
        print(f"  Output = OR of neurons {active_out}")


# ─────────────────────────────────────────────────────────────────────────────
# Loss and scoring
# ─────────────────────────────────────────────────────────────────────────────

def compute_loss(net: BooleanNetwork, data: dict) -> int:
    """Count misclassified rows in data. Only training rows are used."""
    return sum(net(x) != y for x, y in data.items())


def score_flip(net: BooleanNetwork, data: dict, wid) -> int:
    """
    How many training errors does flipping weight wid fix?

    Positive = improvement.
    Zero     = neutral (doesn't help or hurt overall).
    Negative = makes things worse.

    This is the Boolean analog of the gradient magnitude:
    in a real NN, |∂L/∂w| tells you how much that weight matters.
    Here, score tells you how many training examples benefit from the flip.

    We simulate the flip, measure new loss, then undo — same as
    computing a directional derivative along one axis.
    """
    old_loss = compute_loss(net, data)
    net.flip_weight(wid)
    new_loss = compute_loss(net, data)
    net.flip_weight(wid)                  # always restore
    return old_loss - new_loss


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(
    net:        BooleanNetwork,
    train_data: dict,
    max_steps:  int  = 500,
    verbose:    bool = True,
    pair_strategy: str = 'aggressive',   # 'aggressive' or 'hybrid'
) -> BooleanNetwork:
    """
    Train the network on train_data using Boolean derivative guided search.

    At each step:
      1. Forward pass over all training rows → compute loss.
      2. Collect all misclassified rows and try each one in turn.
      3. For each such error row, compute Boolean derivatives; stop at the
         first row that yields any non-zero derivative.
      4. Among those candidates, score each flip globally.
         If a beneficial single flip exists, apply it and continue.
      5. If no single flip helps, try all pairs of weights (joint derivative)
         on all error rows, score the beneficial pairs globally, and apply the best.
      6. If still no improvement, flip a small random subset of weights
         (perturbation) to escape the local minimum, then continue.
      7. Repeat until loss = 0 or stuck counter exceeds MAX_STUCK.

    Parameters
    ──────────
    net        : the BooleanNetwork to train (modified in place)
    train_data : dict { input_tuple: label }  — observed rows only
    max_steps  : hard step limit
    verbose    : print step-by-step log
    """
    MAX_STUCK = 5
    n_train = len(train_data)
    n_total = 2 ** net.n_inputs
    stuck_counter = 0

    if verbose:
        print(f"\n{'='*64}")
        print(f"  Training  |  n={net.n_inputs}  |  "
              f"{n_train}/{n_total} rows ({100*n_train/n_total:.0f}%)")
        net.describe()
        print(f"{'='*64}")

    for step in range(1, max_steps + 1):
        loss = compute_loss(net, train_data)

        if verbose:
            print(f"\nStep {step:>4}  |  Loss {loss}/{n_train}")

        if loss == 0:
            if verbose:
                print("  ✓ Zero training loss — converged.")
            break

        # ── Collect all misclassified rows ─────────────────────────────
        error_rows = [x for x, y in train_data.items() if net(x) != y]

        # ── Single‑flip search ─────────────────────────────────────────
        single_candidates = []
        chosen_error_x = None
        chosen_expected = None
        chosen_got = None

        for error_x in error_rows:
            expected = train_data[error_x]
            got      = net(error_x)
            last_error_x  = error_x
            last_expected = expected
            last_got      = got

            for wid in net.all_weight_ids():
                if net.compute_derivative(error_x, wid) == 1:
                    single_candidates.append(wid)

            if single_candidates:
                chosen_error_x = error_x
                chosen_expected = expected
                chosen_got = got
                break

        single_success = False
        if single_candidates:
            if verbose:
                xstr = "(" + " ".join(f"x{i}={v}" for i, v in enumerate(chosen_error_x)) + ")"
                print(f"         Error at {xstr}: got={chosen_got}, expected={chosen_expected}")
                n_layer = sum(1 for w in single_candidates if w[0]=='layer')
                n_out   = sum(1 for w in single_candidates if w[0]=='out')
                print(f"         Non-zero derivatives: {n_layer} layer, {n_out} output")

            scores = {wid: score_flip(net, train_data, wid) for wid in single_candidates}
            best_wid = max(scores, key=lambda w: scores[w])
            best_score = scores[best_wid]

            if best_score > 0:
                net.flip_weight(best_wid)
                if verbose:
                    _log_flip(net, best_wid, best_score)
                stuck_counter = 0
                single_success = True

        if single_success:
            continue

        # ── Pair‑flip search (strategy‑selected) ─────────────────────────
        if verbose:
            if single_candidates:
                print("         No single flip improves loss — trying pairs...")
            else:
                print("         No non‑zero derivatives — trying pairs...")

        all_wids = list(net.all_weight_ids())
        pair_success = False

        if pair_strategy == 'aggressive':
            # ── AGGRESSIVE: fast, active‑only, single‑row ──────────────
            # Collect active weights (derivative=1 on ANY error row) with early exit
            active_wids = set()
            for wid in all_wids:
                for err_x in error_rows:
                    if net.compute_derivative(err_x, wid) == 1:
                        active_wids.add(wid)
                        break   # move to next weight as soon as it's active

            if active_wids:
                search_wids = list(active_wids)
                if verbose:
                    print(f"         Active weights: {len(active_wids)} / {len(all_wids)}")

                # Test pairs among active_wids on the first error row only
                first_err = error_rows[0]
                orig_out = net(first_err)
                pair_candidates = []
                for a in range(len(search_wids)):
                    wa = search_wids[a]
                    for b in range(a + 1, len(search_wids)):
                        wb = search_wids[b]
                        net.flip_weight(wa)
                        net.flip_weight(wb)
                        flipped_out = net(first_err)
                        net.flip_weight(wb)
                        net.flip_weight(wa)
                        if flipped_out != orig_out:
                            pair_candidates.append((wa, wb))

                if pair_candidates:
                    pair_scores = {}
                    for wa, wb in pair_candidates:
                        pair_scores[(wa, wb)] = _score_pair_flip(net, train_data, wa, wb)
                    best_pair = max(pair_scores, key=lambda k: pair_scores[k])
                    best_pair_score = pair_scores[best_pair]

                    if best_pair_score > 0:
                        net.flip_weight(best_pair[0])
                        net.flip_weight(best_pair[1])
                        if verbose:
                            print(f"         → PAIR FLIP (aggressive): {best_pair} fixes {best_pair_score} errors")
                        stuck_counter = 0
                        pair_success = True
            else:
                if verbose:
                    print("         No active weights — falling through to perturbation.")

        elif pair_strategy == 'hybrid':
            # ── HYBRID: scan all error rows, fallback to full search ────
            # 1. Build a global active set (derivative=1 on ANY row)
            active_wids = set()
            for wid in all_wids:
                for err_x in error_rows:
                    if net.compute_derivative(err_x, wid) == 1:
                        active_wids.add(wid)
                        break

            # If no active weights, use all weights
            if active_wids:
                search_wids = list(active_wids)
                if verbose:
                    print(f"         Active weights: {len(active_wids)} / {len(all_wids)}")
            else:
                search_wids = all_wids
                if verbose:
                    print(f"         No active weights — using all {len(all_wids)} weights")

            # 2. Try each error row until a globally helpful pair is found
            for test_row in error_rows:
                orig_out = net(test_row)
                pair_candidates = []
                for a in range(len(search_wids)):
                    wa = search_wids[a]
                    for b in range(a + 1, len(search_wids)):
                        wb = search_wids[b]
                        net.flip_weight(wa)
                        net.flip_weight(wb)
                        flipped_out = net(test_row)
                        net.flip_weight(wb)
                        net.flip_weight(wa)
                        if flipped_out != orig_out:
                            pair_candidates.append((wa, wb))

                if not pair_candidates:
                    continue   # try next error row

                # Score and apply if beneficial
                pair_scores = {}
                for wa, wb in pair_candidates:
                    pair_scores[(wa, wb)] = _score_pair_flip(net, train_data, wa, wb)
                best_pair = max(pair_scores, key=lambda k: pair_scores[k])
                best_pair_score = pair_scores[best_pair]

                if best_pair_score > 0:
                    net.flip_weight(best_pair[0])
                    net.flip_weight(best_pair[1])
                    if verbose:
                        print(f"         → PAIR FLIP (hybrid): {best_pair} fixes {best_pair_score} errors")
                    stuck_counter = 0
                    pair_success = True
                    break

            # 3. If still no success, fallback to full search on first row
            if not pair_success and active_wids:
                if verbose:
                    print("         Hybrid: falling back to full pair search on first row...")
                first_err = error_rows[0]
                orig_out = net(first_err)
                pair_candidates = []
                for a in range(len(all_wids)):
                    wa = all_wids[a]
                    for b in range(a + 1, len(all_wids)):
                        wb = all_wids[b]
                        net.flip_weight(wa)
                        net.flip_weight(wb)
                        flipped_out = net(first_err)
                        net.flip_weight(wb)
                        net.flip_weight(wa)
                        if flipped_out != orig_out:
                            pair_candidates.append((wa, wb))
                if pair_candidates:
                    pair_scores = {}
                    for wa, wb in pair_candidates:
                        pair_scores[(wa, wb)] = _score_pair_flip(net, train_data, wa, wb)
                    best_pair = max(pair_scores, key=lambda k: pair_scores[k])
                    best_pair_score = pair_scores[best_pair]
                    if best_pair_score > 0:
                        net.flip_weight(best_pair[0])
                        net.flip_weight(best_pair[1])
                        if verbose:
                            print(f"         → PAIR FLIP (hybrid fallback): {best_pair} fixes {best_pair_score} errors")
                        stuck_counter = 0
                        pair_success = True
        else:
            raise ValueError(f"Unknown pair_strategy: {pair_strategy}")

        if pair_success:
            continue

        # ── Random perturbation (last resort) ───────────────────────────
        stuck_counter += 1
        if stuck_counter > MAX_STUCK:
            if verbose:
                print(f"  ✗ Stuck {MAX_STUCK} times — stopping.")
            break

        # Flip a small random set of safe weights (increasing size)
        safe_wids = [wid for wid in all_wids if net.is_safe_flip(wid)]
        if not safe_wids:
            # Should not happen, but fallback to any weight
            safe_wids = all_wids

        n_perturb = min(1 + stuck_counter // 2, 5)
        n_perturb = min(n_perturb, len(safe_wids))
        perturb_wids = random.sample(safe_wids, n_perturb)
        for wid in perturb_wids:
            net.flip_weight(wid)
        if verbose:
            print(f"  ↳ Random perturbation [{stuck_counter}/{MAX_STUCK}]: "
                  f"flipped {len(perturb_wids)} safe weight(s)")
            for wid in perturb_wids:
                _log_flip(net, wid, None)
                
    else:
        if verbose:
            print(f"\n  ✗ Did not converge within {max_steps} steps.")

    return net

# ── Helper functions for logging ─────────────────────────────────────────────
def _log_flip(net, wid, score):
    """Pretty‑print a single weight flip."""
    kind = wid[0]
    if kind == 'layer':
        _, k, i, j = wid
        new_val = net.layers[k].W[i][j]
        action  = "ADD" if new_val else "REMOVE"
        prev_n  = net.n_inputs if k == 0 else net.layer_widths[k-1]
        inp_str = f"x{i-prev_n}" if i >= prev_n else f"x{i}"
        if i >= prev_n:
            inp_str = f"¬x{i - prev_n}"
        msg = f"         → {action} connection [Layer {k}, input {inp_str} → neuron {j}]"
        if score is not None:
            msg += f"  (fixes {score} error{'s' if score!=1 else ''})"
        print(msg)
    else:
        _, j = wid
        new_val = net.out_w[j]
        action  = "CONNECT" if new_val else "DISCONNECT"
        msg = f"         → {action} output neuron {j}"
        if score is not None:
            msg += f"  (fixes {score} error{'s' if score!=1 else ''})"
        print(msg)


def _score_pair_flip(net, data, wid_a, wid_b):
    """How many errors does flipping both weights fix?"""
    old_loss = compute_loss(net, data)
    net.flip_weight(wid_a)
    net.flip_weight(wid_b)
    new_loss = compute_loss(net, data)
    net.flip_weight(wid_b)
    net.flip_weight(wid_a)
    return old_loss - new_loss


def train_with_restarts(
    n_inputs: int,
    layer_widths: list,
    train_data: dict,
    max_steps: int = 500,
    n_restarts: int = 10,
    verbose: bool = True,
    pair_strategy: str = 'aggressive',   # pass‑through
) -> BooleanNetwork:
    """
    Train multiple networks from different random seeds and keep the best.

    Parameters
    ----------
    n_inputs, layer_widths : network architecture
    train_data : training dict
    max_steps : steps per training run
    n_restarts : how many random initializations to try
    verbose : print progress

    Returns
    -------
    The BooleanNetwork with the lowest training loss found.
    """
    best_net = None
    best_loss = float('inf')

    for r in range(1, n_restarts + 1):
        seed = random.randint(0, 2**31 - 1)
        if verbose:
            print(f"\n{'#'*64}")
            print(f"  RESTART {r}/{n_restarts}  (seed={seed})")
            print(f"{'#'*64}")

        net = BooleanNetwork(n_inputs=n_inputs, layer_widths=layer_widths, seed=seed)
        train(net, train_data, max_steps=max_steps, verbose=verbose, pair_strategy=pair_strategy)

        loss = compute_loss(net, train_data)
        if verbose:
            print(f"  Restart {r} finished with loss {loss}/{len(train_data)}")

        if loss < best_loss:
            best_loss = loss
            best_net = net
            if loss == 0:
                if verbose:
                    print("  ✓ Perfect solution found, stopping restarts.")
                break

    return best_net


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def make_dataset(
    n:                 int,
    target_fn,
    observed_fraction: float = 0.5,
    seed:              int   = 99,
) -> tuple:
    """
    Split the full 2^n truth table into train and test sets.

    observed_fraction : fraction of rows the network gets to train on
    The rest are held out to measure generalization (test accuracy).
    """
    rng      = random.Random(seed)
    all_rows = all_inputs(n)
    rng.shuffle(all_rows)

    n_train     = max(1, int(len(all_rows) * observed_fraction))
    train_rows  = all_rows[:n_train]
    test_rows   = all_rows[n_train:]

    train = {x: target_fn(x) for x in train_rows}
    test  = {x: target_fn(x) for x in test_rows}
    return train, test


def evaluate(
    net:        BooleanNetwork,
    train_data: dict,
    test_data:  dict,
    verbose:    bool = True,
) -> tuple:
    """Report train and test accuracy. Returns (train_acc, test_acc)."""
    def accuracy(data):
        if not data:
            return 0, 0
        ok = sum(net(x) == y for x, y in data.items())
        return ok, len(data)

    tr_ok, tr_n = accuracy(train_data)
    te_ok, te_n = accuracy(test_data)

    if verbose:
        print(f"\n{'='*64}")
        print(f"  Evaluation")
        print(f"{'='*64}")
        net.describe_learned()
        print()
        print(f"  Train accuracy : {tr_ok}/{tr_n}  ({100*tr_ok/tr_n:.0f}%)"
              f"  ← rows it trained on")
        if te_n:
            print(f"  Test  accuracy : {te_ok}/{te_n}  ({100*te_ok/te_n:.0f}%)"
                  f"  ← rows it never saw")
            if te_ok < te_n:
                misses = [(x, net(x), y)
                          for x, y in test_data.items() if net(x) != y]
                print(f"\n  Generalization misses ({len(misses)}):")
                for x, got, exp in misses:
                    xs = "(" + " ".join(f"x{i}={v}" for i,v in enumerate(x)) + ")"
                    print(f"    {xs}  predicted={got}  true={exp}")
        else:
            print("  Test  accuracy : N/A (full dataset used for training)")

    tr_acc = tr_ok / tr_n if tr_n else 0.0
    te_acc = te_ok / te_n if te_n else None
    return tr_acc, te_acc


# ─────────────────────────────────────────────────────────────────────────────
# Example target functions
# ─────────────────────────────────────────────────────────────────────────────

def majority(x):
    """1 if more than half the inputs are 1."""
    return int(sum(x) > len(x) / 2)

def parity(x):
    """1 if an odd number of inputs are 1 (= XOR for any n)."""
    return sum(x) % 2

def xor2(x):
    """XOR of exactly 2 inputs. Canonical non-linear problem."""
    return x[0] ^ x[1]

def at_least_two(x):
    """1 if at least 2 inputs are 1."""
    return int(sum(x) >= 2)

def custom(x):
    """Define your own function here. x is a tuple of bits."""
    # Example: (x0 AND NOT x1) OR (x2 AND x3)
    return int((x[0] and not x[1]) or (x[2] and x[3]))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    runs = [
        #  description              n  target_fn    layer_widths  frac
        ("XOR (n=2), 1 layer",      2, xor2,        [4],          1.0),
        ("XOR (n=2), 2 layers",     2, xor2,        [4, 4],       1.0),
        ("Parity (n=3), 1 layer",   3, parity,      [4],          1.0),
        ("Parity (n=3), 2 layers",  3, parity,      [6, 4],       1.0),
        ("Majority (n=3), 2 layers",3, majority,    [4, 4],       1.0),
        ("Majority (n=5), 2 layers",5, majority,    [8, 4],       0.75),
    ]

    for desc, n, fn, widths, frac in runs:
        print(f"\n\n{'#'*64}")
        print(f"  EXAMPLE: {desc}")
        print(f"{'#'*64}")
        train_data, test_data = make_dataset(n, fn, observed_fraction=frac)
        net = train_with_restarts(
            n_inputs=n,                # ← use the loop variable
            layer_widths=widths,       # ← use the loop variable
            train_data=train_data,
            max_steps=500,
            n_restarts=10,
            verbose=True
        )
        evaluate(net, train_data, test_data, verbose=True)
        input("\n  Press Enter for next example...")