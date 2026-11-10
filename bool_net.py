"""
Boolean Neural Network — Full Multi-Layer Implementation with Backpropagation
=============================================================================

This module implements a neural network where every component is Boolean.
Training is performed by a Boolean backpropagation algorithm: the Boolean
derivative of the output w.r.t. every weight is computed via a backward
pass per input sample, using the Boolean chain rule.

This is *not* a single pass whose cost is proportional to network size the
way real-valued backprop's is. Composing sensitivities across a layer
boundary isn't sound for a discrete Boolean derivative (see
`_sens_through_augmentation`), so each layer's backward step re-simulates
the forward pass through every remaining downstream layer. Measured cost
per sample is closer to O(depth^2 x width^3) — see the README's "How This
Compares to Standard Backprop" section for the derivation and benchmark.

Key ideas
─────────
1.  AND‑layer with complement augmentation (negation via extra inputs).
2.  Boolean derivative (∂f/∂w = f(w=0) XOR f(w=1)) is computed via
    chain rule during a backward pass — no brute‑force re‑evaluation,
    but not a single linear-time pass either (see above).
3.  Training: on each error row, backprop gives a *candidate* set of
    weights whose flip would change the output on that row. Candidates
    are then scored by their actual effect on total training loss (not
    by raw vote count — see README Step 10), and the best-scoring flip
    is applied. Pair‑flip and random perturbation remain as fallbacks
    when no single flip helps.
4.  Bit-packing: training data is stored column-wise as integer bitmasks
    (PackedData), so forward_packed() evaluates the whole dataset with one
    big-int AND per connection, and backward_packed() runs the backward
    pass for every error row at once. Results are bit-for-bit identical to
    the per-row forward()/backward(), which are kept as the reference.
5.  Regularization: optional `lam` penalizes active weights (README Step 13).
"""

import random
from itertools import product as iter_product


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def augment(x: tuple) -> tuple:
    """Append the bitwise complement of every element."""
    return x + tuple(1 - v for v in x)


def all_inputs(n: int) -> list:
    """All 2^n binary input tuples for n variables."""
    return list(iter_product([0, 1], repeat=n))


# ─────────────────────────────────────────────────────────────────────────────
# Bit-packed dataset
# ─────────────────────────────────────────────────────────────────────────────

class PackedData:
    """
    A dataset stored column-wise as bitmasks: bit r of x_masks[i] is the
    value of input i on row r, and bit r of y_mask is row r's label. A
    neuron's output over the WHOLE dataset is then a single integer, and an
    AND-neuron is just `&` of its required input masks — one Python big-int
    operation per connection instead of a loop over rows.

    Rows keep the dict's iteration order, so row r here is the r-th key of
    the original dict (this keeps error-row order, and therefore training's
    tie-breaks, identical to the unpacked code).
    """

    def __init__(self, data: dict):
        self.rows = list(data.keys())
        self.n_rows = len(self.rows)
        self.full = (1 << self.n_rows) - 1
        n_inputs = len(self.rows[0]) if self.rows else 0
        x_masks = [0] * n_inputs
        y_mask = 0
        for r, (x, y) in enumerate(data.items()):
            bit = 1 << r
            for i, v in enumerate(x):
                if v:
                    x_masks[i] |= bit
            if y:
                y_mask |= bit
        self.x_masks = x_masks
        self.y_mask = y_mask
        # Augmented inputs [x, ¬x] for layer 0, precomputed once.
        self.aug_masks = x_masks + [~m & self.full for m in x_masks]

    def __len__(self):
        return self.n_rows

    def rows_in(self, mask: int) -> list:
        """The rows whose bit is set in mask, in row order."""
        out = []
        while mask:
            low = mask & -mask
            out.append(self.rows[low.bit_length() - 1])
            mask ^= low
        return out


def as_packed(data) -> PackedData:
    return data if isinstance(data, PackedData) else PackedData(data)


# ─────────────────────────────────────────────────────────────────────────────
# Single layer
# ─────────────────────────────────────────────────────────────────────────────

class BooleanLayer:
    """
    A layer of independent AND‑neurons.

    Weight matrix W has shape (input_width × output_width).
        W[i][j] = 1  →  neuron j requires input i to be 1
        W[i][j] = 0  →  neuron j ignores input i
    """

    def __init__(self, input_width: int, output_width: int, rng: random.Random):
        self.input_width = input_width
        self.output_width = output_width
        m = input_width // 2                     # number of original signals
        self.W = [[0] * output_width for _ in range(input_width)]

        for j in range(output_width):
            col = [0] * input_width
            while True:
                # For each original signal i, randomly pick one of the three
                # allowed non‑contradictory patterns: (0,0), (1,0), (0,1)
                for i in range(m):
                    choice = rng.randint(0, 2)
                    if choice == 0:
                        col[i] = 0
                        col[i + m] = 0
                    elif choice == 1:
                        col[i] = 1
                        col[i + m] = 0
                    else:
                        col[i] = 0
                        col[i + m] = 1
                # Reject all‑zero column (neuron would always fire)
                if any(col[i] == 1 for i in range(input_width)):
                    break
            for i in range(input_width):
                self.W[i][j] = col[i]

    def forward(self, x: tuple) -> tuple:
        """Evaluate all neurons on input x (length input_width)."""
        h = []
        for j in range(self.output_width):
            required = [i for i in range(self.input_width) if self.W[i][j] == 1]
            h.append(int(all(x[i] for i in required)))
        return tuple(h)

    def flip(self, i: int, j: int):
        """Flip W[i][j] between 0 and 1."""
        self.W[i][j] ^= 1

    # ── Local derivative helpers for backpropagation ────────────────────────

    def _required_met(self, x_in: tuple, j: int, exclude: int = None) -> bool:
        """True if all required inputs except 'exclude' are 1."""
        for i in range(self.input_width):
            if i == exclude:
                continue
            if self.W[i][j] == 1 and x_in[i] == 0:
                return False
        return True

    def weight_derivative_local(self, i: int, j: int, x_in: tuple) -> int:
        """
        ∂(neuron_j) / ∂W[i][j]  for input x_in.
        Returns 1 if flipping W[i][j] would change the neuron's output.
        """
        # The AND output changes iff all OTHER required inputs are 1
        # AND the current input i is 0 (so the AND fails due to i).
        if x_in[i] == 1:
            return 0
        return int(self._required_met(x_in, j, exclude=i))

    def input_derivative_local(self, i: int, j: int, x_in: tuple) -> int:
        """
        ∂(neuron_j) / ∂x_in[i]  for input x_in.
        Returns 1 if flipping x_in[i] alone would change neuron j's output.
        """
        if self.W[i][j] == 0:
            return 0
        return int(self._required_met(x_in, j, exclude=i))

    def mask_str(self, j: int, n_inputs: int, var_prefix: str = "x") -> str:
        """
        Human‑readable description of neuron j's AND‑mask.

        `var_prefix` should be "x" only for a layer whose input really is
        the network's original inputs (layer 0). For any later layer, the
        "input" is actually the previous layer's neuron outputs, not the
        original variables — pass a different prefix (e.g. "h") so the
        printed formula doesn't misleadingly read as being about x0, x1, ...
        """
        parts = []
        for i in range(self.input_width):
            if self.W[i][j] == 1:
                if i < n_inputs:
                    parts.append(f"{var_prefix}{i}")
                else:
                    parts.append(f"¬{var_prefix}{i - n_inputs}")
        return " ∧ ".join(parts) if parts else "1 (always)"


# ─────────────────────────────────────────────────────────────────────────────
# Full network with backpropagation
# ─────────────────────────────────────────────────────────────────────────────

class BooleanNetwork:
    """
    Multi‑layer Boolean neural network with backpropagation.
    """

    def __init__(self, n_inputs: int, layer_widths: list, seed: int = 42):
        self.n_inputs = n_inputs
        self.layer_widths = layer_widths
        rng = random.Random(seed)

        self.layers = []
        prev_width = 2 * n_inputs                # first input is augmented
        for L in layer_widths:
            self.layers.append(BooleanLayer(prev_width, L, rng))
            prev_width = 2 * L                   # output augmented before next layer

        # Output gates: one weight per last‑layer neuron
        self.out_w = [1] * layer_widths[-1]

    # ── Forward pass (with caching for backprop) ────────────────────────────

    def forward(self, x: tuple) -> int:
        """Standard forward pass, returns 0 or 1."""
        h = augment(x)
        for k, layer in enumerate(self.layers):
            h = layer.forward(h)
            if k < len(self.layers) - 1:
                h = augment(h)
        return int(any(self.out_w[j] and h[j] for j in range(len(h))))

    def forward_with_cache(self, x: tuple):
        """
        Returns (output, cache).
        cache = {
            'aug_inputs':  [augmented inputs to each layer],
            'raw_outputs': [raw outputs of each layer before augmentation],
        }
        """
        cache = {'aug_inputs': [], 'raw_outputs': []}
        h = augment(x)
        for k, layer in enumerate(self.layers):
            cache['aug_inputs'].append(h)
            h = layer.forward(h)
            cache['raw_outputs'].append(h)
            if k < len(self.layers) - 1:
                h = augment(h)
        output = int(any(self.out_w[j] and h[j] for j in range(len(h))))
        return output, cache

    def __call__(self, x: tuple) -> int:
        return self.forward(x)

    def forward_packed(self, pd: PackedData) -> int:
        """
        Forward pass over every row of a PackedData at once. Returns the
        output bitmask (bit r = network output on row r). Equivalent to
        calling forward() on each row, but costs one big-int AND per active
        connection regardless of the number of rows.
        """
        if pd.n_rows == 0:      # no rows: can't infer input width, and
            return 0            # there's nothing to evaluate anyway
        full = pd.full
        h = pd.aug_masks
        last = len(self.layers) - 1
        for k, layer in enumerate(self.layers):
            W = layer.W
            outs = []
            for j in range(layer.output_width):
                m = full
                for i in range(layer.input_width):
                    if W[i][j]:
                        m &= h[i]
                outs.append(m)
            h = outs if k == last else outs + [~m & full for m in outs]
        out = 0
        for j, w in enumerate(self.out_w):
            if w:
                out |= h[j]
        return out

    # ── Sensitivity computations for backpropagation ───────────────────────

    def _output_sensitivity(self, h_last: tuple) -> list:
        """
        Returns list sens (length = number of last‑layer neurons) where
        sens[j] = 1 iff flipping h_last[j] would flip the final OR output.
        """
        L = len(h_last)
        sens = [0] * L
        active = [j for j in range(L) if self.out_w[j] and h_last[j] == 1]
        output = 1 if active else 0

        if output == 1:
            if len(active) == 1:
                sens[active[0]] = 1
        else:
            for j in range(L):
                if self.out_w[j]:
                    sens[j] = 1
        return sens

    def _backward_layer(self, k: int, aug_input: tuple, raw_output: tuple,
                        sens_out: list):
        """
        Backpropagate through layer k.

        Returns:
            weight_derivs : dict {(i,j): 1} for weights with derivative 1

        Note: sensitivity w.r.t. the *previous layer's* raw output is not
        computed here. It can't be soundly derived from single-neuron
        marginal sensitivities (input signal p can flip several neurons
        of this layer at once, and their combined downstream effect isn't
        guaranteed to equal the OR of their individual marginal effects) —
        see _sens_through_augmentation, which computes it exactly instead.
        """
        layer = self.layers[k]
        L = layer.output_width
        input_width = layer.input_width

        weight_derivs = {}
        for i in range(input_width):
            for j in range(L):
                if sens_out[j] and layer.weight_derivative_local(i, j, aug_input):
                    weight_derivs[(i, j)] = 1

        return weight_derivs

    def _sens_through_augmentation(self, k: int,
                                   aug_input: tuple, raw_output: tuple,
                                   final_out: int, cache: dict) -> list:
        """
        Convert sensitivity w.r.t. the augmented input of layer k into
        sensitivity w.r.t. the raw output of layer k‑1.

        For each original signal p in the previous layer, we simulate
        flipping it (which toggles both x_in[p] and x_in[p+m]) and see if
        the final output changes.
        """
        m_prev = self.layer_widths[k-1]
        sens_prev = [0] * m_prev

        # Use the cached forward values to avoid recomputing from scratch
        # For each p, we modify the augmented input to layer k and
        # re‑run from layer k onward.
        for p in range(m_prev):
            # Build modified augmented input for layer k
            mod_aug = list(aug_input)
            mod_aug[p] = 1 - mod_aug[p]
            mod_aug[p + m_prev] = 1 - mod_aug[p + m_prev]
            new_raw = self.layers[k].forward(tuple(mod_aug))
            h = new_raw
            # Propagate through remaining layers
            for k2 in range(k + 1, len(self.layers)):
                h = augment(h)
                h = self.layers[k2].forward(h)
            new_out = int(any(self.out_w[j] and h[j] for j in range(len(h))))
            if new_out != final_out:
                sens_prev[p] = 1
        return sens_prev

    def backward(self, x: tuple) -> dict:
        """
        Full backward pass for one input sample.
        Returns a dictionary mapping weight identifiers to their
        Boolean derivative (0 or 1).

        weight id format:
            ('layer', k, i, j)   for W[k][i][j]
            ('out', j)           for out_w[j]
        """
        final_out, cache = self.forward_with_cache(x)
        h_last = cache['raw_outputs'][-1]
        L_last = len(h_last)

        # Output sensitivity for last layer neurons
        sens_out = self._output_sensitivity(h_last)

        all_derivs = {}

        # Process layers from last down to first
        for k in reversed(range(len(self.layers))):
            aug_input = cache['aug_inputs'][k]
            raw_output = cache['raw_outputs'][k]

            w_derivs = self._backward_layer(k, aug_input, raw_output, sens_out)
            for (i, j), d in w_derivs.items():
                all_derivs[('layer', k, i, j)] = d

            # If not the first layer, compute sensitivity for previous layer
            if k > 0:
                sens_out = self._sens_through_augmentation(
                    k, aug_input, raw_output, final_out, cache)

        # Output gate derivatives
        for j in range(len(self.out_w)):
            orig = self.out_w[j]
            self.out_w[j] = 0
            f0 = self.forward(x)
            self.out_w[j] = 1
            f1 = self.forward(x)
            self.out_w[j] = orig
            all_derivs[('out', j)] = f0 ^ f1

        return all_derivs

    # ── Bit-packed backward pass (all rows at once) ─────────────────────────

    def _forward_packed_from(self, k: int, h_aug: list, full: int) -> int:
        """Packed forward starting at layer k, given its augmented input
        masks. Returns the output mask."""
        last = len(self.layers) - 1
        h = h_aug
        for k2 in range(k, last + 1):
            layer = self.layers[k2]
            W = layer.W
            outs = []
            for j in range(layer.output_width):
                m = full
                for i in range(layer.input_width):
                    if W[i][j]:
                        m &= h[i]
                outs.append(m)
            h = outs if k2 == last else outs + [~m & full for m in outs]
        out = 0
        for j, w in enumerate(self.out_w):
            if w:
                out |= h[j]
        return out

    def backward_packed(self, pd: PackedData, rows: int = None) -> dict:
        """
        backward() for every row selected by the `rows` bitmask at once
        (default: all rows). Returns {weight_id: mask}, where bit r of mask
        is ∂f/∂w on row r — exactly what backward(pd.rows[r]) would give.
        Weights whose mask is 0 are omitted. Keys are in backward()'s own
        canonical order (layers last→first, then i, then j; then outputs).

        This is the same algorithm, just vectorized across rows: every step
        of backward() (local AND derivatives, re-simulating a flipped signal
        through the downstream layers) acts on each row independently, so
        it can act on all rows' bits simultaneously. Re-simulation stays
        exact — no unsound composition of sensitivities is introduced.
        """
        full = pd.full
        if rows is None:
            rows = full
        if pd.n_rows == 0 or rows == 0:
            return {}
        last = len(self.layers) - 1

        # Forward with cache
        aug_inputs, raw_outputs = [], []
        h = pd.aug_masks
        for k, layer in enumerate(self.layers):
            aug_inputs.append(h)
            W = layer.W
            outs = []
            for j in range(layer.output_width):
                m = full
                for i in range(layer.input_width):
                    if W[i][j]:
                        m &= h[i]
                outs.append(m)
            raw_outputs.append(outs)
            h = outs if k == last else outs + [~m & full for m in outs]
        h_last = raw_outputs[-1]
        active = [h_last[j] if self.out_w[j] else 0 for j in range(len(h_last))]
        final_out, one, two = 0, 0, 0
        for a in active:              # track "≥1 active" and "≥2 active"
            two |= one & a
            one |= a
        final_out = one
        not_out = ~final_out & full

        # Output sensitivity of each last-layer neuron (cf. _output_sensitivity)
        sens = [((active[j] & ~two) | not_out) if self.out_w[j] else 0
                for j in range(len(h_last))]

        derivs = {}
        for k in reversed(range(len(self.layers))):
            layer = self.layers[k]
            x_in = aug_inputs[k]
            W = layer.W
            n_in = layer.input_width
            for i in range(n_in):
                not_xi = ~x_in[i] & full & rows
                if not not_xi:
                    continue
                for j in range(layer.output_width):
                    s = sens[j] & not_xi
                    if not s:
                        continue
                    # all required inputs except i are 1 — if i isn't
                    # required, that's just neuron j's cached output.
                    if W[i][j]:
                        m = full
                        for i2 in range(n_in):
                            if i2 != i and W[i2][j]:
                                m &= x_in[i2]
                    else:
                        m = raw_outputs[k][j]
                    d = s & m
                    if d:
                        derivs[('layer', k, i, j)] = d
            if k > 0:
                # Sensitivity w.r.t. previous layer's raw outputs: flip
                # signal p (and its complement) on every row, re-simulate.
                m_prev = self.layer_widths[k - 1]
                new_sens = []
                for p in range(m_prev):
                    mod = list(x_in)
                    mod[p] ^= full
                    mod[p + m_prev] ^= full
                    new_sens.append(
                        self._forward_packed_from(k, mod, full) ^ final_out)
                sens = new_sens

        # Output gate derivatives: f(out_w[j]=1) XOR f(out_w[j]=0)
        L = len(h_last)
        for j in range(L):
            others = 0
            for j2 in range(L):
                if j2 != j:
                    others |= active[j2]
            d = h_last[j] & ~others & rows
            if d:
                derivs[('out', j)] = d
        return derivs

    # ── Weight enumeration (used by fallback strategies) ────────────────────

    def all_weight_ids(self):
        """Iterate over all learnable weight identifiers."""
        for k, layer in enumerate(self.layers):
            for i in range(layer.input_width):
                for j in range(layer.output_width):
                    yield ('layer', k, i, j)
        for j in range(len(self.out_w)):
            yield ('out', j)

    def flip_weight(self, wid):
        """Apply a single weight flip."""
        if wid[0] == 'layer':
            _, k, i, j = wid
            self.layers[k].flip(i, j)
        else:
            _, j = wid
            self.out_w[j] ^= 1

    def is_safe_flip(self, wid) -> bool:
        """
        Check that flipping wid, by itself, does not create a contradiction
        (a neuron requiring both a signal and its complement).

        This is exact when only ONE weight is changing. It is NOT sufficient
        when multiple weights are flipped together in the same batch: two
        flips that are each individually safe against the *original* state
        can still jointly create a contradiction (e.g. simultaneously
        requiring signal i and its complement, when neither was required
        before). Callers that apply several flips at once should either
        re-check safety against the *updated* state before each flip, or
        use has_contradiction() to check the resulting state directly.
        """
        if wid[0] == 'out':
            return True
        _, k, i, j = wid
        layer = self.layers[k]
        current = layer.W[i][j]
        if current == 1:
            return True
        m = layer.input_width // 2
        i_comp = i + m if i < m else i - m
        return layer.W[i_comp][j] == 0

    def has_contradiction(self, k: int = None, j: int = None) -> bool:
        """
        Check whether any neuron currently requires both a signal and its
        complement (making it permanently dead: always outputs 0).
        Optionally restrict the scan to a single layer (k) and/or a single
        neuron within that layer (j) to keep the check cheap when the caller
        already knows which neuron(s) could have been affected.
        """
        layers_to_check = [self.layers[k]] if k is not None else self.layers
        for layer in layers_to_check:
            m = layer.input_width // 2
            cols = [j] if j is not None else range(layer.output_width)
            for jj in cols:
                for i in range(m):
                    if layer.W[i][jj] == 1 and layer.W[i + m][jj] == 1:
                        return True
        return False

    def complexity(self) -> int:
        """
        Number of active (=1) weights: AND-layer connections plus connected
        output gates. This is the description length the regularizer
        penalizes; fewer literals per AND-neuron means a more general term.
        """
        return (sum(sum(row) for layer in self.layers for row in layer.W)
                + sum(self.out_w))

    def get_state(self):
        """Snapshot of all weights (for restoring the best state seen)."""
        return ([[row[:] for row in layer.W] for layer in self.layers],
                self.out_w[:])

    def set_state(self, state):
        Ws, out_w = state
        for layer, W in zip(self.layers, Ws):
            layer.W = [row[:] for row in W]
        self.out_w = out_w[:]

    # ── Description ─────────────────────────────────────────────────────────

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
        n = self.n_inputs
        print("  Learned function:")
        for k, layer in enumerate(self.layers):
            prev_n = n if k == 0 else self.layer_widths[k-1]
            var_prefix = "x" if k == 0 else "h"
            print(f"  Layer {k}:" + ("" if k == 0 else f"  (h_i = layer {k-1}'s neuron i output)"))
            for j in range(layer.output_width):
                label = layer.mask_str(j, prev_n, var_prefix=var_prefix)
                active = "●" if k == len(self.layers)-1 and self.out_w[j] else " "
                print(f"    {active} neuron {j}: {label}")
        active_out = [j for j in range(len(self.out_w)) if self.out_w[j]]
        print(f"  Output = OR of neurons {active_out}")


# ─────────────────────────────────────────────────────────────────────────────
# Loss and fallback scoring (used only for pair/perturbation search)
# ─────────────────────────────────────────────────────────────────────────────

def compute_loss(net: BooleanNetwork, data) -> int:
    """Number of misclassified rows. `data` may be a dict or a PackedData;
    pass a PackedData when calling repeatedly on the same data (train()
    does) to avoid re-packing each time."""
    pd = as_packed(data)
    return ((net.forward_packed(pd) ^ pd.y_mask) & pd.full).bit_count()


def error_mask(net: BooleanNetwork, pd: PackedData) -> int:
    """Bitmask of misclassified rows."""
    return (net.forward_packed(pd) ^ pd.y_mask) & pd.full


def influence_rows(n_inputs: int, max_rows: int = 4096, seed: int = 0) -> PackedData:
    """
    Rows on which total influence is measured: the full 2^n domain when it
    has at most max_rows rows, otherwise a fixed random sample (seeded, so
    the objective is deterministic). Labels are unused.
    """
    if 2 ** n_inputs <= max_rows:
        rows = all_inputs(n_inputs)
    else:
        rng = random.Random(seed)
        rows = list({tuple(rng.randint(0, 1) for _ in range(n_inputs))
                     for _ in range(max_rows)})
        rows.sort()
    return PackedData({x: 0 for x in rows})


def total_influence(net: BooleanNetwork, pd: PackedData) -> float:
    """
    Total influence (average sensitivity) of the network's function:

        I(f) = E_x [ #{ i : f(x) ≠ f(x ⊕ e_i) } ]  ∈ [0, n]

    i.e. the expected number of input bits whose flip changes the output —
    the Boolean derivative ∂f/∂x_i, averaged over inputs and summed over
    variables. It's the Boolean analogue of a smoothness / Lipschitz
    penalty: 0 for a constant, n for parity, about 1.9 for 5-input majority.

    Bit-packed, flipping x_i on every row at once is just swapping the
    masks of x_i and ¬x_i in the augmented input, so this costs n+1 packed
    forward passes.
    """
    if pd.n_rows == 0:
        return 0.0
    base = net.forward_packed(pd)
    n = net.n_inputs
    aug = pd.aug_masks
    total = 0
    for i in range(n):
        mod = list(aug)
        mod[i], mod[i + n] = mod[i + n], mod[i]
        total += (net._forward_packed_from(0, mod, pd.full) ^ base).bit_count()
    return total / pd.n_rows


class Objective:
    """
    Regularized training objective, as a callable net → number:

        objective = training_errors
                  + lam × (number of active weights)      (Step 13)
                  + mu  × total_influence(f)              (Step 15)

    With lam == mu == 0 it returns exactly compute_loss (an int), so the
    unregularized training path is unchanged.

    About lam: for a *single* flip with 0 < lam < 1, fixing an error always
    outweighs the ±1 change in complexity, so step-by-step the weight
    penalty acts as a tie-breaker. But whole networks are compared too
    (best-state restore in train(), restart selection in
    train_with_restarts), and there any lam > 0 will accept one extra
    training error in exchange for more than 1/lam fewer weights.

    About mu: influence is measured on ALL inputs (or a large sample — see
    influence_rows), not just training rows, so it constrains how the
    function behaves on unseen inputs — that's the point.
    """

    def __init__(self, data, lam: float = 0.0, mu: float = 0.0,
                 n_inputs: int = None):
        self.pd = as_packed(data)
        self.lam = lam
        self.mu = mu
        self._n_inputs = n_inputs
        self._infl_pd = None

    @property
    def regularized(self) -> bool:
        return bool(self.lam or self.mu)

    def influence(self, net) -> float:
        if self._infl_pd is None:
            self._infl_pd = influence_rows(self._n_inputs or net.n_inputs)
        return total_influence(net, self._infl_pd)

    def __call__(self, net) -> float:
        v = compute_loss(net, self.pd)
        if self.lam:
            v += self.lam * net.complexity()
        if self.mu:
            v += self.mu * self.influence(net)
        return v


def compute_objective(net: BooleanNetwork, data, lam: float = 0.0,
                      mu: float = 0.0):
    """One-off evaluation of the regularized objective (see Objective)."""
    return Objective(data, lam, mu)(net)


def _score_pair_flip(net, objective, wid_a, wid_b):
    old = objective(net)
    net.flip_weight(wid_a)
    net.flip_weight(wid_b)
    new = objective(net)
    net.flip_weight(wid_b)
    net.flip_weight(wid_a)
    return old - new


def _score_single_flip(net, objective, wid):
    """Net objective reduction (old - new) from applying a single flip, over
    the FULL dataset — not just the error rows that voted for it. A flip that
    fixes some error rows can simultaneously break rows that were already
    correct; this is what actually decides whether the flip helps. When the
    objective is regularized, the change in penalty is included too."""
    old = objective(net)
    net.flip_weight(wid)
    new = objective(net)
    net.flip_weight(wid)
    return old - new


def prune(net: BooleanNetwork, data, lam: float, verbose: bool = False) -> int:
    """
    Greedy Boolean Occam's razor: repeatedly remove the active weight whose
    removal most reduces the regularized objective, until no removal helps.
    For 0 < lam < 1 this means "drop any connection that doesn't cost a
    training error". Removing a weight can never create a contradiction,
    so no safety check is needed. Returns the number of weights removed.
    """
    if lam <= 0:
        return 0
    objective = Objective(data, lam)
    removed = 0
    while True:
        best_wid, best_score = None, 0
        for wid in net.all_weight_ids():      # deterministic order
            if _weight_value(net, wid) != 1:
                continue
            score = _score_single_flip(net, objective, wid)
            if score > best_score:
                best_wid, best_score = wid, score
        if best_wid is None:
            return removed
        net.flip_weight(best_wid)
        removed += 1
        if verbose:
            _log_flip(net, best_wid, None)


def polish(net: BooleanNetwork, objective: Objective, verbose: bool = False) -> int:
    """
    Greedy local search over ALL single flips (additions as well as
    removals), applying the best objective-reducing flip until none helps.
    prune() only removes weights, which is enough for the weight-count
    penalty; for the influence penalty adding a literal can just as well be
    what makes the function smoother. Only contradiction-free flips are
    considered. Terminates because every applied flip strictly decreases
    the objective. Returns the number of flips applied.
    """
    applied = 0
    while True:
        best_wid, best_score = None, 0
        for wid in net.all_weight_ids():
            if not net.is_safe_flip(wid):
                continue
            score = _score_single_flip(net, objective, wid)
            if score > best_score:
                best_wid, best_score = wid, score
        if best_wid is None:
            return applied
        net.flip_weight(best_wid)
        applied += 1
        if verbose:
            _log_flip(net, best_wid, None)


def _weight_value(net, wid) -> int:
    if wid[0] == 'layer':
        _, k, i, j = wid
        return net.layers[k].W[i][j]
    return net.out_w[wid[1]]


# ─────────────────────────────────────────────────────────────────────────────
# Training loop using backpropagation
# ─────────────────────────────────────────────────────────────────────────────

def train(
    net:        BooleanNetwork,
    train_data: dict,
    max_steps:  int  = 500,
    verbose:    bool = True,
    seed:       int  = None,
    lam:        float = 0.0,
    mu:         float = 0.0,
) -> BooleanNetwork:
    """
    Train using Boolean backpropagation.

    1. Identify misclassified rows.
    2. For each error row, run net.backward(x) to get the set of weights
       whose flip would change the output for that row.
    3. Count how many error rows each weight would fix.
    4. Flip the weight with the highest count (if > 0).
    5. If no single flip helps, try a pair‑flip search among active weights.
    6. If still stuck, apply a small random perturbation.

    `seed` controls the ONLY source of randomness inside this function (the
    perturbation fallback in step 6). Pass a seed to make a training run
    fully reproducible given the same net, data, and seed — otherwise this
    behaves as before and draws from OS entropy, differing run to run.

    `lam` is the Boolean regularization strength (see compute_objective).
    With lam == 0 (default) training is unchanged. With lam > 0, candidate
    flips are scored by the regularized objective, the lowest-objective
    state seen is restored at the end (a late random perturbation can't
    leave the net worse than its best), and a final prune() pass removes
    every connection that isn't paying for itself.

    `mu` adds the total-influence (smoothness) penalty — see Objective. With
    mu > 0 the final pass is polish() (any single flip, not just removals).
    """
    rng = random.Random(seed)
    best_state, best_obj = None, float('inf')
    packed = as_packed(train_data)   # pack once; all loss evals use it
    objective = Objective(packed, lam, mu, n_inputs=net.n_inputs)
    MAX_STUCK = 5
    n_train = len(train_data)
    stuck_counter = 0
    all_wids = list(net.all_weight_ids())

    if verbose:
        print(f"\n{'='*64}")
        print(f"  Training with backprop  |  n={net.n_inputs}  |  "
              f"{n_train}/{2**net.n_inputs} rows")
        net.describe()
        print(f"{'='*64}")

    for step in range(1, max_steps + 1):
        loss = compute_loss(net, packed)
        if verbose:
            reg = f"  |  active weights {net.complexity()}" if lam else ""
            if mu:
                reg += f"  |  influence {objective.influence(net):.3f}"
            print(f"\nStep {step:>4}  |  Loss {loss}/{n_train}{reg}")
        if objective.regularized:
            obj = objective(net)
            if obj < best_obj:
                best_obj, best_state = obj, net.get_state()

        if loss == 0:
            if verbose:
                print("  ✓ Zero training loss — converged.")
            break

        # Collect all misclassified rows
        error_rows = packed.rows_in(error_mask(net, packed))  # used by pair search

        # ── 1. Single‑flip candidates via backprop ─────────────────
        # One bit-packed backward pass over all error rows at once: each
        # weight gets a mask of the error rows where its derivative is 1.
        err = error_mask(net, packed)
        deriv_masks = net.backward_packed(packed, err)
        # wid -> number of error rows where derivative=1. Inserted in the
        # same order the old per-row loop used (by first error row where the
        # derivative is 1, then canonical weight order) so that tie-breaks
        # in the stable sort below are unchanged.
        weight_votes = {
            wid: m.bit_count() for wid, m in
            sorted(deriv_masks.items(), key=lambda kv: (kv[1] & -kv[1]))
        }

        # Only consider candidates that don't introduce a contradiction, and
        # verify each against the FULL dataset (not just the error rows that
        # voted for it) — a flip can fix some wrong rows while breaking rows
        # that were already correct. Vote count is used only to order the
        # search, not to decide whether a flip is actually accepted.
        safe_candidates = sorted(
            (wid for wid in weight_votes if net.is_safe_flip(wid)),
            key=weight_votes.get, reverse=True,
        )
        best_wid, best_net_score = None, 0
        for wid in safe_candidates:
            score = _score_single_flip(net, objective, wid)
            if score > best_net_score:
                best_net_score, best_wid = score, wid

        if best_wid is not None:
            net.flip_weight(best_wid)
            if verbose:
                _log_flip(net, best_wid, best_net_score)
            stuck_counter = 0
            continue

        # ── 2. No single flip helped — try pair flips ──────────────
        if verbose:
            print("         No beneficial single flip — searching pairs...")

        # Active weights (derivative=1 on any error row). The net hasn't
        # changed since step 1, so the masks from that backward pass are
        # still valid — no need to recompute.
        active_wids = set(deriv_masks)
        if not active_wids:
            active_wids = set(all_wids)   # fallback to all weights

        active_list = sorted(active_wids)  # deterministic order: a set's
        # iteration order depends on string hashing, which Python salts
        # differently per process (PYTHONHASHSEED) unless pinned — sorting
        # by value removes that dependence so pair search order (and its
        # tie-breaks in max(pair_scores, ...) below) is reproducible.
        pair_success = False
        # Try pairs on the first error row (quick filter)
        first_err = error_rows[0]
        orig_out = net(first_err)
        pair_candidates = []
        for a in range(len(active_list)):
            wa = active_list[a]
            for b in range(a + 1, len(active_list)):
                wb = active_list[b]
                net.flip_weight(wa)
                net.flip_weight(wb)
                flipped_out = net(first_err)
                # Two flips that are each individually safe can still
                # jointly create a contradiction (e.g. requiring a signal
                # and its complement together) — check the resulting state
                # directly rather than relying on is_safe_flip per-weight.
                contradiction = net.has_contradiction()
                net.flip_weight(wb)
                net.flip_weight(wa)
                if flipped_out != orig_out and not contradiction:
                    pair_candidates.append((wa, wb))

        if pair_candidates:
            pair_scores = {pair: _score_pair_flip(net, objective, pair[0], pair[1])
                           for pair in pair_candidates}
            best_pair = max(pair_scores, key=pair_scores.get)
            best_pair_score = pair_scores[best_pair]
            if best_pair_score > 0:
                net.flip_weight(best_pair[0])
                net.flip_weight(best_pair[1])
                if verbose:
                    print(f"         → PAIR FLIP: {best_pair} fixes {best_pair_score} errors")
                stuck_counter = 0
                pair_success = True

        if pair_success:
            continue

        # ── 3. Random perturbation to escape plateau ───────────────
        stuck_counter += 1
        if stuck_counter > MAX_STUCK:
            if verbose:
                print(f"  ✗ Stuck {MAX_STUCK} times — stopping.")
            break

        safe = [w for w in all_wids if net.is_safe_flip(w)]
        if not safe:
            safe = all_wids
        n_perturb = min(1 + stuck_counter // 2, 5)
        n_perturb = min(n_perturb, len(safe))
        candidates = rng.sample(safe, n_perturb)
        # Re-check safety against the UPDATED state right before each flip:
        # a weight sampled as safe against the pre-batch state can become
        # unsafe once an earlier flip in this same batch has been applied
        # (e.g. sampling both a signal and its complement for the same
        # neuron — each looks safe alone, but applying both together
        # creates a contradiction).
        applied = []
        for w in candidates:
            if net.is_safe_flip(w):
                net.flip_weight(w)
                applied.append(w)
        if verbose:
            print(f"  ↳ Random perturbation [{stuck_counter}/{MAX_STUCK}]: "
                  f"flipped {len(applied)} safe weight(s)")
            for w in applied:
                _log_flip(net, w, None)

    else:
        if verbose:
            print(f"\n  ✗ Did not converge within {max_steps} steps.")

    if objective.regularized:
        if objective(net) > best_obj:
            net.set_state(best_state)
        if mu:
            n_changed = polish(net, objective, verbose=verbose)
            what = "Polished"
        else:
            n_changed = prune(net, packed, lam, verbose=verbose)
            what = "Pruned"
        if verbose:
            print(f"  {what}: {n_changed} flip(s) → {net.complexity()} active, "
                  f"loss {compute_loss(net, packed)}/{n_train}")
    return net


def _log_flip(net, wid, score):
    """Pretty‑print a weight flip."""
    kind = wid[0]
    if kind == 'layer':
        _, k, i, j = wid
        new_val = net.layers[k].W[i][j]
        action = "ADD" if new_val else "REMOVE"
        prev_n = net.n_inputs if k == 0 else net.layer_widths[k-1]
        var_prefix = "x" if k == 0 else "h"
        if i >= prev_n:
            inp_str = f"¬{var_prefix}{i - prev_n}"
        else:
            inp_str = f"{var_prefix}{i}"
        msg = f"         → {action} connection [Layer {k}, input {inp_str} → neuron {j}]"
        if score is not None:
            msg += f"  (fixes {score} error{'s' if score!=1 else ''})"
        print(msg)
    else:
        _, j = wid
        new_val = net.out_w[j]
        action = "CONNECT" if new_val else "DISCONNECT"
        msg = f"         → {action} output neuron {j}"
        if score is not None:
            msg += f"  (fixes {score} error{'s' if score!=1 else ''})"
        print(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Train with restarts (wrapper)
# ─────────────────────────────────────────────────────────────────────────────

def train_with_restarts(
    n_inputs: int,
    layer_widths: list,
    train_data: dict,
    max_steps: int = 500,
    n_restarts: int = 10,
    verbose: bool = True,
    seed: int = None,
    lam: float = 0.0,
    mu: float = 0.0,
) -> BooleanNetwork:
    """
    `seed` controls both the per-restart network initialization AND the
    randomness inside each restart's train() call, so the entire multi-restart
    run is reproducible end to end given the same seed. Leave it as None for
    the previous behavior (draws from OS entropy, differs run to run).

    With lam > 0 every restart runs (no early stop at zero loss, since a
    later restart may find an equally accurate but simpler network) and the
    restart with the lowest regularized objective is kept.
    """
    rng = random.Random(seed)
    objective = Objective(train_data, lam, mu, n_inputs=n_inputs)
    best_net = None
    best_loss = float('inf')
    for r in range(1, n_restarts + 1):
        net_seed = rng.randint(0, 2**31 - 1)
        train_seed = rng.randint(0, 2**31 - 1)
        if verbose:
            print(f"\n{'#'*64}")
            print(f"  RESTART {r}/{n_restarts}  (seed={net_seed})")
            print(f"{'#'*64}")
        net = BooleanNetwork(n_inputs=n_inputs, layer_widths=layer_widths, seed=net_seed)
        train(net, train_data, max_steps=max_steps, verbose=verbose,
              seed=train_seed, lam=lam, mu=mu)
        loss = objective(net)
        if verbose:
            label = "objective" if objective.regularized else "loss"
            print(f"  Restart {r} finished with {label} {loss}")
        if loss < best_loss:
            best_loss = loss
            best_net = net
            if loss == 0 and not objective.regularized:
                if verbose:
                    print("  ✓ Perfect solution found, stopping restarts.")
                break
    return best_net


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble (majority vote over independently trained nets)
# ─────────────────────────────────────────────────────────────────────────────

class Ensemble:
    """
    Majority vote of several networks. Callable like a BooleanNetwork, so
    evaluate(..., verbose=False) and compute_loss work on it unchanged.
    Use an odd number of members to avoid ties (a tie predicts 0).
    """

    def __init__(self, members: list):
        self.members = members

    def __call__(self, x: tuple) -> int:
        return int(2 * sum(m(x) for m in self.members) > len(self.members))

    def forward_packed(self, pd: PackedData) -> int:
        masks = [m.forward_packed(pd) for m in self.members]
        need = len(masks) // 2 + 1
        out = 0
        for r in range(pd.n_rows):
            bit = 1 << r
            if sum(1 for m in masks if m & bit) >= need:
                out |= bit
        return out

    def complexity(self) -> int:
        return sum(m.complexity() for m in self.members)


def train_ensemble(
    n_inputs: int,
    layer_widths: list,
    train_data: dict,
    n_members: int = 9,
    max_steps: int = 500,
    seed: int = None,
    lam: float = 0.0,
    mu: float = 0.0,
) -> Ensemble:
    """
    Train n_members networks independently (one train() run each — no
    restarts), keep those that reached the lowest objective among them,
    and return their majority vote.
    """
    rng = random.Random(seed)
    objective = Objective(train_data, lam, mu, n_inputs=n_inputs)
    trained = []
    for _ in range(n_members):
        net = BooleanNetwork(n_inputs, layer_widths, seed=rng.randint(0, 2**31 - 1))
        train(net, train_data, max_steps=max_steps, verbose=False,
              seed=rng.randint(0, 2**31 - 1), lam=lam, mu=mu)
        trained.append((objective(net), net))
    best = min(o for o, _ in trained)
    keep = [net for o, net in trained if o == best]
    if len(keep) % 2 == 0:        # keep it odd: drop the last
        keep = keep[:-1]
    return Ensemble(keep)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation and dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_dataset(n: int, target_fn, observed_fraction: float = 0.5,
                 seed: int = 99) -> tuple:
    rng = random.Random(seed)
    all_rows = all_inputs(n)
    rng.shuffle(all_rows)
    n_train = max(1, int(len(all_rows) * observed_fraction))
    train = {x: target_fn(x) for x in all_rows[:n_train]}
    test  = {x: target_fn(x) for x in all_rows[n_train:]}
    return train, test


def evaluate(net: BooleanNetwork, train_data: dict, test_data: dict,
             verbose: bool = True) -> tuple:
    def acc(data):
        if not data: return 0, 0
        ok = sum(net(x) == y for x, y in data.items())
        return ok, len(data)

    tr_ok, tr_n = acc(train_data)
    te_ok, te_n = acc(test_data)

    if verbose:
        print(f"\n{'='*64}")
        print(f"  Evaluation")
        print(f"{'='*64}")
        net.describe_learned()
        print(f"\n  Train accuracy : {tr_ok}/{tr_n}  ({100*tr_ok/tr_n:.0f}%)")
        if te_n:
            print(f"  Test  accuracy : {te_ok}/{te_n}  ({100*te_ok/te_n:.0f}%)")
            if te_ok < te_n:
                misses = [(x, net(x), y) for x, y in test_data.items() if net(x) != y]
                print(f"\n  Generalization misses ({len(misses)}):")
                for x, got, exp in misses:
                    xs = "(" + " ".join(f"x{i}={v}" for i,v in enumerate(x)) + ")"
                    print(f"    {xs}  predicted={got}  true={exp}")
        else:
            print("  Test  accuracy : N/A")
    return tr_ok/tr_n if tr_n else 0.0, te_ok/te_n if te_n else None


# ─────────────────────────────────────────────────────────────────────────────
# Example target functions
# ─────────────────────────────────────────────────────────────────────────────

def majority(x): return int(sum(x) > len(x)/2)
def parity(x):   return sum(x) % 2
def xor2(x):     return x[0] ^ x[1]
def at_least_two(x): return int(sum(x) >= 2)
def custom(x):   return int((x[0] and not x[1]) or (x[2] and x[3]))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    runs = [
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
            n_inputs=n,
            layer_widths=widths,
            train_data=train_data,
            max_steps=500,
            n_restarts=10,
            verbose=True
        )
        evaluate(net, train_data, test_data, verbose=True)
        input("\n  Press Enter for next example...")