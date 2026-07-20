"""
Boolean Neural Network — Full Multi-Layer Implementation with Backpropagation
=============================================================================

This module implements a neural network where every component is Boolean.
Training is now performed by a true Boolean backpropagation algorithm:
the Boolean derivative of the output w.r.t. every weight is computed in a
single backward pass per input sample, using the Boolean chain rule.

Key ideas
─────────
1.  AND‑layer with complement augmentation (negation via extra inputs).
2.  Boolean derivative (∂f/∂w = f(w=0) XOR f(w=1)) is computed via
    chain rule during a backward pass — no brute‑force re‑evaluation.
3.  Training: on each error row, backprop gives the set of weights that
    would flip the output. We aggregate these over all error rows and
    flip the weight that fixes the most errors.  Stuck escaping via
    pair‑flip and random perturbation remains as a fallback.
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

    def mask_str(self, j: int, n_inputs: int) -> str:
        """Human‑readable description of neuron j's AND‑mask."""
        parts = []
        for i in range(self.input_width):
            if self.W[i][j] == 1:
                if i < n_inputs:
                    parts.append(f"x{i}")
                else:
                    parts.append(f"¬x{i - n_inputs}")
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
            sens_in       : list of length len(aug_input), sensitivity of
                            final output w.r.t. each input wire.
        """
        layer = self.layers[k]
        L = layer.output_width
        input_width = layer.input_width

        weight_derivs = {}
        for i in range(input_width):
            for j in range(L):
                if sens_out[j] and layer.weight_derivative_local(i, j, aug_input):
                    weight_derivs[(i, j)] = 1

        sens_in = [0] * input_width
        for i in range(input_width):
            for j in range(L):
                if sens_out[j] and layer.input_derivative_local(i, j, aug_input):
                    sens_in[i] = 1
                    break
        return weight_derivs, sens_in

    def _sens_through_augmentation(self, k: int, sens_in: list,
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

            w_derivs, sens_in = self._backward_layer(k, aug_input, raw_output, sens_out)
            for (i, j), d in w_derivs.items():
                all_derivs[('layer', k, i, j)] = d

            # If not the first layer, compute sensitivity for previous layer
            if k > 0:
                sens_out = self._sens_through_augmentation(
                    k, sens_in, aug_input, raw_output, final_out, cache)
            # For the first layer (k=0), sens_in gives sensitivity to original inputs,
            # but we don't need it for weight updates.

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
        """Check that flipping wid does not create a contradiction."""
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
            print(f"  Layer {k}:")
            for j in range(layer.output_width):
                label = layer.mask_str(j, prev_n)
                active = "●" if k == len(self.layers)-1 and self.out_w[j] else " "
                print(f"    {active} neuron {j}: {label}")
        active_out = [j for j in range(len(self.out_w)) if self.out_w[j]]
        print(f"  Output = OR of neurons {active_out}")


# ─────────────────────────────────────────────────────────────────────────────
# Loss and fallback scoring (used only for pair/perturbation search)
# ─────────────────────────────────────────────────────────────────────────────

def compute_loss(net: BooleanNetwork, data: dict) -> int:
    return sum(net(x) != y for x, y in data.items())


def _score_pair_flip(net, data, wid_a, wid_b):
    old = compute_loss(net, data)
    net.flip_weight(wid_a)
    net.flip_weight(wid_b)
    new = compute_loss(net, data)
    net.flip_weight(wid_b)
    net.flip_weight(wid_a)
    return old - new


def _score_flip(net, data, wid):
    """
    True effect of flipping a single weight on total training loss.

    NOTE: this is deliberately NOT the same thing as "how many error rows
    have derivative=1 for this weight". A weight's Boolean derivative being 1
    on an error row only tells you the output *would* flip on that row in
    isolation — it says nothing about whether the same flip also flips the
    output on rows that were already correct. Scoring by vote count alone
    can pick a flip that fixes k error rows while breaking k+1 correct ones,
    a net loss increase disguised as a "positive" move. Scoring by actual
    loss delta (as we already do for pair flips) avoids that.
    """
    old = compute_loss(net, data)
    net.flip_weight(wid)
    new = compute_loss(net, data)
    net.flip_weight(wid)
    return old - new


# ─────────────────────────────────────────────────────────────────────────────
# Training loop using backpropagation
# ─────────────────────────────────────────────────────────────────────────────

def train(
    net:        BooleanNetwork,
    train_data: dict,
    max_steps:  int  = 500,
    verbose:    bool = True,
) -> BooleanNetwork:
    """
    Train using Boolean backpropagation.

    1. Identify misclassified rows.
    2. For each error row, run net.backward(x) to get the set of weights
       whose flip would change the output for that row — this is only a
       CANDIDATE filter (weights with no effect on any error row can't help).
    3. Score every candidate by its true effect on total training loss
       (flip → compute_loss → unflip), exactly like the pair‑flip search
       already does. A weight that flips several error rows to correct can
       simultaneously flip other, previously‑correct rows to wrong — vote
       counting alone can't see that, so we check real loss instead.
    4. Flip the candidate with the best (most negative) loss delta, if any
       candidate actually reduces loss.
    5. If no single flip helps, try a pair‑flip search among candidates.
    6. If still stuck, apply a small random perturbation.
    """
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
        loss = compute_loss(net, train_data)
        if verbose:
            print(f"\nStep {step:>4}  |  Loss {loss}/{n_train}")

        if loss == 0:
            if verbose:
                print("  ✓ Zero training loss — converged.")
            break

        # Collect all misclassified rows
        error_rows = [x for x, y in train_data.items() if net(x) != y]

        # ── 1. Single‑flip candidates via backprop ─────────────────
        # Backprop gives us a cheap CANDIDATE filter: only weights with
        # derivative=1 on at least one error row can possibly help.
        candidate_wids = set()
        for x in error_rows:
            derivs = net.backward(x)
            for wid, d in derivs.items():
                if d == 1:
                    candidate_wids.add(wid)

        best_wid, best_score = None, 0
        for wid in candidate_wids:
            score = _score_flip(net, train_data, wid)   # true loss delta
            if score > best_score:
                best_score = score
                best_wid = wid

        if best_wid is not None:
            net.flip_weight(best_wid)
            if verbose:
                _log_flip(net, best_wid, best_score)
            stuck_counter = 0
            continue

        # ── 2. No single flip helped — try pair flips ──────────────
        if verbose:
            print("         No beneficial single flip — searching pairs...")

        active_wids = candidate_wids if candidate_wids else set(all_wids)

        active_list = list(active_wids)
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
                net.flip_weight(wb)
                net.flip_weight(wa)
                if flipped_out != orig_out:
                    pair_candidates.append((wa, wb))

        if pair_candidates:
            pair_scores = {pair: _score_pair_flip(net, train_data, pair[0], pair[1])
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
        perturb = random.sample(safe, n_perturb)
        for w in perturb:
            net.flip_weight(w)
        if verbose:
            print(f"  ↳ Random perturbation [{stuck_counter}/{MAX_STUCK}]: "
                  f"flipped {len(perturb)} safe weight(s)")
            for w in perturb:
                _log_flip(net, w, None)

    else:
        if verbose:
            print(f"\n  ✗ Did not converge within {max_steps} steps.")
    return net


def _log_flip(net, wid, score):
    """Pretty‑print a weight flip."""
    kind = wid[0]
    if kind == 'layer':
        _, k, i, j = wid
        new_val = net.layers[k].W[i][j]
        action = "ADD" if new_val else "REMOVE"
        prev_n = net.n_inputs if k == 0 else net.layer_widths[k-1]
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
) -> BooleanNetwork:
    best_net = None
    best_loss = float('inf')
    for r in range(1, n_restarts + 1):
        seed = random.randint(0, 2**31 - 1)
        if verbose:
            print(f"\n{'#'*64}")
            print(f"  RESTART {r}/{n_restarts}  (seed={seed})")
            print(f"{'#'*64}")
        net = BooleanNetwork(n_inputs=n_inputs, layer_widths=layer_widths, seed=seed)
        train(net, train_data, max_steps=max_steps, verbose=verbose)
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