# Boolean Neural Network
### A from-scratch neural network where every weight, neuron, and operation is Boolean

---

## What This Project Is

This project builds a **neural network entirely out of Boolean logic** — from first principles, from scratch, motivated by a single question:

> *Can Boolean algebra be differentiated?*

The answer is yes — not in the calculus sense, but through the **Boolean derivative**, a well‑defined operator that tells you whether flipping a variable changes a function's output. From that single insight, we reconstructed the entire neural network training pipeline, culminating in a fully Boolean backpropagation algorithm.

| Neural Network concept | Boolean equivalent we built |
|---|---|
| Real-valued weight `w ∈ ℝ` | Binary weight `w ∈ {0, 1}` |
| Weighted sum of inputs | AND of selected inputs |
| Activation function (ReLU, sigmoid) | Already binary — no activation needed |
| Dense layer (n → L) | AND-layer: weight matrix `W` of shape `(n × L)` |
| Negative weights | Complement augmentation `[x, ¬x]` |
| Gradient `∂L/∂w` | Boolean derivative `∂f/∂w ∈ {0, 1}` |
| **Backpropagation** | **Single backward pass using Boolean chain rule** |
| SGD weight update | Flip the best weight (or multiple at once) |
| Hidden layers | Stack AND-layers |
| Train/test split | Partial truth table observation |
| Generalization | Accuracy on unseen input combinations |

The result is a system that learns unknown Boolean functions from labeled examples, using the same conceptual loop as a real neural network: **forward pass → measure loss → compute derivatives via backprop → update parameters → repeat.**

---

## Background: The Key Insight

### Can Boolean functions be differentiated?

Classical calculus differentiation requires a continuous domain — you need to take a limit as a value approaches another infinitesimally. Boolean variables live on `{0, 1}`, so there is no "between." Traditional derivatives are undefined.

But an analog exists: the **Boolean derivative** (also called the Boolean difference).

For a Boolean function `f` and a variable `v`, the Boolean derivative at input `x` is:

```
∂f/∂v  =  f(v=0, x)  XOR  f(v=1, x)
```

The result is `1` if flipping `v` changes the output of `f` at that input, and `0` if `v` is irrelevant there. This is a perfect analog of the gradient: it identifies which variables are *responsible* for the current output.

### The Boolean Chain Rule

While the raw Boolean derivative works for a single weight, a network with many layers needs to compute all derivatives efficiently. The key is the **Boolean chain rule**:

If a signal flows from `x` through an intermediate `g` to final output `f`, then:

```
∂f/∂x  =  ∂f/∂g  AND  ∂g/∂x
```

The logical AND replaces the multiplication of real gradients. In words: *the output flips when `x` flips iff flipping `x` flips `g` AND flipping `g` flips the output.*

This chain rule lets us propagate sensitivity signals backward through the network in a single pass, exactly like standard backpropagation — but using pure logic instead of calculus.

---

## Progress: What We Built, Step by Step

### Step 1 — Single-function debugging with Boolean derivatives

**Problem:** Given `f(x, y, z) = (x AND y) OR (NOT z)`, input `(0, 0, 1)` gives output `0` but should give `1`. Which input should be flipped?

**Solution:** Compute `∂f/∂x`, `∂f/∂y`, `∂f/∂z` at the error point:

```
∂f/∂x = f(x=0) XOR f(x=1) = 0 XOR 0 = 0  →  flipping x does nothing
∂f/∂y = f(y=0) XOR f(y=1) = 0 XOR 0 = 0  →  flipping y does nothing
∂f/∂z = f(z=0) XOR f(z=1) = 1 XOR 0 = 1  →  flip z to fix the output ✓
```

This works because `x` and `y` are both `0`, making the `AND` term dead regardless. Only `z` matters. Exactly like a neural network where some weights are in a saturated region and their gradients are zero.

---

### Step 2 — Learning Boolean functions from a truth table

**Problem:** Given only input/output examples (not the function itself), can the network discover the underlying function?

This is the full neural network scenario: an unknown function generates training data, and the model must learn to replicate it.

**Approach — Sum of Products (SOP) representation:**

The hypothesis is represented as a set of AND-terms ORed together:

```
f(x) = w₀·T₀(x)  OR  w₁·T₁(x)  OR  ...  OR  wₖ·Tₖ(x)
```

where each `Tᵢ` is one candidate AND-term (e.g., `x0∧x2`, `x1`, `x0∧x1∧x3`) and each weight `wᵢ ∈ {0,1}` controls whether that term is included in the function.

**Training loop (mirrors neural network training exactly):**

```
1. Forward pass  — evaluate f on all training rows
2. Find errors   — rows where f(x) ≠ target(x)
3. Bool deriv    — ∂f/∂wᵢ at the first error point
4. Score         — which weight flip fixes the most errors globally?
5. Update        — flip the best weight
6. Repeat
```

**Example — learning majority(x, y, z):**

Starting from `f = 0` (all weights zero), the learner:
- Step 1: adds term `x0∧x1` (fixes 2 errors)
- Step 2: adds term `x1∧x2` (fixes 1 error)
- Result: `f = x0∧x1 OR x1∧x2` — partially correct, then hits a local minimum

**Problems discovered (same as real NNs):**
- **Greedy local minima**: a locally good step can trap the learner
- **XOR/parity failure**: SOP with one OR-of-AND layer is equivalent to a single-layer perceptron — it provably cannot represent XOR

---

### Step 3 — Partial data: train/test split

**Problem:** The previous model trained on all `2^n` truth table rows — perfect, complete knowledge. Real-world data is never complete.

**Solution:** Added a `make_dataset()` function that:
1. Takes the full truth table
2. Shuffles it randomly
3. Splits it into **train** (observed) and **test** (held-out) sets
4. Trains only on the train set
5. Reports both train accuracy and test accuracy separately

**Key observation — the same phenomena as real neural networks appeared immediately:**

```
Fraction of data  Train acc  Test acc  What happened
─────────────────────────────────────────────────────────────
100%              88%        N/A       Stuck in local minimum
 75%             100%        50%       Overfit — memorized training rows
 50%             100%        50%       Overfit more aggressively
 25%             100%        67%       Simpler hypothesis → better generalization
```

The 25% run actually *generalized better* than the 50% run — with fewer examples to memorize, it was forced into a simpler, more general hypothesis. This is the **bias-variance tradeoff** appearing naturally.

---

### Step 4 — Redesigning the parameter representation

**The scaling problem with SOP:**

The original model pre-enumerated all `2^n − 1` AND-terms as candidates, with one weight per term:

```
n=3  →      7 parameters
n=10 →  1,023 parameters
n=20 →  1,048,575 parameters
n=100 → ~10³⁰ parameters  (completely intractable)
```

This is doubly exponential and breaks for any real-world input size.

**The solution: weight matrix representation**

Instead of enumerating all possible AND-terms, each neuron gets a **binary mask over the inputs** that it learns:

```
Weight matrix W of shape (input_width × L)

           neuron₀  neuron₁  neuron₂  neuron₃
    x₀  [    1        0        1        0    ]
    x₁  [    0        1        1        0    ]
    x₂  [    0        1        0        1    ]

Neuron 0 computes: x₀             (mask 1,0,0)
Neuron 1 computes: x₁ ∧ x₂        (mask 0,1,1)
Neuron 2 computes: x₀ ∧ x₁        (mask 1,1,0)
Neuron 3 computes: x₂             (mask 0,0,1)
```

Parameter count is now **O(n × L)** — linear in both inputs and layer width:

```
n=3,   L=4  →   12 parameters
n=10,  L=8  →   80 parameters
n=100, L=8  →  800 parameters
```

`L` becomes a **hyperparameter** analogous to layer width in a neural network.

**The derivative now targets individual matrix entries:**

```
∂f/∂W[i][j]  =  f(W[i][j]=0, x)  XOR  f(W[i][j]=1, x)
```

- Flipping `W[i][j]` from `0→1` makes neuron `j` stricter (requires input `i`)
- Flipping `W[i][j]` from `1→0` makes neuron `j` looser (stops requiring input `i`)

---

### Step 5 — Complement augmentation (enabling negation)

**The problem:** AND-neurons with binary weights can only enforce "input must be 1." They cannot express "input must be 0." This means they can't represent any function involving negation — which rules out XOR, NAND, and many others.

**The solution — complement augmentation:**

Before feeding data into any layer, append the bitwise complement of every input:

```
x = (1, 0, 1)  →  augment(x) = (1, 0, 1, 0, 1, 0)
                                  originals   complements
```

The augmented input has width `2n` instead of `n`. Now a neuron can enforce "x₁ must be 0" by connecting to `¬x₁` (the complement) instead of `x₁`. This is exactly equivalent to allowing **negative weights** in a real neural network.

Augmentation is applied:
- At the input (before the first layer)
- Between every pair of hidden layers (so each layer can negate any previous layer's neuron output)

---

### Step 6 — Multi-layer architecture

**Why one layer isn't enough:**

A single AND-OR layer is equivalent to a single-layer perceptron. It can only represent **linearly separable** Boolean functions. XOR and parity are not linearly separable — they fundamentally require at least one hidden layer.

**The full architecture:**

```
Input x  (n bits)
  │
  ▼  augment → [x, ¬x]  (2n bits)
  │
  ▼  Layer 0: W₀  (2n × L₀)  →  h₀  (L₀ bits)
  │
  ▼  augment → [h₀, ¬h₀]  (2·L₀ bits)
  │
  ▼  Layer 1: W₁  (2·L₀ × L₁)  →  h₁  (L₁ bits)
  │
  ▼  augment → ...
  │
  ▼  Layer k: Wₖ  →  hₖ  (Lₖ bits)
  │
  ▼  Output weights  out_w  (Lₖ bits, binary)
  │
  ▼  OR of { hₖ[j]  where  out_w[j] = 1 }
  │
output  (1 bit)
```

The output weight vector `out_w` is a learnable final layer — it selects which last-layer neurons contribute to the final answer. This is equivalent to a dense output layer of width 1 in a real network.

**Everything in code maps 1:1 to neural network concepts:**

```python
# Forward pass: augment → AND-layer → augment → AND-layer → OR
h = augment(x)
for k, layer in enumerate(self.layers):
    h = layer.forward(h)          # AND-neurons
    if k < len(self.layers) - 1:
        h = augment(h)            # give next layer the ability to negate
return int(any(self.out_w[j] and h[j] for j in range(len(h))))
```

---

### Step 7 — Fixing dead neurons (contradiction‑free initialization)

**The problem:** Random initialization frequently produced neurons like `x₀ ∧ ¬x₀` — a logical contradiction that is always 0. These "dead neurons" waste capacity and produce zero gradient, analogous to the **dying ReLU** problem.

**The fix:** For each neuron, we build its column by randomly choosing for every original signal `x_i` one of the three allowed (non-contradictory) patterns:

- `(0,0)` — ignore `x_i`
- `(1,0)` — require `x_i = 1`
- `(0,1)` — require `x_i = 0`

The only forbidden combination is `(1,1)`. This guarantees every neuron is satisfiable from the start, while maintaining a uniform distribution over all valid masks.

---

### Step 8 — Escaping local minima

The greedy single‑flip training often got stuck in local minima where no one weight change reduces the loss. We introduced several mechanisms to escape:

- **Pair‑flip search:** When stuck, try flipping two weights at once — two co‑dependent changes can open an escape route. A cheap candidate filter (only pairs that flip the output on a single error row) keeps it practical.
- **Random perturbation:** If nothing helps, flip a small random set of safe weights (increasingly many after repeated failures) — analogous to simulated annealing.
- **Random restarts:** Train multiple networks from different seeds and keep the best one — a common trick in non‑convex optimization.

With these additions, the network reliably learned XOR, parity, and other previously unreachable functions.

---

### Step 9 — True Boolean backpropagation (the chain rule in action)

The brute‑force derivative computation (evaluate each weight by two full forward passes) scaled as `O(W × N)`. We replaced it with a **single‑pass backward algorithm** that computes all `∂f/∂w` for an input `x` in time proportional to the network size.

**How it works:**

1. **Forward pass with caching** — stores every layer’s input and output.
2. **Output sensitivity** — for the final OR gate, determine which last‑layer neurons, if toggled, would flip the answer.
3. **Backward propagation through AND‑layers** — for a neuron `y = AND(req)`, the sensitivity to an input `x_i` is `1` iff all other required inputs are `1` (so `y` depends on `x_i`). The sensitivity to a weight `W[i]` is `1` iff all other required inputs are `1` and `x_i = 0` (so flipping the weight would change `y`).
4. **Handling complement augmentation** — flipping a signal `h[p]` toggles both `x_in[p]` and `x_in[p+m]`. For each original signal we simulate this flip by re‑running only the downstream part of the network, which is cheap (linear in the remaining depth).
5. **Aggregation** — accumulate the derivative signals for each weight across all misclassified rows, then flip the weight that fixes the most errors.

This is a **pure Boolean backpropagation** — no floating‑point numbers, no gradient approximations, just logical AND and XOR. The training loop now uses this backprop to collect candidate flips much more efficiently than the old brute‑force scoring.

---

## Current State of the Code

### Files

| File | Description |
|------|-------------|
| `bool_net.py` | Full multi‑layer Boolean neural network with backpropagation |

### Architecture

```
BooleanLayer
  W: list[list[int]]   shape (input_width × output_width)
  forward(x)           → AND-neuron evaluation
  flip(i, j)           → single weight update
  _required_met(...)   → helper for local derivatives
  weight_derivative_local(i,j,x) → ∂neuron_j/∂W[i][j]
  input_derivative_local(i,j,x)  → ∂neuron_j/∂x_in[i]

BooleanNetwork
  layers: list[BooleanLayer]
  out_w:  list[int]           output gate weights
  forward(x)                  → full forward pass
  forward_with_cache(x)       → output + intermediate values
  backward(x)                 → dict {weight_id: derivative} for all weights
  _output_sensitivity(h_last) → which last‑layer neurons affect the output
  _backward_layer(...)        → backprop through one AND‑layer
  _sens_through_augmentation(…) → backprop through complement augmentation

train(net, train_data, ...)    → backprop‑based training with pair/perturbation fallbacks
train_with_restarts(...)       → multiple seeds, keep best
evaluate(...)                   → accuracy report
make_dataset(n, fn, frac, ...) → train/test split
```

### Usage

```python
from bool_net import *

def my_fn(x):
    return x[0] ^ x[1]   # XOR

train_data, test_data = make_dataset(n=2, target_fn=my_fn, observed_fraction=0.75)
net = BooleanNetwork(n_inputs=2, layer_widths=[4, 4], seed=42)
train(net, train_data, max_steps=500, verbose=True)
evaluate(net, train_data, test_data, verbose=True)
```

---

## Known Problems (Updated)

### 1. Dead and contradictory neurons ✅ Fixed

Random initialization no longer creates unsatisfiable neurons. Every neuron is guaranteed to have at least one satisfiable input pattern.

### 2. Greedy search gets stuck ✅ Mitigated

Pair‑flip search and random perturbations provide escape routes from local minima. Combined with random restarts, the optimizer now converges reliably on all tested benchmarks.

### 3. XOR / parity failure ✅ Fixed

With the fixes above plus true backpropagation, the network consistently learns XOR (n=2) and parity (n=3) in two‑layer configurations.

### 4. Single output only 🔧 Planned

The current network predicts a single Boolean value. Multi‑class / multi‑label output requires extending the output layer to a vector of OR gates.

### 5. No regularization 🔧 Planned

The network can overfit on small training sets. Boolean L1/L2 regularization (preferring simpler functions) and early stopping are on the roadmap.

---

## Future Plans and Ideas

### Immediate next steps

#### Multiple outputs
Extend the final layer from a single OR gate to a vector of OR gates — one per output class. This would allow multi‑label Boolean classification.

#### Boolean regularization
Add a preference for sparser functions — fewer active weights. The Boolean equivalent of L1 regularization:

```
Loss = training_errors + λ × (number of active weights)
```

When two candidate flips fix the same number of errors, prefer the one that results in fewer total active weights — the Boolean Occam's razor.

#### Simultaneous weight updates (batch flips)
With backprop we can compute which weights matter for each error row. Aggregating over a minibatch allows us to flip **all** weights that help, not just the single best — akin to full gradient descent.

---

### Medium‑term

#### Stochastic minibatch training
Use random subsets of error rows to compute derivative counts, improving scalability to larger datasets.

#### Scaling with NumPy / C++ / GPU
Move the core forward and backward operations to array‑based or compiled code for larger input sizes and deeper networks. The Boolean operations are highly amenable to bit‑wise vectorisation.

---

### Long‑term

#### Theoretical analysis and comparisons
- Relate the training dynamics to PAC learning bounds and Boolean circuit complexity.
- Compare empirically with the recently published Boolean variation‑based backprop (XNOR‑gate neurons) to highlight the unique properties of the AND‑mask architecture.

#### Deeper optimization strategies
- Explore hybrid methods that combine the current greedy global‑scoring with pure backprop‑style simultaneous flips.
- Investigate whether a SAT solver can serve as a fallback when the learner gets stuck on hard functions.

---

### Speculative: connections to existing fields

This project independently reconstructed ideas from several established fields. Exploring these connections could be fruitful:

- **Binary Neural Networks (BNNs):** Real research area (XNOR-Net, BinaryConnect) that trains neural networks with binary weights using real-valued gradients as a proxy. Our approach is fully Boolean throughout — no real-valued gradient proxy.

- **Boolean circuit complexity:** The theoretical study of how many gates are needed to compute functions. Our layer width `L` directly maps to circuit width. Known results (e.g., XOR requiring depth 2) directly predict our network's failures and successes.

- **Quine-McCluskey algorithm:** A classical method for minimizing Boolean expressions (finding the simplest SOP representation). This is the "ideal" version of what our learner is trying to do — exploring the connection could yield a smarter search strategy.

- **Satisfiability (SAT) solvers:** When the learner gets stuck, the problem of finding a weight configuration that achieves zero training loss is equivalent to a SAT problem. A SAT solver could be used as a fallback when greedy search fails.

- **Probably Approximately Correct (PAC) learning:** A formal framework for analyzing how many training examples are needed to learn a Boolean function with high probability. The theoretical guarantees of PAC learning would directly tell us how much data we need for our learner to generalize.

---

## The Core Insight, Restated

Every concept from neural network training has a clean Boolean analog. The table below summarizes the complete correspondence:

```
Neural Network                      Boolean Network (this project)
────────────────────────────────────────────────────────────────────────────
Architecture
  Real weight w ∈ ℝ                 Binary weight w ∈ {0, 1}
  Weighted sum + activation         AND of selected inputs
  Negative weights                  Complement inputs [x, ¬x]
  Dense layer n → L                 AND-layer: matrix W of shape (n × L)
  Layer width L                     Same — hyperparameter controlling capacity
  Stack layers for depth            Same — stack AND-layers

Training
  Forward pass                      Same — evaluate f on training rows
  Loss (MSE, cross-entropy)         Hamming distance (# of wrong rows)
  Backpropagation                   Boolean chain rule (AND of sensitivities)
  Gradient ∂L/∂w                    Boolean derivative ∂f/∂w ∈ {0, 1}
  Gradient = 0 → weight irrelevant  Derivative = 0 → weight irrelevant here
  SGD: w -= lr × gradient           Flip: w ^= 1 (no learning rate)
  Mini-batch                        Subset of error rows (aggregate counts)
  Momentum / Adam                   Pair‑flip search, random perturbations

Failure modes
  Dying ReLU                        Dead neuron (x ∧ ¬x) — fixed by init
  Local minimum / saddle point      Escaped via pair flips & restarts
  XOR fails on 1 layer              Same — proven impossible
  Overfitting                       Same — regularization planned

Generalization
  Train/test split                  Partial truth table observation
  Regularization (L1/L2)            Sparsity penalty on active weights
  Early stopping                    Stop when test loss stops improving
```

The differences are in the *mechanics*, not the *concepts*. A Boolean network is a neural network with the real number line collapsed to two points — and now it even has its own native backpropagation.
