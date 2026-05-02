# Boolean Neural Network
### A from-scratch neural network where every weight, neuron, and operation is Boolean

---

## What This Project Is

This project builds a **neural network entirely out of Boolean logic** — from first principles, from scratch, motivated by a single question:

> *Can Boolean algebra be differentiated?*

The answer is yes — not in the calculus sense, but through the **Boolean derivative**, a well-defined operator that tells you whether flipping a variable changes a function's output. From that single insight, we reconstructed the entire neural network training pipeline:

| Neural Network concept | Boolean equivalent we built |
|---|---|
| Real-valued weight `w ∈ ℝ` | Binary weight `w ∈ {0, 1}` |
| Weighted sum of inputs | AND of selected inputs |
| Activation function (ReLU, sigmoid) | Already binary — no activation needed |
| Dense layer (n → L) | AND-layer: weight matrix `W` of shape `(n × L)` |
| Negative weights | Complement augmentation `[x, ¬x]` |
| Gradient `∂L/∂w` | Boolean derivative `∂f/∂w ∈ {0, 1}` |
| SGD weight update | Flip the best weight |
| Hidden layers | Stack AND-layers |
| Train/test split | Partial truth table observation |
| Generalization | Accuracy on unseen input combinations |

The result is a system that learns unknown Boolean functions from labeled examples, using the same conceptual loop as a real neural network: **forward pass → measure loss → compute derivatives → update parameters → repeat.**

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

### The backpropagation analogy

If a Boolean function produces the wrong output, the Boolean derivative tells us which inputs, if flipped, would fix it — exactly like backpropagation in a neural network tells us which weights to adjust:

```
Neural network backprop          Boolean derivative
────────────────────────         ────────────────────────────────
∂Loss/∂w  is large      →        ∂f/∂w = 1  (this weight matters here)
∂Loss/∂w  is zero       →        ∂f/∂w = 0  (this weight is irrelevant here)
Update: w -= lr × grad  →        Update: flip w (no learning rate needed)
```

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

## Current State of the Code

### Files

| File | Description |
|---|---|
| `bool_net.py` | Full multi-layer Boolean neural network implementation |
| `boolean_learner.py` | Earlier single-layer version with partial data support |
| `boolean_learner.html` | Interactive visual demo of the single-layer learner |

### `bool_net.py` — Architecture

```
BooleanLayer
  W: list[list[int]]   shape (input_width × output_width)
  forward(x)           → AND-neuron evaluation
  flip(i, j)           → single weight update

BooleanNetwork
  layers: list[BooleanLayer]
  out_w:  list[int]           output gate weights
  forward(x)                  → full forward pass
  derivative_layer(x, k, i, j) → ∂f/∂W[k][i][j]
  derivative_out(x, j)         → ∂f/∂out_w[j]
  all_weight_ids()             → iterator over all learnable weights
  flip_weight(wid)             → apply one update

train(net, train_data, ...)    → training loop
evaluate(net, train_data, test_data, ...) → accuracy report
make_dataset(n, fn, frac, ...) → train/test split
```

### Usage

```python
from bool_net import *

# Define your target function
def my_fn(x):
    return x[0] ^ x[1]   # XOR

# Split the truth table
train_data, test_data = make_dataset(n=2, target_fn=my_fn, observed_fraction=0.75)

# Build the network  (2 inputs, two hidden layers of width 4)
net = BooleanNetwork(n_inputs=2, layer_widths=[4, 4], seed=42)

# Train
train(net, train_data, max_steps=500, verbose=True)

# Evaluate
evaluate(net, train_data, test_data, verbose=True)
```

---

## Known Problems

### 1. Dead and contradictory neurons

The random initializer creates neurons like `x₀ ∧ ¬x₀` — a contradiction that is always `0` regardless of input. These are "dead neurons": they contribute nothing, waste capacity, and block the Boolean derivative from ever being `1` at their position.

In a real neural network this is called the **dying ReLU problem**: a neuron whose output is always `0` receives no gradient and never updates.

**Why it happens:** the random `{0,1}` initialization does not check for contradictions between a variable and its complement in the same column. When both `x₀` and `¬x₀` are required by the same neuron, the neuron can never fire.

**Status:** Known, not yet fixed.

---

### 2. Greedy search gets stuck

The training loop picks the single best weight flip at each step (the highest-scoring candidate by global error reduction). When no single flip improves the total training loss, it stops — even if two co-dependent flips together would make progress.

This is the **saddle point / local minimum problem** from neural network optimization. Greedy coordinate descent (one weight at a time) is the Boolean equivalent of SGD with very small batch size and no momentum — it gets trapped.

**Status:** Known, not yet fixed.

---

### 3. XOR still fails at 2 layers

The architecture is theoretically capable of representing XOR — a 2-layer Boolean circuit can express any Boolean function. But in practice, the combination of dead neurons and greedy search prevents convergence for XOR with the current implementation.

This is analogous to a 2-layer neural network that provably can learn XOR, but failing in practice due to poor initialization and optimizer choice.

**Status:** Known, blocked by problems 1 and 2.

---

### 4. Single output only

The current network produces exactly one output bit. Real classification problems often need multiple outputs (multi-class classification) or a richer final layer.

**Status:** Not yet designed or implemented.

---

### 5. No regularization

With partial data, the network overfits to the training rows. It has no mechanism to prefer simpler functions (fewer active terms) over complex ones when both fit the training data equally well.

In real neural networks this is addressed with L1/L2 weight regularization, dropout, or early stopping. Boolean equivalents exist but are not implemented.

**Status:** Not yet designed or implemented.

---

## Future Plans and Ideas

### Immediate fixes (next steps)

#### Fix 1: Smarter initialization
Before a layer is initialized, check every column for contradictions — if both `xᵢ` and `¬xᵢ` appear in the same AND-mask, the neuron is dead by construction. The fix is to ensure no column contains a variable and its complement simultaneously.

```python
# Pseudocode for contradiction-free initialization
for each neuron j:
    for each original input i:
        if W[i][j] == 1:
            W[i + n][j] = 0   # force ¬xᵢ to be excluded
```

This directly solves the dead neuron problem and is the Boolean equivalent of careful weight initialization (Xavier/He initialization in real networks).

#### Fix 2: Multi-flip search / restarts
When the greedy search gets stuck (no single flip helps), instead of stopping:
1. Try all **pairs** of flips (two co-dependent changes)
2. Or: randomly reinitialize the worst-performing neurons and retry
3. Or: implement a simple **random restart** — re-initialize the whole network with a different seed and keep the best result

This is analogous to SGD with momentum, Adam optimizer, or random restarts in neural network training.

---

### Medium-term: multiple outputs

Extend the final layer from a single OR gate to a vector of OR gates — one per output class. This would allow the network to solve multi-label or multi-class problems:

```
Layer k output  →  [out_w₀, out_w₁, ..., out_wₘ]  →  [f₀, f₁, ..., fₘ]
```

Each `fᵢ` is one output bit, trained independently with its own loss. This makes the architecture functionally equivalent to a multi-output neural network classifier.

---

### Medium-term: Boolean regularization

To combat overfitting on partial data, add a preference for **sparser functions** — fewer active weights. The Boolean equivalent of L1 regularization:

```
Loss = training_errors + λ × (number of active weights)
```

During training, when two candidate flips fix the same number of errors, prefer the one that results in fewer total active weights. This biases the learner toward simpler hypotheses — the Boolean version of Occam's razor, and the mechanism behind good generalization.

---

### Long-term: proper backpropagation through layers

The current derivative computation re-evaluates the entire network twice per weight (one forward pass with the weight at 0, one with it at 1). For a network with `W` total weights and `N` training rows, this is `O(W × N)` forward passes per training step — expensive for large networks.

In a real neural network, backpropagation computes all gradients in a single backward pass using the chain rule, making it `O(W + N)`. A Boolean analog is possible using **Boolean chain rule / influence propagation**:

```
∂f/∂W[0][i][j]  =  (∂f/∂h₀[j])  AND  (∂h₀[j]/∂W[0][i][j])
```

Where `∂f/∂h₀[j]` is the sensitivity of the output to neuron `j`'s value, and can be computed in one backward sweep. This would make training scale to deeper, wider networks.

---

### Long-term: stochastic training

Currently the learner picks the *first* error row and uses it to compute derivatives. A natural extension: pick a **random batch** of error rows, compute derivatives on each, and aggregate — choosing the flip that most consistently has a non-zero derivative across the batch. This is the Boolean analog of **mini-batch SGD** and would reduce the variance of the update signal.

---

### Speculative: connections to existing fields

This project independently reconstructed ideas from several established fields. Exploring these connections could be fruitful:

- **Binary Neural Networks (BNNs)**: Real research area (XNOR-Net, BinaryConnect) that trains neural networks with binary weights using real-valued gradients as a proxy. Our approach is fully Boolean throughout — no real-valued gradient proxy.

- **Boolean circuit complexity**: The theoretical study of how many gates are needed to compute functions. Our layer width `L` directly maps to circuit width. Known results (e.g., XOR requiring depth 2) directly predict our network's failures and successes.

- **Quine-McCluskey algorithm**: A classical method for minimizing Boolean expressions (finding the simplest SOP representation). This is the "ideal" version of what our learner is trying to do — exploring the connection could yield a smarter search strategy.

- **Satisfiability (SAT) solvers**: When the learner gets stuck, the problem of finding a weight configuration that achieves zero training loss is equivalent to a SAT problem. A SAT solver could be used as a fallback when greedy search fails.

- **Probably Approximately Correct (PAC) learning**: A formal framework for analyzing how many training examples are needed to learn a Boolean function with high probability. The theoretical guarantees of PAC learning would directly tell us how much data we need for our learner to generalize.

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
  Gradient ∂L/∂w                    Boolean derivative ∂f/∂w ∈ {0, 1}
  Gradient = 0 → weight irrelevant  Derivative = 0 → weight irrelevant here
  SGD: w -= lr × gradient           Flip: w ^= 1 (no learning rate)
  Mini-batch                        Subset of error rows (not yet implemented)
  Epochs                            Steps (one weight flip per step)

Failure modes
  Dying ReLU                        Dead neuron (x ∧ ¬x contradiction)
  Local minimum / saddle point      Greedy search stuck (no flip helps)
  XOR fails on 1 layer              Same — proven impossible
  Overfitting                       Same — memorizes training rows

Generalization
  Train/test split                  Partial truth table observation
  Regularization (L1/L2)            Sparsity penalty on active weights
  Early stopping                    Stop when test loss stops improving
```

The differences are in the *mechanics*, not the *concepts*. A Boolean network is a neural network with the real number line collapsed to two points.