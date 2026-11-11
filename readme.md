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

With these additions, the network reliably learned XOR and other previously unreachable functions — though parity turned out to need one more fix before "reliably" was actually true. See Step 10.

---

### Step 9 — True Boolean backpropagation (the chain rule in action)

The brute‑force derivative computation (evaluate each weight by two full forward passes) scaled as `O(W × N)`. We replaced it with a **backward algorithm** that computes all `∂f/∂w` for an input `x` without needing a separate forward pass per weight (see item 4 below for the actual scaling — it's cheaper than brute force, but not quite the strict linear‑in‑network‑size single pass a real backprop achieves).

**How it works:**

1. **Forward pass with caching** — stores every layer’s input and output.
2. **Output sensitivity** — for the final OR gate, determine which last‑layer neurons, if toggled, would flip the answer.
3. **Backward propagation through AND‑layers** — for a neuron `y = AND(req)`, the sensitivity to an input `x_i` is `1` iff all other required inputs are `1` (so `y` depends on `x_i`). The sensitivity to a weight `W[i]` is `1` iff all other required inputs are `1` and `x_i = 0` (so flipping the weight would change `y`).
4. **Handling complement augmentation** — flipping a signal `h[p]` toggles both `x_in[p]` and `x_in[p+m]`. For each original signal we simulate this flip by re‑running only the downstream part of the network — this is *not* strictly a single pass (it's roughly `O(depth)` re‑simulations per layer). *[Correction, Step 12: the `O(depth² × L²)` estimate below undercounted by a factor of `L` — it missed that the resimulation itself is looped over once per original signal in the previous layer. Measured and derived cost is `O(depth² × width³)`, matching the comparison table further down.]*
5. **Candidate filtering** — `backward(x)` on a misclassified row tells us which weights have derivative `1`, i.e. which weights *could possibly* fix that row in isolation. This is only ever used as a cheap filter to narrow down candidates — see Step 10 for why treating it as anything more than a filter (e.g. "the weight with the most votes across error rows") is a bug.

This is a **pure Boolean backpropagation** — no floating‑point numbers, no gradient approximations, just logical AND and XOR. The training loop uses this backprop to narrow down candidate flips instead of brute‑force scoring every weight in the network.

---

### Step 10 — A scoring bug that masqueraded as convergence

Step 9 gave us `backward(x)`, which is exact — verified against brute force (flip a weight, run `forward()`, compare) across networks up to 3 hidden layers deep, 0 mismatches out of 60,000+ checks. The bug wasn't in the derivative math. It was in how the training loop *used* the derivatives.

**The original selection rule:** for every misclassified row, collect the weights with derivative `1` (via `backward()`), and tally *votes* — how many error rows each weight would fix. Flip whichever weight has the most votes.

**Why that's wrong:** a weight's derivative being `1` on an error row only says the output *would* flip on that row, in isolation. It says nothing about what the same flip does to rows that were already correct. A flip with 3 votes on error rows can simultaneously break 4 previously‑correct rows — a net loss *increase* that the vote count reports as a "good" move.

**What this caused in practice:** an infinite 2‑cycle. One run (parity, `n=3`, layers `[6,4]`) got trapped flipping the exact same weight back and forth forever:

```
Step 25 | Loss 4 → flip layer(1,1,1), "fixes 3 errors" → actually breaks 2 correct rows → Loss 3
Step 26 | Loss 3 → flip layer(1,1,1) back, "fixes 2 errors" → actually breaks 3 correct rows → Loss 4
Step 27 | Loss 4 → same flip as step 25 → Loss 3
  ... repeats indefinitely
```

Worse: because the vote‑counting rule always found a "positive" flip in this cycle, `stuck_counter` reset to `0` every step. The pair‑flip / perturbation / restart mechanisms from Step 8 are correctly implemented and do score by true loss — but they never got triggered here, because the loop never registered as "stuck." The escape hatches worked; they just never opened.

This directly undercut the "✅ Fixed" claim this document previously made about XOR/parity. Concretely, running the documented setup (`train_with_restarts`, 10 restarts × 500 steps) on parity (`n=3`, `[6,4]`) across 10 random seeds:

```
Before fix:  2/10 seeds converged to zero training loss
```

Not "consistently learns," despite the claim.

**The fix:** keep `backward()`'s output as a cheap *candidate filter* (Step 9, item 5), but score each candidate by its actual effect on total training loss — flip, `compute_loss()`, unflip — exactly the same principle the pair‑flip search (Step 8) already used correctly:

```python
def _score_flip(net, data, wid):
    old = compute_loss(net, data)
    net.flip_weight(wid)
    new = compute_loss(net, data)
    net.flip_weight(wid)
    return old - new   # positive = flipping this weight actually reduces loss
```

The single‑flip step now flips the candidate with the best score, only if that score is positive.

**Verification after the fix**, 20 fresh random seeds per function, same `train_with_restarts` settings:

```
Function                    Before      After
──────────────────────────────────────────────
Parity (n=3), layers [6,4]   2/10       20/20
Parity (n=3), layers [4]      —         20/20   (was silently stuck at 62% train acc)
XOR (n=2),   layers [4,4]   10/10       20/20
Majority (n=3), [4,4]         —         20/20
```

Training also got *faster* in wall‑clock time on the standard benchmarks (e.g. parity `[6,4]`: 4.08s → 0.34s) despite each step now doing more work per candidate — because no steps are wasted looping in a cycle anymore.

**One side effect worth noting:** now that training actually reaches zero loss reliably, overfitting on small held‑out splits is more visible, not less. Majority (`n=5`, 75/25 split) went from 100% train / 50% test to 100% train / 12.5% test — the model has more room to memorize now that it isn't stuck. This is expected given Known Problem #5 below (no regularization yet), not a new issue.

---

### Step 11 — Removing dead, misleading sensitivity code

Step 10 fixed the scoring bug but left something noted as a loose end: `_backward_layer` computed a `sens_in` value by OR‑combining `input_derivative_local(i, j, ...)` across every neuron `j` sharing input wire `i`. That combination rule is only valid when wire `i` feeds a single downstream neuron. When one wire fans out to several neurons whose outputs later interact — e.g. get XORed together further downstream — OR‑combining their individual single‑flip sensitivities can give the wrong answer: flipping the wire can leave the true final output unchanged even though each path looks sensitive in isolation. (This is the same *reconvergent fanout* problem classical digital‑circuit sensitivity analysis has always had to deal with — see the comparison section below.)

The reason this never caused a visible bug: `sens_in` was computed and passed to `_sens_through_augmentation`, which never actually read it — that function recomputes sensitivity by direct resimulation instead, which sidesteps the problem entirely by being exact rather than compositional. So the invalid computation was dead code. It just happened to be dead code sitting one accidental refactor away from becoming a real bug, and the comment on `_sens_through_augmentation` claimed it "uses cached forward values to avoid recomputing from scratch," which was also inaccurate — it does the opposite, and that's the correct choice, just misdescribed.

**The fix:** removed the `sens_in` computation from `_backward_layer` entirely (it returns just `weight_derivs` now), dropped the now‑unused `raw_output` and `cache` parameters that had been threaded through `_backward_layer` and `_sens_through_augmentation` without ever being read, and rewrote the comments to accurately describe what the resimulation step does and why. `input_derivative_local` (the correct, single‑neuron‑local primitive itself) is kept as a building block, now documented as currently unused rather than silently orphaned.

Re‑ran both verification suites after the change: the brute‑force weight‑derivative check (0/60,000+ mismatches, unchanged) and the 20‑seed convergence stress test (20/20 across parity, XOR, majority, unchanged). Pure cleanup, no behavior change — the value of doing it is in preventing a future bug, not fixing a present one.

---

### Step 12 — Fresh numbers, a verified complexity claim, and the `[4,4]` question

With seeding now fully reproducible (Step 10) and no known correctness bugs left open (Step 11), three things were worth nailing down with actual data rather than restating prior claims. See `test.py` for the full harness; all numbers below come from running it.

**1. Reran the headline benchmarks with more seeds.** 30 fresh seeds each, `train_with_restarts`, full truth table, same architectures already cited in this document:

```
Function                    Result
─────────────────────────────────────
XOR (n=2),   layers [4,4]   30/30 seeds → zero training loss
Parity (n=3), layers [4]    30/30 seeds → zero training loss
Parity (n=3), layers [6,4]  30/30 seeds → zero training loss
```

Consistent with Step 10's 20/20, at a larger sample. This replaces any earlier "converges reliably" language with an actual, reproducible count — rerun `test.py` yourself and you'll get the same numbers (same seeds).

**2. The complexity claim needed correcting — in the code, not just here.** `bool_net.py`'s module docstring still described training as computing every derivative "in a single backward pass per input sample," language that reads as *proportional to network size*, the way real backprop is. That was never true after Step 9 introduced resimulation, and this document's own comparison table below already said `O(depth² × width³)` — but that number had never actually been measured, only derived. It's now been checked empirically: timing `backward()` while sweeping width (8→64, depth fixed) and depth (4→24, width fixed) and fitting a log‑log slope gives measured exponents in the high‑2s for width and high‑1s for depth (2.6–2.9 and 1.7–1.9 across two independent machines), both climbing toward 3 and 2 respectively as the swept dimension grows (the fitted slope starts below the asymptotic value at small sizes, because `Σ(depth−k) ≈ depth²/2 − depth/2` still has a non‑negligible linear term there — see `test.py` section 2 for the full sweep). Unlike the convergence counts above, these are wall‑clock timings, not seeded computations — expect the exact decimals to vary by machine; the trend toward `width³`/`depth²` is what should reproduce. That confirms `O(depth² × width³)`, not a linear‑in‑size single pass. The module docstring has been rewritten accordingly.

**3. Is `[4,4]` genuinely too small to represent 4‑input parity exactly?** No — it's representable, it just has a lower per‑restart success rate than giving it more capacity or more restart budget:

```
Config                        Per-restart success rate (20 independent trials, full data)
────────────────────────────────────────────────────────────────────────────────────────
[4, 4]      (baseline)         5/20
[6, 4]      (wider layer 0)    4/20
[8, 8]      (much wider)      10/20
[4, 4, 4]   (extra depth)      4/20
```

`[4,4]` isn't an outlier next to `[6,4]` or `[4,4,4]` — extra depth alone doesn't help here, and extra width helps only when it's substantial (`[8,8]`, not `[6,4]`). That pattern — succeeds sometimes, comparable to other modest configs, meaningfully improved only by a much wider net — is the signature of a harder search landscape, not an unrepresentable function. The direct test: does the existing `train_with_restarts` machinery already handle this at its current default?

```
train_with_restarts(..., layer_widths=[4,4], n_restarts=10)   13/15 seeds → zero loss
train_with_restarts(..., layer_widths=[4,4], n_restarts=20)   15/15 seeds → zero loss
```

So: `[4,4]` is sufficient capacity for exact 4‑input parity. The earlier open question resolves as "needs more restarts," not "needs a different architecture" — raising `n_restarts` from 10 to 20 was enough to close the gap entirely in this sample. Worth noting this is restart‑budget evidence, not a representability proof; a config that fails across every restart tried is still consistent with a very lucky init being required, not strictly impossible.

---

### Step 13 — Boolean regularization (the Occam's razor penalty)

Once training reliably reached zero loss (Step 10), overfitting stopped being hidden: the net would memorize the training rows with roughly 45–50 active weights for a 5‑input function that needs only a handful. The objective is now:

```
objective = training_errors + λ × (number of active weights)
```

`train(..., lam=λ)` and `train_with_restarts(..., lam=λ)` use it in four places:

1. **Flip scoring:** single and pair candidates are scored by the change in the objective, not just in errors.
2. **Best‑state restore:** `train` restores the lowest‑objective state it saw, so a late random perturbation can't leave the net worse than its best.
3. **Final `prune()` pass:** a greedy pass removes any active weight whose removal lowers the objective. With λ < 1, that means "drop every connection that doesn't cost a training error."
4. **Restart selection:** with λ > 0, `train_with_restarts` runs every restart (an equally accurate but simpler net may come later) and keeps the one with the lowest objective.

`lam=0` is the default and behaves exactly as before. This was checked against weight‑level fingerprints of trained networks from the pre‑change code, and it's now pinned by a regression test.

**Results** (`test.py` section 4). Each of 8 seeds uses a different train/test split and a different training seed; `n_restarts=5`.

```
Function / split          λ=0                      λ=0.1                    λ=0.5
                          train  test  |W|         train  test  |W|         train  test  |W|
Majority n=5, 75%         99.5%  56.2%  46.6       100%   59.4%  18.4       92.2%  56.2%  10.2
Majority n=5, 50%         100%   57.0%  45.5       100%   60.9%  11.5       92.2%  60.2%   6.9
AtLeastTwo n=5, 50%       100%   64.1%  51.9       100%   68.0%   6.5       96.9%  66.4%   4.1
custom n=4, 50%           100%   57.8%  25.8       100%   64.1%   6.8       92.2%  64.1%   4.4
custom n=4, 75%           100%   68.8%  23.9       100%   90.6%   8.4       96.9%  71.9%   7.1
```

What this shows:

- **λ=0.1 is a clear improvement in every case.** Networks are 2.5–8× smaller with no loss of training accuracy, and test accuracy goes up 3–22 points. The biggest gain is `custom` at 75%, where the true function is genuinely small.
- **λ=0.5 overshoots.** It starts giving up training rows (92–97%) to get a smaller net, and test accuracy doesn't improve on λ=0.1. That's expected: once whole networks are compared, any λ accepts one extra training error in exchange for more than 1/λ fewer weights.
- **Majority is still near chance (≈60%).** A threshold function has no small description as an AND‑of‑ANDs over this architecture's literals, so "fewest weights" isn't the right bias for it. Fewer weights helps, but it isn't enough on its own.
- **Caveats:** the test sets are small (8–16 rows), so these are trends, not tight estimates. λ > 0 also costs 2–10× more wall‑clock time, because every restart runs and the pruning pass re‑checks the whole dataset for each weight. Bit‑packing (Future Plans #1) is the natural fix for that.

Parity is deliberately left out of this comparison. Every input that differs from a held‑out parity row by one bit has the opposite label, so any simplicity bias predicts it wrong: 4‑input parity on a 75% split gets 100% train / 0% test, with or without λ. That's the right answer for an Occam learner, not a bug.

---

### Step 14 — Bit‑packing the dataset (forward *and* backward)

Profiling showed about 87% of training time inside `compute_loss`. Every candidate flip is scored against the full dataset, and that meant one Python `forward()` per row. Now:

- **`PackedData`** stores the dataset column‑wise: input *i* becomes one integer whose bit *r* is that input's value on row *r*, and the labels are stored the same way. It's built once per `train()` call.
- **`forward_packed`:** an AND‑neuron's output over the whole dataset is `&` of its required inputs' masks. That's one big‑int operation per connection, however many rows there are. The loss is `popcount(output ^ labels)`.
- **`backward_packed`** runs the backward pass for every error row at once. This stays exact because every step of `backward()` (local AND derivatives, and re‑simulating a flipped signal through the downstream layers) acts on each row independently, so it can act on all rows' bits together. Each weight gets a mask of the rows where its derivative is 1, and its vote count is `popcount(mask)`. The pair search reuses these masks instead of re‑running backward on every error row.

**No behavior change, verified two ways.**
1. Across 30 trained networks (6 function/architecture/split configs × 5 seeds), the weights match the per‑row implementation bit for bit. That includes tie‑breaks: votes are inserted in the order the old per‑row loop used (first error row where the derivative is 1, then canonical weight order).
2. New tests compare `forward_packed`, `compute_loss` and `backward_packed` against per‑row `forward()`/`backward()` and brute force on random networks, including 128‑row datasets, so the masks are wider than a machine word. The per‑row functions stay in the code as the reference.

**Speed:** 3 seeds × `train_with_restarts(n_restarts=5)`, wall‑clock on one machine.

```
Case                    λ     per-row (before)   bit-packed   speedup
Majority n=5, 75%       0          12.37s           6.93s       1.8×
AtLeastTwo n=5, 50%     0           0.39s           0.15s       2.6×
custom n=4, 75%         0           0.45s           0.04s      11×
Parity n=4, full        0          26.84s          10.61s       2.5×
Majority n=5, 75%       0.1        31.86s           2.18s      14.6×
AtLeastTwo n=5, 50%     0.1         7.38s           0.78s       9.5×
custom n=4, 75%         0.1         1.68s           0.43s       3.9×
Parity n=4, full        0.1        41.88s           9.99s       4.2×
```

The λ>0 runs gain the most, because pruning is almost entirely loss evaluations. That was Step 13's cost caveat, now mostly paid off.

**Where the remaining time goes.** In the λ=0 majority case, about 80% is now the pair‑search pre‑filter. For each of O(active²) candidate pairs it flips both weights, runs a per‑row `forward()` on the first error row, and calls a full‑network `has_contradiction()`. Bit‑packing doesn't touch that loop. The obvious fixes are a single‑row `forward_packed`, a contradiction check limited to the two flipped neurons, and incremental re‑evaluation of only the neurons a flip affects. Those are the next target. The `O(depth² × width³)` re‑simulation inside backward is still there too; it's now paid once per step instead of once per error row.

---

### Step 15 — A smoothness regularizer, an ensemble, and two baselines (mostly a negative result)

Step 13's weight‑count penalty helped, but it is a crude idea of "simple". This step tested two alternatives, and tuned and confirmed them on separate seeds.

**1. Total influence (`mu`).** `I(f) = E_x[ #{i : f(x) ≠ f(x⊕eᵢ)} ]` is the expected number of input bits whose flip changes the output. It is the Boolean derivative `∂f/∂xᵢ` averaged over inputs and summed over variables: the Boolean version of a smoothness or Lipschitz penalty. It is measured on *all* inputs, not just training rows, so it constrains the network's behavior on unseen rows. Bit‑packed, flipping `xᵢ` on every row at once is just swapping the `xᵢ`/`¬xᵢ` masks, so it costs `n+1` packed forward passes. `train(..., mu=μ)` adds `μ × I(f)` to the objective and finishes with `polish()`, a greedy pass over all safe single flips (additions as well as removals).

**2. Ensemble.** `train_ensemble()` returns the majority vote of 9 independently trained networks.

**3. Live‑weight counting was checked and dropped.** After λ=0.1 training, 30/30 networks had zero weights outside the output's cone of influence: pruning a dead weight never costs an error, so `prune()` already removes them all. Counting only live weights would change nothing.

**Method.** Configs were tuned on seeds 100–119, then the chosen ones were run on fresh seeds 200–219 (new splits and new training seeds). Only the fresh‑seed numbers are quoted below. μ = 0.25, 0.5 and 1.0 gave identical results on the tuning seeds, so μ=0.5 stands in for all of them. A one‑flip change in influence is far too small to outweigh a training error, so in practice μ only acts as a tie‑breaker among networks that fit the data equally well.

**Results** (test accuracy, mean ± standard error over 20 seeds; `test.py` section 5):

```
                         majority  nearest   none   λ=0.1   μ=0.5   λ=0.1   ensemble(9)
                         label     neighbour                        +μ=0.5  λ=0.1
Majority n=5, 75%         41.2      49.4     52.5   55.0    49.4    51.9    53.8   (±2–3)
Majority n=5, 50%         44.1      58.8     55.9   60.0    53.1    58.1    59.7   (±1–2)
AtLeastTwo n=5, 50%       82.5      77.5     72.5   71.6    83.1    75.0    71.2   (±1–2)
AtLeastThree n=6, 50%     65.8      69.4     65.9   70.5    68.6    68.6    69.1   (±1–1.6)
custom n=4, 50%           48.8      63.1     55.0   71.2    53.1    61.9    65.0   (±3–4.5)
custom n=4, 75%           55.0      66.2     62.5   87.5    66.2    92.5    87.5   (±4–6)
```

What this shows:

- **The influence penalty is essentially a "predict the common label" bias.** Its one clear win (AtLeastTwo, 72→83%) only matches the trivial predict‑the‑majority‑label baseline (82.5%). That function is 1 on 26 of 32 inputs. Minimizing influence over the whole domain is cheapest when the unseen region is constant, because a constant region contributes no influence except at its boundary with the training rows. On balanced targets (majority, custom) it pushes learned influence further *below* the target's (e.g. 1.36 vs 1.88) and hurts test accuracy. That makes it the wrong notion of smoothness here, and **not recommended**. The code stays in, tested, as a documented negative result.
- **λ=0.1 remains the best general default.** It is best or tied for best on 5/6 cases, and it beats nearest neighbour where the target has a small description (custom: 71 vs 63, 87.5 vs 66). λ=0.1+μ=0.5 scored 92.5% on custom 75%, but that is within one standard error of λ=0.1 and worse everywhere else.
- **The ensemble adds nothing** over a single λ=0.1 network on these sizes. It is kept as a cheap tool.
- **Nobody learns majority from 16–24 rows.** Every method, including nearest neighbour, is near chance. The majority‑label baseline is even *below* 50%, because when a balanced function is split, the held‑out rows lean toward the label that is rarer in training. Some targets just don't have enough data here.
- **AND/OR circuits don't beat nearest neighbour across the board.** On AtLeastTwo, both trivial baselines beat every circuit learner. This is the honest bar future changes have to clear.

---

### Step 16 — A real dataset: the MONK's problems

Toy truth tables can only say so much. The MONK's problems (Thrun et al., 1991; UCI ML Repository, CC BY 4.0) are the classic benchmark for exactly this kind of learner. There are 6 categorical attributes (432 possible inputs), each problem has a fixed official train/test split, and the test set is the entire domain. `datasets.py` one‑hot encodes the attributes into **17 Boolean inputs** (34 after complement augmentation), far more than anything trained so far. A test checks the loaded labels against the documented target concepts:

```
MONK-1: (a1 = a2) or (a5 = 1)                          → a 4-term DNF over the one-hot bits
MONK-2: exactly two of {a1..a6} equal 1                → threshold-like, like majority
MONK-3: (a5 = 3 and a4 = 1) or (a5 ≠ 4 and a2 ≠ 3)     → 2-term DNF, 5% label noise in train
```

The baselines are in `baselines.py`, with no sklearn dependency. The decision tree is an ID3‑style binary tree over the one‑hot bits; it is not the multiway ID3 used in the original 1991 comparison, so published ID3 figures don't transfer.

**Results** (`test.py` section 6). BoolNet uses `train_with_restarts(n_restarts=5)` over 10 training seeds and shows mean ± standard error (and the best seed). The baselines are deterministic.

```
                      majority  nearest   decision   BoolNet [8,4]              BoolNet [12,6]
                      label     neighbour tree       λ=0          λ=0.1         λ=0.1
MONK-1        train    50.0     100       100        79.8         92.7          93.0
              test     50.0      83.8      92.6      76.9 ± 5.8   88.8 ± 2.9    86.5 ± 2.7   (best: 100)
MONK-2        train    62.1     100       100        65.0         75.4          77.4
              test     67.1      72.2      86.6      67.2 ± 0.4   66.5 ± 0.8    67.8 ± 0.8
MONK-3        train    50.8     100       100        80.3         94.7          93.2
(noisy)       test     47.2      87.5      89.8      80.3 ± 7.0   96.6 ± 0.8    93.8 ± 2.3   (best: 100)
```

What this shows:

- **MONK-3 is a genuine win.** At 96.6% ± 0.8, BoolNet beats both the tree (89.8%) and nearest neighbour (87.5%) by a margin well outside the error bar. Its ~95% training accuracy is what you'd hope for with 5% label noise: it mostly *declines to memorize* the 6 noisy rows that the tree and nearest neighbour fit exactly.
- **MONK-1 is limited by the search, not the model.** Individual seeds reach 100% test accuracy, which is the exact concept, but training accuracy averages 93%. The search often fails to fit the training data at all, even though a small network that fits it perfectly exists. The tree's 92.6% is ahead of the mean.
- **MONK-2 fails, like majority did.** BoolNet stays at the 67% you get from always predicting the common label and can't even fit the training data. "Exactly two of six" has no compact AND‑of‑ANDs form.
- **Wider is worse and 6–8× slower.** `[12,6]` searches a bigger space and does no better than `[8,4]` on any problem.

**A fix along the way: `train()` now always returns its best state.** Previously, with λ=0, a run that never reached zero loss returned whatever state it ended in, often right after a random perturbation had made things worse. It now restores the lowest‑loss state it saw. A test pins this down by recording the loss at every step: the old code fails it (seed 2 ended at 18 errors after having reached 16). Runs that reach zero loss were never affected, so every convergence count in this README is unchanged. Re‑running Steps 13 and 15 changed a single cell, by 0.2 points.

**Correcting the first hypothesis.** λ=0.1's big training‑accuracy advantage on MONK's (93% vs 80%) is **not** from that best‑state restore: the restore alone moved λ=0 from 77.4% to 79.8% training. λ=1e‑9 reproduces λ=0.1's numbers exactly, so the weight penalty's size doesn't matter. By elimination, the cause is how flips are scored:

- The restore now applies to both settings.
- Both run every restart, since neither reaches zero loss.
- The final prune can't raise training error.

With λ > 0, a flip that changes no training row's prediction but removes a weight scores +λ instead of 0, so it's accepted as progress. With λ = 0 the search sees no improving flip, counts itself stuck, falls back to random perturbations, and gives up after 5. So on MONK's the penalty matters less as an Occam prior than as **a way for the search to keep moving when the error count is flat**. *(Step 17 tested this directly and found it is only part of the story.)*

**What this points to next:** the search, not the objective. Seeds that reach 100% on MONK-1 exist, but the search finds them unreliably, and bigger networks make it worse. Candidate directions:
- deliberate neutral moves (accept zero‑change flips, with a tabu list to avoid cycling);
- a smarter escape than random perturbation;
- more restarts now that each one is cheap;
- the pair‑search speedup from Step 14, which these 34‑input layers make much more pressing.

---

### Step 17 — Neutral moves, isolated: only part of why λ helps the search

Step 16 concluded, by elimination, that λ > 0 trains better on MONK's because it turns error‑neutral, weight‑removing flips into accepted moves. `train(..., neutral=...)` tests that directly. When no candidate strictly improves the objective, it accepts a neutral one instead of falling through to pair search or random perturbation:

- `"remove"` accepts only neutral removals: exactly the extra move λ > 0 allows.
- `"any"` accepts any neutral flip that isn't in a 20‑entry tabu list of recent flips.

It is off by default, so default behavior is unchanged (verified by fingerprint).

**Results** (`test.py` section 7; MONK's, `[8,4]`, 10 seeds; train accuracy / test accuracy / final active weights):

```
                            MONK-1                 MONK-2                 MONK-3 (noisy)
λ=0                         79.8 / 76.9 /  99      65.0 / 67.2 / 111      80.3 / 80.3 / 106
λ=0 + neutral removals      85.5 / 80.4 /  99      64.9 / 67.2 / 113      87.0 / 88.9 / 104
λ=0 + any neutral move      86.0 / 79.1 / 100      65.0 / 67.1 / 113      87.0 / 88.9 / 104
λ=1e-9                      92.7 / 88.8 /  14      75.4 / 67.0 /  23      94.7 / 96.6 / 12
λ=0.1                       92.7 / 88.8 /  14      75.4 / 66.5 /  22      94.7 / 96.6 / 12
```

What this shows:

- **Neutral moves explain part of the gap, not all of it.** On MONK‑1 and MONK‑3 they recover about half of λ's training‑accuracy gain (+6–7 of +13–14 points). On MONK‑2 they recover none of it. Step 16's by‑elimination conclusion was incomplete.
- **The size column shows the rest.** With neutral moves the networks still end with about 100 active weights; with any λ > 0 they end with 12–23. Neutral removals only fire when the search is stuck. λ instead prefers removals at *every* step, whenever two flips fix the same number of errors, so the network stays small throughout training. **Working hypothesis (not yet isolated): sparse networks are easier to search.**
- **Adding neutral moves to λ > 0 is a no‑op.** Every single flip changes the weight count by ±1, so with λ > 0 no flip ever scores exactly 0, and the rows are identical. A tiny λ already supersedes the neutral‑move option; it stays in the code (tested, off by default) as a documented result.

**Next, if the sparsity hypothesis holds:**
- **Initialize sparser.** Each neuron currently includes each input signal (as `x` or `¬x`) with probability 2/3, so networks start dense, at roughly 100 weights on MONK's.
- **Make λ = 1e‑9 the default.** It costs essentially nothing in fit, and it would be a deliberate behavior change.

---

### Step 18 — Sparse initialization: where the search starts is what matters

Step 17's hypothesis was that sparse networks are easier to search. `BooleanNetwork(..., density=d)` tests it from the other direction: start sparse instead of relying on λ to make the network sparse. `density` is the probability that a neuron uses each input signal (as `x` or `¬x`). The original init corresponds to 2/3 and is still the default (`density=None` keeps its exact random sequence, which a test pins). On MONK's `[8,4]`, the default starts with about 115 active weights; density 0.3 starts with 56, 0.15 with 33, and 0.05 with 20.

**Results** (`test.py` section 8; 10 seeds; test accuracy mean ± standard error, with train accuracy in parentheses):

```
                         MONK-1                  MONK-2                  MONK-3 (noisy)
decision tree            92.6        (100)       86.6        (100)       89.8        (100)
λ=0, default init        76.9 ± 5.8  (79.8)      67.2 ± 0.4  (65.0)      80.3 ± 7.0  (80.3)
λ=0, density 0.3         97.6 ± 1.3  (99.2)      71.8 ± 1.9  (80.0)      96.6 ± 0.3  (95.7)
λ=0, density 0.15        97.1 ± 1.2  (98.5)      76.2 ± 1.2  (85.7)      95.9 ± 0.4  (96.3)
λ=0, density 0.05        98.7 ± 1.0  (99.6)      79.9 ± 1.2  (87.3)      96.0 ± 0.6  (96.1)
λ=1e-9, default init     88.8 ± 2.9  (92.7)      67.0 ± 0.8  (75.4)      96.6 ± 0.8  (94.7)
λ=1e-9, density 0.05     99.0 ± 1.0  (99.6)      79.4 ± 2.1  (88.3)      95.7 ± 0.9  (97.0)
```

What this shows:

- **Sparse initialization is the biggest single improvement so far.** MONK‑1 goes from 77% to about 99% test, clearly beating the decision tree (92.6%), so the network now usually recovers the exact concept. MONK‑2, which looked hopeless (stuck at the 67% of predicting the common label), climbs to about 80%. That beats nearest neighbour (72.2%) but is still behind the tree (86.6%). MONK‑3 holds its ~96%, now without needing λ.
- **The hypothesis was close but not quite right.** With a sparse start, λ=0 ends with 40–90 active weights, yet it trains as well as λ=1e‑9, which ends with about 15. So what matters is *starting* sparse, not *staying* sparse. A plausible reason: a dense AND‑neuron is false on almost every row, so almost no single flip changes its output and backprop finds few useful candidates. A sparse neuron fires often, so the derivative signal is rich. This explanation is untested.
- **λ is now mostly redundant for fitting.** Its training advantage at the default init (Steps 16–17) disappears once the start is sparse. It still makes the final network 3–4× smaller, which makes the learned formula much easier to read.
- **Cost:** sparse MONK‑2 runs are 5–7× slower (150–220 s), because the search keeps making progress instead of giving up early.

**Not yet checked:** whether a sparse start hurts the toy benchmarks. Parity in particular needs neurons with many literals, which a sparse start has to build up one flip at a time. That has to be measured before `density` gets a new default, which would also be a deliberate behavior change.

---

## How This Compares to Standard (Real‑Valued) Backprop

**Why real backprop is a single, cheap pass.** Differentiation is linear: the multivariate chain rule says a total derivative is a *sum* of path contributions. That linearity is what lets standard backprop cache one local Jacobian per layer during the forward pass, then compose them backward with a single matrix multiply per layer — never touching the raw data again. Cost stays proportional to network size, `O(depth × width²)`, same order as the forward pass.

**Why this Boolean version can't do the same trick everywhere.** The Boolean derivative `∂f/∂v = f(v=0) XOR f(v=1)` is a finite difference over a discrete domain, not a linearization, and it does not have an equivalent sum rule. Concretely: if a signal `x` fans out to two neurons `h1` and `h2` that later get XORed together downstream, flipping `x` can leave the final output completely unchanged even though each path, considered alone, looks sensitive. Naively OR‑combining independent path sensitivities gets this wrong. This is exactly the **reconvergent fanout** problem from digital circuit testing — the same reason path‑tracing test‑generation algorithms (e.g. PODEM, the D‑algorithm) can't treat fan‑out paths independently, and no coincidence that the term "Boolean difference" for this exact derivative operator originates in that literature (Sellers, Hsiao & Bearnson, 1968).

Within a single AND‑layer, where each weight belongs to exactly one neuron, no such fan‑out ambiguity exists, so the local chain rule (`sens_out[j] AND local_derivative`) is exact — verified against brute force across networks up to 3 hidden layers deep, 0 mismatches out of 60,000+ checks. But once a neuron's *output* fans out to multiple neurons in the next layer, the shortcut breaks, so `_sens_through_augmentation` deliberately falls back to direct resimulation (flip the neuron, re‑run the forward pass through everything downstream) rather than trying to compose cached sensitivities. Exact, but not free.

**Scaling comparison:**

| | Standard backprop | This implementation |
|---|---|---|
| Cost per backward pass | `O(depth × width²)` | `O(depth² × width³)` — resimulation compounds with depth (derived here; measured and confirmed in Step 12) |
| Why | Cached local Jacobian, composed once per layer | Each earlier layer needs a growing resimulation through everything downstream |
| vs. naive brute force (`O(depth² × width⁴)`, evaluate every weight by two full forward passes) | n/a | Still a real win — saves roughly a factor of `width` |

Three more scaling dimensions worth separating out:
- **Parameter count** scales well — `O(depth × width²)`, linear in the number of inputs `n` (the whole point of the Step 4 redesign away from enumerated SOP terms).
- **Training‑loop cost per step** got *more* expensive per candidate after the Step 10 fix, since scoring now runs `compute_loss` over the full dataset for every candidate weight rather than trusting a vote count — in practice a net win in wall‑clock time (far fewer wasted steps), but cost now also scales with dataset size, not just network size.
- **Constant factors** dominate in practice right now — everything is plain Python loops over lists/tuples, no vectorization. The current demo scale (`n ≤ 5`, widths `≤ 8`, depth `≤ 3`) is comfortably fast; an order of magnitude past that, Python‑loop overhead bites before the algorithmic complexity does.

---

## Current State of the Code

### Files

| File | Description |
|------|-------------|
| `bool_net.py` | Full multi‑layer Boolean neural network with backpropagation |
| `tests/test_bool_net.py` | Fast regression tests (~6s): backward exactness vs brute force, convergence, reproducibility, regularization. Run `python3 -m unittest discover -s tests -v` |
| `test.py` | Slower experiment harness that produces the README's numbers (section 5 uses all CPU cores) |
| `datasets.py` | Loader for the MONK's problems (official splits, one‑hot encoded to 17 bits) |
| `baselines.py` | Reference learners: majority label, Hamming nearest neighbour, ID3‑style decision tree |
| `data/monks/` | MONK's problems as downloaded from the UCI ML Repository (CC BY 4.0) |
| `tests/test_datasets_baselines.py` | Loader checked against the documented target concepts; baseline sanity tests |
| `boolean_learner.py` / `.html` | Earlier single‑layer version |

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
  backward(x)                 → dict {weight_id: derivative} for all weights (one row)
  forward_packed(pd)          → output bitmask over every row of a PackedData at once
  backward_packed(pd, rows)   → dict {weight_id: row-mask of derivative=1} (all rows at once)
  _output_sensitivity(h_last) → which last‑layer neurons affect the output
  _backward_layer(...)        → backprop through one AND‑layer
  _sens_through_augmentation(…) → backprop through complement augmentation

complexity()                → number of active weights (what λ penalizes)
get_state() / set_state(s)  → weight snapshot / restore

PackedData(data)               → dataset as per-input bitmasks (bit r = row r); built once per train()
compute_loss(net, data)        → popcount(output_mask XOR label_mask); accepts dict or PackedData
compute_objective(net, data, lam) → errors + lam × active weights
_score_single_flip(net, data, wid, lam) → true objective delta from a single candidate flip
_score_pair_flip(net, data, a, b, lam)  → true objective delta from a candidate pair flip
prune(net, data, lam)          → greedily remove weights that don't pay for themselves
Objective(data, lam, mu)       → callable net → errors + λ·weights + μ·influence
total_influence(net, pd)       → average sensitivity Σᵢ Pr[f(x) ≠ f(x⊕eᵢ)] (bit-packed)
polish(net, objective)         → greedy over all safe single flips (used when mu > 0)
train_ensemble(...) / Ensemble → majority vote of independently trained nets
train(net, train_data, ...)    → backprop‑filtered candidates, scored by true loss, with pair/perturbation fallbacks
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

# With Boolean regularization (prefer simpler networks — see Step 13):
net = train_with_restarts(n_inputs=2, layer_widths=[4, 4], train_data=train_data,
                          lam=0.1, verbose=False, seed=0)
```

---

## Known Problems (Updated)

### 1. Dead and contradictory neurons ✅ Fixed

Random initialization no longer creates unsatisfiable neurons. Every neuron is guaranteed to have at least one satisfiable input pattern.

### 2. Greedy search gets stuck ✅ Fixed

Pair‑flip search and random perturbations provide escape routes from local minima, combined with random restarts. These were implemented correctly from the start, but a separate bug in the single‑flip selection rule (see Step 10) could mask the "stuck" state with an infinite oscillation, so the escape routes never got invoked when it mattered most. Fixed in Step 10 by scoring single flips the same way pair flips already were: by true effect on total loss, not by an error‑row vote count.

### 3. XOR / parity failure ✅ Fixed, verified

The network learns XOR (n=2) and parity (n=3) in two‑layer configurations. This is backed by actual numbers, not just an architecture argument: 30/30 random seeds converge to zero training loss on the documented benchmark settings, `train_with_restarts`, full truth table (see Step 12; supersedes Step 10's 20/20 at a smaller sample size). Before the Step 10 fix, this same claim was checked and found to only hold 2/10 times for parity — worth remembering as a reminder to verify "should work" claims empirically, not just architecturally.

Parity(n=4) with the small `[4,4]` architecture was an open question (is it representationally too small, or just under‑searched?) — resolved in Step 12: `[4,4]` is representationally sufficient. It has a lower per‑restart success rate than wider configs, but raising `train_with_restarts`' `n_restarts` from 10 to 20 was enough to reach 15/15 seeds converging.

### 4. Single output only 🔧 Planned

The current network predicts a single Boolean value. Multi‑class / multi‑label output requires extending the output layer to a vector of OR gates.

### 5. No regularization ✅ Implemented (Step 13)

`lam` penalizes active weights. At λ=0.1 networks are 2.5–8× smaller with no loss of training accuracy, and test accuracy goes up on every benchmarked function. Generalization on threshold functions like majority is still weak (≈60%), because "fewest weights" isn't the right inductive bias for them.

---

## Future Plans and Ideas

Roughly in priority order — highest expected value for lowest conceptual risk first.

### 1. Bit‑pack the dataset dimension ✅ Done (Step 14)

The one idea here that's specific to Boolean networks in a way real backprop can't exploit: represent each neuron's output *across the entire training set* as a single integer bitmask (or a NumPy bit array), and turn `forward()` for a whole layer into a handful of bitwise AND/OR/XOR operations over those masks instead of a Python loop over rows. This doesn't fix the `O(depth²)` resimulation cost from the comparison section above, but it collapses the dataset‑size dimension out of the complexity almost entirely — probably the biggest win for the least risk, and a natural fit for the "Scaling with NumPy" idea already below, made concrete.

### 2. Cone‑of‑influence pruning

`_sens_through_augmentation` currently resimulates through *every* downstream neuron regardless of whether it actually depends on the flipped signal. Many won't — their AND‑mask never touches that wire or its complement. Skipping neurons outside the cone of influence shrinks the constant factor without changing the algorithm's structure.

### 3. Boolean regularization ✅ Done (Step 13), follow‑ups remain

The basic penalty works (see Step 13). Tried and rejected in Step 15: total‑influence smoothness (it collapses to "predict the common label"), live‑weight counting (no effect: `prune()` already removes dead weights), and majority‑vote ensembles (no gain). Open follow‑ups:
- **Better pruning:** try pair or swap moves in `prune()` (remove two, add one), not only single removals.
- **Choose λ by validation** instead of fixing it by hand.
- **Better inductive biases** for threshold‑like functions, where fewest‑weights alone doesn't help.

### 4. Multiple outputs

Extend the final layer from a single OR gate to a vector of OR gates — one per output class, for multi‑label Boolean classification.

### 5. Simultaneous weight updates (batch flips)

Aggregate over several error rows and flip more than one beneficial weight per step. One caution carried over directly from Step 10: if several weights are flipped together, they need to be scored by their *combined* effect on true loss, not by summing individual single‑flip scores — two individually‑good flips aren't guaranteed to still be good together, for exactly the reason Step 10 exists.

### 6. Stochastic minibatch training

Use random subsets of error rows to compute derivative counts, improving scalability to larger datasets. Pairs naturally with idea #1 (bit‑packing) if the minibatch itself is represented as a bitmask.

### 7. A genuinely O(depth) exact backward pass

The open research‑y question, not a small patch. Would require tracking something richer than a single sensitivity bit per neuron — closer to a compact Boolean‑function summary of how the final output depends on the *set* of neurons affected by a given earlier signal, which starts to resemble BDD‑based (binary decision diagram) methods. Worth flagging as a long‑term direction rather than an immediate fix.

### Also still on the list

- **Theoretical analysis:** relate training dynamics to PAC learning bounds and Boolean circuit complexity; compare empirically with published Boolean‑variation backprop (XNOR‑gate neurons).
- **SAT solver fallback:** when the learner gets stuck on a genuinely hard function, finding a zero‑loss weight configuration is equivalent to a SAT problem — a SAT solver could serve as a last‑resort fallback.

---

### Speculative: connections to existing fields

This project independently reconstructed ideas from several established fields. Exploring these connections could be fruitful:

- **Reconvergent fanout / VLSI circuit testing (ATPG):** The exact problem behind why this project's backward pass can't cheaply compose sensitivities across layer boundaries (see "How This Compares to Standard Backprop" above) is the classical reconvergent‑fanout problem from digital circuit test‑pattern generation — the same reason algorithms like PODEM and the D‑algorithm can't treat fan‑out paths independently. The term "Boolean difference" for `f(v=0) XOR f(v=1)` itself originates in that literature (Sellers, Hsiao & Bearnson, 1968), which we arrived at independently before recognizing the connection. Circuit‑sensitivity and path‑sensitization techniques from that field are a plausible source of ideas for a smarter (possibly sub‑`O(depth²)`) backward pass.

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
  Overfitting                       Same — λ × active weights penalty

Generalization
  Train/test split                  Partial truth table observation
  Regularization (L1/L2)            Sparsity penalty on active weights
  Early stopping                    Stop when test loss stops improving
```

The differences are in the *mechanics*, not the *concepts*. A Boolean network is a neural network with the real number line collapsed to two points — and now it even has its own native backpropagation.