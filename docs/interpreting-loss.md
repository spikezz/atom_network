# Reading the loss number: why raw loss needs a baseline

When you look at the capacity grid, every cell prints a single `loss`. It is
tempting to read it as an absolute score — "lower is better, 0 is perfect." That
is **almost** right, but a raw loss value is meaningless on its own. To turn it
into a score you can trust, you need two reference points for each task:

- a **floor** — the best loss any model could possibly reach (often *not* 0), and
- a **ceiling** — the loss of a model that learned *nothing* (predicts a constant).

A loss only means "good" or "bad" *relative to those two anchors*. This note
derives both anchors for the two tasks in this repo, shows that the dead cells in
the grid land *exactly* on the ceiling, and gives a practical way to read the
numbers.

> TL;DR: within one task the loss is on the **same scale** for every
> architecture (same data + same loss function), so it *is* a valid comparison
> of *this run's fit*. But interpret it against the baselines below — and
> remember that below the "solved" threshold, the differences are optimization
> noise, not architecture quality.

---

## The two anchors, per task

| task | loss fn | **ceiling** = learned nothing (predict a constant) | **floor** = best possible |
|---|---|---|---|
| regression (`y = x₁²+x₂² + 0.1·noise`) | MSE | **3.93** = `var(y)` (predict the mean) | **0.01** = noise variance (irreducible) |
| classification (`label = 1 when x₁·x₂ > 0`) | BCE | **0.69** = `H(p)`, p≈0.52 (predict the base rate) | **≈ 0** (deterministic labels) |

These are not hand-wavy — they are computed from the actual data
(`load_task(task).make_data(1000, 42)`), and the **dead cells in the grid sit
right on the ceiling**:

```
regression dead cells  → loss = 3.929   (var(y) = 3.9286)
classification dead    → loss = 0.692   (H(0.519) = 0.6924)
```

That exact match is the tell-tale sign of a *collapsed* network: it stopped being
a function of its input and just outputs a constant.

---

## Why the ceiling is what it is

A network whose hidden units all die (dying-ReLU collapse) can no longer move its
output with the input — the best it can do is emit a single constant. Training
then drives that constant to whatever value minimises the loss:

**Regression / MSE.** The constant `c` that minimises `mean((y − c)²)` is the mean
`ȳ`, and the resulting error is, by definition, the variance of `y`:

```
min_c  E[(y − c)²]  =  Var(y)
```

For this dataset `x₁, x₂ ~ N(0,1)`, so `x²` has mean 1 and variance 2, giving
`Var(x₁²+x₂²) = 4`, plus a touch of noise → `Var(y) ≈ 3.93`. That is why **every**
dead regression cell parks at 3.93: they are all predicting the mean.

**Classification / BCE.** The constant probability `q` that minimises binary
cross-entropy is the base rate `p = P(y=1)`, and the resulting loss is the binary
entropy `H(p) = −p·ln p − (1−p)·ln(1−p)`. Here `p ≈ 0.519`, so `H(p) ≈ 0.692`
(very close to `ln 2 ≈ 0.693`, the value for a perfectly balanced 50/50 split).
A dead classifier predicting ~0.5 everywhere lands there.

> **Mnemonic:** the ceiling is "the loss of the best *constant* predictor."
> Regression → `var(y)`. Balanced classification → `ln 2 ≈ 0.69`.

---

## Why the floor usually isn't 0

For **regression**, the target carries `0.1·noise`, i.e. an irreducible
random component with variance `0.1² = 0.01`. No model — however large — can
predict noise, so **0.01 is the hard floor** (the "Bayes error" of this dataset).
The best grid cells reach ~0.02–0.05, i.e. they are already brushing up against
the noise floor; the remaining gap is mostly the noise they *cannot* learn, not
"room to improve." Chasing a regression loss below ~0.01 here is chasing noise.

For **classification**, the labels are deterministic (`x₁·x₂ > 0`), so the floor
is ≈ 0. The only irreducible difficulty is an infinitely thin strip along the
axes (where `x₁≈0` or `x₂≈0`) that a smooth sigmoid can't make perfectly sharp —
a measure-zero set. The best cells reach ~0.001–0.004, effectively perfect.

---

## Turning loss into an interpretable score: normalise it

Raw loss is on a per-task scale; to make it human-readable, divide out the
ceiling.

### Regression → R² ("fraction of variance explained")

```
R² = 1 − MSE / Var(y) = 1 − MSE / 3.93
```

| MSE (loss) | R² | what it looks like in the grid |
|---|---|---|
| 3.93 | 0.00 | flat — predicting the mean (**dead**) |
| ~2.0 | ~0.49 | a single ramp/wedge (width-2, depth-1 ceiling) |
| 0.52 | 0.87 | a folded, faceted bowl (width-2 + depth) |
| 0.05 | 0.99 | a smooth bowl (width ≥ 3) |
| 0.02 | 0.99+ | a smooth bowl at the noise floor |
| 0.01 | ~0.997 | the **maximum achievable** R² (noise floor) |

So "loss 0.05" really means "explains 99% of the variance" — which reads as a
*score* far better than the bare number.

### Classification → accuracy (with a caveat)

The intuitive normalised score is **accuracy**, but note **BCE ≠ accuracy**: BCE
also rewards *calibration/confidence*, so two nets with the same accuracy can
have different BCE. As a rough guide here:

| BCE (loss) | meaning |
|---|---|
| 0.69 | ~50% — chance (**dead**) |
| ~0.48 | ~75–80% — one diagonal split solved, the other missed |
| < 0.05 | ~100% — full XOR checkerboard (**solved**) |

---

## The other direction: below "solved", differences are noise

Among already-solved cells you will see losses like 0.001 vs 0.003 vs 0.05. **Do
not over-read the ranking.** Those gaps are dominated by initialization and
optimization randomness, not by the architecture being "better." The same
architecture re-run with a different seed easily reshuffles that order.

The robust way to read the grid is therefore to **bucket**, not to sort by
decimals:

| bucket | regression (R²) | classification (BCE) |
|---|---|---|
| **solved** | R² ≳ 0.98 (loss ≲ 0.08) | BCE ≲ 0.05 |
| **partial** | 0 < R² ≲ 0.9 (ramp / folded) | 0.05 < BCE < 0.6 |
| **failed (dead)** | R² ≈ 0 (loss ≈ 3.93) | BCE ≈ 0.69 |

Everything inside the "solved" bucket should be treated as a tie.

---

## A reading recipe for the grid

1. **Anchor:** is the loss near the ceiling (3.93 / 0.69)? → the cell learned
   nothing; it is *dead*, not merely "a bit worse."
2. **Normalise:** convert to R² (regression) or rough accuracy (classification)
   so the number reads as a score.
3. **Floor-check (regression):** is it near 0.01? → it's as good as the data
   allows; the residual is noise, stop optimizing.
4. **Bucket:** solved / partial / failed. Don't rank decimals inside "solved."
5. **Cross-check the picture:** the "what the network learned" vs "the truth"
   panels confirm the number — a dead cell is a flat colour, a solved cell
   matches the truth.

---

## Two things the loss does *not* tell you

- **Generalisation.** The printed loss is *training* loss (there is no held-out
  split in this educational setup). It happens to track generalisation here only
  because the nets are heavily under-parameterised (≤111 params vs 1000 points)
  and the targets are smooth — so they *can't* memorise. On a real model you
  must hold out a test set; a low training loss can be pure overfitting.
- **The architecture's capability.** A single cell's loss = architecture **+**
  this seed's init **+** this run's optimization luck. The same architecture can
  read 3.93 (dead) on one seed and 0.05 (solved) on another. To score an
  architecture's *capability*, aggregate over seeds (success rate / best-of-N) —
  that is exactly what a seed sweep does, and why it is the right tool for
  "is this architecture able?" rather than "did this run succeed?".

---

*Numbers in this note were computed from `atom/tasks/` with
`load_task(task).make_data(1000, 42)` and confirmed against the dead cells in the
`outputs/*_4_4` grids.*
