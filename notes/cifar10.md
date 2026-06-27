# CIFAR-10 / ResNet-18 — optimizer tuning notes

ResNet-18 (CIFAR stem: 3×3 stride-1 conv, no maxpool) trained from scratch on
CIFAR-10, 8 GPUs (per-GPU batch 128 → global batch 1024), always-on augmentation
(random crop + horizontal flip) and weight decay. The objective `value` is the
**top-1 test error** (the model returns it as its eval metric); target = 0.05.

Analyse a results parquet with `scripts/investigate_cifar10.py`
(`leaderboard` / `sweep <S>` / `grid <S>` / `traj <S> --lr` / `plateau <S> --lr`).

## Leaderboard (best test error, single repetition)

| Rank | Optimizer | Config | Best err | Notes |
|------|-----------|--------|----------|-------|
| 1 | **Scion** | lr=0.002, cooldown_frac=0.4 | **4.80%** | needs full budget + anneal |
| 2 | SGD | lr=0.8, mom=0.9, nesterov, wd=5e-4 | 4.89% | strong, hit 5% target |
| 3 | Muon | muon_lr=0.01, adam_lr=0.0028 | 5.55% | front-loaded; budget-cheap |
| 4 | Adam | lr=0.0028, wd=0.05 | 5.91% | baseline; overfits |

Gaps at the top (≤0.1pp) are within the ~0.3pp run-to-run noise — a 3-rep
confirmation run is set up in `bench_configs/cifar10_resnet18_8gpu.yaml`.

All matrix-aware optimizers (Scion, Muon) beat Adam, and Scion edges out
well-tuned SGD — the expected outcome for a conv net.

## Adam

- **Best**: lr=0.0028, wd=0.05, 9600 steps, warmdown 4800 → **5.91–5.97%**.
- Converges fastest early (sub-7% by ~step 6500) then plateaus.
- **Overfits hardest**: train loss → 7e-5 (memorized) while test sits ~6%.
- ~0.3pp run-to-run variance (5.69 / 5.97 on repeat runs) — the noise yardstick.
- Parked lever (untested): label smoothing 0.1 to close the generalization gap.

## SGD

- **Best**: lr=0.8, momentum=0.9, nesterov, wd=5e-4 → **4.89%** (hit 5% target,
  stopped at 9500). resnet_classif recipe adapted to the distributed setting
  (trapezoidal schedule, wd on matrix params only).
- lr=0.8 = linear scaling of the canonical lr=0.1 for the 8× global batch
  (Goyal et al.); trained stably with warmup.
- Noisy/high (12–24%) for ~80% of training; **essentially all the gain is in the
  warmdown** (8.6% → 4.89% over the last ~750 steps).
- Generalizes better than Adam despite a higher train loss (5.6e-4) — implicit
  regularization of SGD+momentum.

## Muon

- **Best**: muon_lr=0.01, adam_lr=0.0028 (BN params) → **5.55%**.
- **LR sweep** (gentle U): 0.0005→5.86, 0.002→5.62, **0.01→5.55**, 0.05→5.95.
  Forgiving — well-behaved across ~100×.
- **Front-loaded**: flat phase plateaus by ~step 3250 (~6.8%) and then wastes
  ~37% of training before the cooldown. Shorter schedules nearly match:
  4800 steps → 5.74% at **half the wall-clock** (90s vs 178s), within noise.
- Cooldown contributes a tidy ~1.3pp. Overfits (train loss → 1.7e-5) but test
  never degrades.
- **Takeaway**: Muon's budget is cheap — cut `num_steps` freely; the flat phase
  past the plateau buys almost nothing.

## Scion

- **Best**: lr=0.002, cooldown_frac=0.4 → **4.80%** (champion). radii at GPT
  defaults (hidden=50, lm_head=3000), momentum=0.1.
- **LR is knife-edge**: 0.0005→5.35, **0.002→4.95** (at default cooldown),
  0.01→8.58 (flat phase stuck, train loss ~0.14), 0.05→14.73 (diverges, train
  loss 20–40). Likely coupled to the untuned GPT-2 radii — a candidate for
  further tuning to widen the usable LR band.
- **Schedule grid** (lr=0.002), best_err over (num_steps × cooldown_frac):

  ```
  cooldown   0.20  0.28  0.40  0.50  0.60  0.80
  num_steps
  4800       5.58   --   5.30   --   5.10  5.32
  6400       5.07   --   4.88   --   5.03   --
  9600        --   4.95  4.80  4.85  4.92   --
  ```

- **Needs the full budget — the opposite of Muon.** Best-per-budget improves
  monotonically (4800→5.10, 6400→4.88, 9600→4.80).
- **Controlled experiment** (iso-cooldown = 3840 anneal steps, growing flat):
  4800/0.8→5.32, 6400/0.6→5.03, 9600/0.4→4.80. Same anneal, error falls as the
  flat phase grows → **Scion's long flat exploration is productive, not wasted.**
- Cooldown also matters with an optimum (~2560–3840 steps; too little → 5.58 at
  4800/0.2, too much → slight regression). Both levers help and roughly multiply.
- **Modest compute win**: 6400/0.4 → 4.88% (–33% compute, still beats SGD).
- **Takeaway**: give Scion both a long flat phase and a healthy anneal; don't cut
  iterations. If pushing further: tune the radii alongside the LR.
