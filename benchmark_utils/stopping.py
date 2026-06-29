"""A target-based stopping criterion for fixed-budget training benchmarks.

``TargetStoppingCriterion`` stops a solver as soon as the monitored metric
reaches a target value. Combined with each solver's finite ``num_steps``
budget, this gives the semantics these benchmarks want: *run until the target
metric (perplexity / error rate) is reached, otherwise exhaust the budget*.
When no target is set it never stops early, so it is a safe benchmark-wide
default (equivalent to ``NoCriterion`` for datasets without a target).

Making the target depend on the dataset
---------------------------------------
The target is a *static property of the (dataset, objective)* — not something
produced by an evaluation. There are two ways to expose it to the criterion:

1. **Through the objective output** — ``evaluate_result`` returns ``target`` so
   it becomes the ``objective_target`` column, and the criterion reads it from
   ``objective_list``. Simple, but conceptually wrong: a constant gets logged
   as a per-evaluation metric, duplicated on every row of the results parquet.

2. **From the live objective (used here)** — the dataset sets ``target`` on the
   objective (via ``set_data``), and the criterion reads it through the
   solver handle it is given (``self.solver._objective.target``). The target
   stays out of the metric stream and travels naturally dataset -> objective.
   This is the cleaner model and the most practical one that works *today*.

   Note it leans on ``_objective`` (a private attribute) because benchopt does
   not give a ``StoppingCriterion`` a public handle to the running objective,
   nor a payload on ``RunContext`` (which only carries names/seeds/paths).

Best vs. practical
------------------
The *best* design would let the criterion fetch static problem properties from
first-class benchopt state instead of a private attribute. ``RunContext``
already wires the run together, so it should expose public references to the
``dataset`` and ``objective`` (not just their names) that the criterion can
read through its ``solver`` handle. That keeps a single source of truth, avoids
adding an ``extra: dict`` payload — an extra layer of indirection for what is
just "let the criterion see the objective". Until benchopt offers that,
reading ``solver._objective.target`` here is the pragmatic choice.
"""
from benchopt.stopping_criterion import StoppingCriterion


class TargetStoppingCriterion(StoppingCriterion):

    def __init__(self, target=None, strategy=None, key_to_monitor="value",
                 minimize=True, **kwargs):
        self.target = target
        # Pass target through to the base so it is stored in ``self.kwargs``
        # and restored when benchopt clones the criterion per run.
        super().__init__(
            target=target, strategy=strategy,
            key_to_monitor=key_to_monitor, minimize=minimize, **kwargs
        )

    def _get_target(self):
        # Prefer a per-dataset target set on the running objective; else use
        # the benchmark-wide value passed at construction (possibly None).
        objective = getattr(getattr(self, "solver", None), "_objective", None)
        target = getattr(objective, "target", None)
        return target if target is not None else self.target

    def check_convergence(self, objective_list):
        target = self._get_target()
        if target is None:
            return False, 0.0  # no target -> run the full budget

        sign = 1 if self.minimize else -1
        value = objective_list[-1][self.key_to_monitor_]
        if sign * value <= sign * target:
            return True, 1.0

        # Progress in [0, 1): how far from start towards the target.
        start = objective_list[0][self.key_to_monitor_]
        denom = sign * (start - target)
        if denom <= 0:
            return False, 0.0
        return False, min(1.0, max(0.0, sign * (start - value) / denom))
