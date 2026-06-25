import torch


def compile_step(optimizer_cls):
    """``torch.compile`` an optimizer's ``step`` method, idempotently.

    Solvers patch the class-level ``step`` so every instance shares the
    compiled version. Class attributes are global and persist across solvers
    and tests in a single process, so a shared optimizer (e.g. ``AdamW``) can
    be compiled twice. Re-compiling an already-compiled function trips torch's
    ``get_compiler_config`` assertion, so skip it when already done.
    """
    step = optimizer_cls.step
    if getattr(step, "_torchdynamo_orig_callable", None) is None:
        optimizer_cls.step = torch.compile(torch.no_grad(step))
