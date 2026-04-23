import importlib

import torch

from benchmark_utils.optimizers.muon import Muon
from benchmark_utils.optimizers.orthogonalization import (
    zeropower_via_polar_express,
)
from benchmark_utils.optimizers.scion_light import ScionLight
from benchmark_utils.optimizers.shampoo import Shampoo
from benchmark_utils.optimizers.shampoo_preconditioner import (
    ShampooPreconditioner,
)
from benchmark_utils.optimizers.soap import SOAP


def test_polar_express_tall_matrix():
    matrix = torch.randn(8, 4)
    orthogonal = zeropower_via_polar_express(matrix)
    assert orthogonal.shape == matrix.shape
    assert torch.isfinite(orthogonal).all()


def test_polar_express_wide_matrix():
    matrix = torch.randn(4, 8)
    orthogonal = zeropower_via_polar_express(matrix)
    assert orthogonal.shape == matrix.shape
    assert torch.isfinite(orthogonal).all()


def test_shampoo_preconditioner_roundtrip():
    helper = ShampooPreconditioner()
    grad = torch.randn(6, 4)
    state = {"step": 0, "exp_avg_sq": torch.zeros_like(grad)}
    helper.init_preconditioner(grad, state, precondition_frequency=1)
    helper.update_preconditioner(grad, state, max_precond_dim=10000)

    projected = helper.project(grad, state)
    restored = helper.project_back(projected, state)

    assert projected.shape == grad.shape
    assert restored.shape == grad.shape
    assert torch.isfinite(projected).all()
    assert torch.isfinite(restored).all()
    assert torch.allclose(restored, grad, atol=1e-4, rtol=1e-4)


def test_shampoo_preconditioner_preconditions_matrix():
    helper = ShampooPreconditioner()
    grad = torch.randn(6, 4)
    state = {"step": 1, "exp_avg_sq": torch.zeros_like(grad)}
    helper.init_preconditioner(grad, state, precondition_frequency=1)
    helper.update_preconditioner(grad, state, max_precond_dim=10000)

    preconditioned = helper.precondition(grad, state)
    assert preconditioned.shape == grad.shape
    assert torch.isfinite(preconditioned).all()


def test_muon_step_smoke():
    param = torch.nn.Parameter(torch.randn(6, 4))
    param.grad = torch.randn_like(param)
    optimizer = Muon([param], lr=0.02)

    before = param.detach().clone()
    optimizer.step()

    assert torch.isfinite(param).all()
    assert not torch.allclose(before, param)


def test_scion_step_smoke():
    param = torch.nn.Parameter(torch.randn(6, 4))
    param.grad = torch.randn_like(param)
    optimizer = ScionLight(
        [{"params": [param], "norm": "Spectral", "norm_kwargs": {}, "scale": 1.0}],
        lr=1e-3,
    )

    before = param.detach().clone()
    optimizer.step()

    assert torch.isfinite(param).all()
    assert not torch.allclose(before, param)


def test_soap_step_smoke():
    param = torch.nn.Parameter(torch.randn(6, 4))
    optimizer = SOAP([param], lr=1e-2, weight_decay=0.0, precondition_frequency=1)

    param.grad = torch.randn_like(param)
    optimizer.step()
    before = param.detach().clone()

    param.grad = torch.randn_like(param)
    optimizer.step()

    assert torch.isfinite(param).all()
    assert not torch.allclose(before, param)


def test_shampoo_step_smoke():
    param = torch.nn.Parameter(torch.randn(6, 4))
    optimizer = Shampoo(
        [param],
        lr=1e-2,
        momentum=0.9,
        weight_decay=0.0,
        precondition_frequency=1,
    )

    before = param.detach().clone()
    param.grad = torch.randn_like(param)
    optimizer.step()

    assert torch.isfinite(param).all()
    assert not torch.allclose(before, param)


def test_solver_imports():
    module_names = [
        "solvers.adam",
        "solvers.muon",
        "solvers.scion",
        "solvers.soap",
        "solvers.shampoo",
    ]
    for module_name in module_names:
        importlib.import_module(module_name)
