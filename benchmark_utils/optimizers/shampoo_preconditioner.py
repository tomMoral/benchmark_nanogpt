from itertools import chain

import torch


class ShampooPreconditioner:
    def __init__(self, data_format="channels_first"):
        self._data_format = data_format

    def merge_dims(self, grad, max_precond_dim):
        """Merge dimensions until each merged axis fits the preconditioner."""
        assert self._data_format in ["channels_first", "channels_last"]
        if self._data_format == "channels_last" and grad.dim() == 4:
            grad = grad.permute(0, 3, 1, 2)
        shape = grad.shape
        new_shape = []

        curr_shape = 1
        for sh in shape:
            temp_shape = curr_shape * sh
            if temp_shape > max_precond_dim:
                if curr_shape > 1:
                    new_shape.append(curr_shape)
                    curr_shape = sh
                else:
                    new_shape.append(sh)
                    curr_shape = 1
            else:
                curr_shape = temp_shape

        if curr_shape > 1 or len(new_shape) == 0:
            new_shape.append(curr_shape)

        return grad.reshape(new_shape)

    def init_preconditioner(
        self,
        grad,
        state,
        precondition_frequency=10,
        shampoo_beta=0.95,
        max_precond_dim=10000,
        precondition_1d=False,
        merge_dims=False,
    ):
        """Initialize the preconditioner statistics."""
        if grad.ndim <= 1:
            return grad
        state["GG"] = []
        if grad.dim() == 1:
            if not precondition_1d or grad.shape[0] > max_precond_dim:
                state["GG"].append([])
            else:
                state["GG"].append(
                    torch.zeros(grad.shape[0], grad.shape[0], device=grad.device)
                )
        else:
            if merge_dims:
                grad = self.merge_dims(grad, max_precond_dim)

            for sh in grad.shape:
                if sh > max_precond_dim:
                    state["GG"].append([])
                else:
                    state["GG"].append(torch.zeros(sh, sh, device=grad.device))

        state["Q"] = None
        state["eigenvalues"] = None
        state["precondition_frequency"] = precondition_frequency
        state["shampoo_beta"] = shampoo_beta

    def project(self, grad, state, merge_dims=False, max_precond_dim=10000):
        """Project a gradient to the eigenbases of the preconditioner."""
        if grad.ndim <= 1:
            return grad
        original_shape = grad.shape
        if merge_dims:
            if grad.dim() == 4 and self._data_format == "channels_last":
                permuted_shape = grad.permute(0, 3, 1, 2).shape
            grad = self.merge_dims(grad, max_precond_dim)

        for mat in state["Q"]:
            if len(mat) > 0:
                grad = torch.tensordot(grad, mat, dims=[[0], [0]])
            else:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)

        if merge_dims:
            if self._data_format == "channels_last" and len(original_shape) == 4:
                grad = grad.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                grad = grad.reshape(original_shape)
        return grad

    def update_preconditioner(
        self,
        grad,
        state,
        max_precond_dim=10000,
        merge_dims=False,
        precondition_1d=False,
    ):
        """Update the preconditioner statistics and refresh eigenbases."""
        if grad.ndim <= 1:
            return grad
        if state["Q"] is not None and "exp_avg" in state:
            state["exp_avg"] = self.project_back(
                state["exp_avg"],
                state,
                merge_dims=merge_dims,
                max_precond_dim=max_precond_dim,
            )
        if grad.dim() == 1:
            if precondition_1d and grad.shape[0] <= max_precond_dim:
                state["GG"][0].lerp_(
                    grad.unsqueeze(1) @ grad.unsqueeze(0),
                    1 - state["shampoo_beta"],
                )
        else:
            if merge_dims:
                new_grad = self.merge_dims(grad, max_precond_dim)
                for idx, sh in enumerate(new_grad.shape):
                    if sh <= max_precond_dim:
                        outer_product = torch.tensordot(
                            new_grad,
                            new_grad,
                            dims=[
                                [
                                    *chain(
                                        range(idx),
                                        range(idx + 1, len(new_grad.shape)),
                                    )
                                ]
                            ]
                            * 2,
                        )
                        state["GG"][idx].lerp_(
                            outer_product, 1 - state["shampoo_beta"]
                        )
            else:
                for idx, sh in enumerate(grad.shape):
                    if sh <= max_precond_dim:
                        outer_product = torch.tensordot(
                            grad,
                            grad,
                            dims=[
                                [
                                    *chain(
                                        range(idx),
                                        range(idx + 1, len(grad.shape)),
                                    )
                                ]
                            ]
                            * 2,
                        )
                        state["GG"][idx].lerp_(
                            outer_product, 1 - state["shampoo_beta"]
                        )

        if state["Q"] is None:
            state["Q"], state["eigenvalues"] = self.get_orthogonal_matrix(
                state["GG"]
            )
        if (
            state["step"] > 0
            and state["step"] % state["precondition_frequency"] == 0
        ):
            state["Q"], state["eigenvalues"] = self.get_orthogonal_matrix_QR(
                state,
                max_precond_dim=max_precond_dim,
                merge_dims=merge_dims,
            )

        if state["step"] > 0 and "exp_avg" in state:
            state["exp_avg"] = self.project(
                state["exp_avg"],
                state,
                merge_dims=merge_dims,
                max_precond_dim=max_precond_dim,
            )

    def project_back(self, grad, state, merge_dims=False, max_precond_dim=10000):
        """Project a tensor back to the original parameter basis."""
        if grad.ndim <= 1:
            return grad
        original_shape = grad.shape
        if merge_dims:
            if self._data_format == "channels_last" and grad.dim() == 4:
                permuted_shape = grad.permute(0, 3, 1, 2).shape
            grad = self.merge_dims(grad, max_precond_dim)
        for mat in state["Q"]:
            if len(mat) > 0:
                grad = torch.tensordot(grad, mat, dims=[[0], [1]])
            else:
                permute_order = list(range(1, len(grad.shape))) + [0]
                grad = grad.permute(permute_order)

        if merge_dims:
            if self._data_format == "channels_last" and len(original_shape) == 4:
                grad = grad.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                grad = grad.reshape(original_shape)
        return grad

    def get_orthogonal_matrix(self, matrices):
        """Compute exact eigenbases for the current preconditioners."""
        bases = []
        eigenvalues = []
        for matrix in matrices:
            if len(matrix) == 0:
                bases.append([])
                eigenvalues.append([])
                continue

            matrix_fp32 = matrix.detach().float()
            eye = torch.eye(matrix.shape[0], device=matrix.device)
            try:
                eigvals, basis = torch.linalg.eigh(matrix_fp32 + 1e-6 * eye)
            except RuntimeError:
                eigvals, basis = torch.linalg.eigh(
                    matrix_fp32.to(torch.float64) + 1e-6 * eye
                )
                basis = basis.to(matrix_fp32.dtype)
                eigvals = eigvals.to(matrix_fp32.dtype)

            basis = torch.flip(basis, [1]).to(matrix.device).type(matrix.dtype)
            eigvals = torch.flip(eigvals, [0]).to(matrix.device)
            bases.append(basis)
            eigenvalues.append(eigvals)
        return bases, eigenvalues

    def get_orthogonal_matrix_QR(
        self, state, max_precond_dim=10000, merge_dims=False
    ):
        """Refresh eigenbases with one power iteration followed by QR."""
        precond_list = state["GG"]
        orth_list = state["Q"]

        matrix = []
        orth_matrix = []
        dtypes = []
        devices = []
        for preconditioner, basis in zip(precond_list, orth_list):
            if len(preconditioner) == 0:
                matrix.append([])
                orth_matrix.append([])
                dtypes.append(None)
                devices.append(None)
                continue
            matrix.append(preconditioner.detach().float())
            orth_matrix.append(basis.detach().float())
            dtypes.append(preconditioner.dtype)
            devices.append(preconditioner.device)

        orig_shape = state["exp_avg_sq"].shape
        if self._data_format == "channels_last" and len(orig_shape) == 4:
            permuted_shape = state["exp_avg_sq"].permute(0, 3, 1, 2).shape
        if merge_dims:
            exp_avg_sq = self.merge_dims(state["exp_avg_sq"], max_precond_dim)
        else:
            exp_avg_sq = state["exp_avg_sq"]

        bases = []
        eigenvalues = []
        for ind, (matrix_i, basis_i) in enumerate(zip(matrix, orth_matrix)):
            if len(matrix_i) == 0:
                bases.append([])
                eigenvalues.append([])
                continue
            estimated_eigenvalues = torch.diag(basis_i.T @ matrix_i @ basis_i)
            sort_idx = torch.argsort(estimated_eigenvalues, descending=True)
            exp_avg_sq = exp_avg_sq.index_select(ind, sort_idx)
            basis_i = basis_i[:, sort_idx]
            power_iter = matrix_i @ basis_i
            refreshed_basis, _ = torch.linalg.qr(power_iter)
            refreshed_eigenvalues = torch.diag(
                refreshed_basis.T @ matrix_i @ refreshed_basis
            ).clamp_min_(0)

            bases.append(
                refreshed_basis.to(devices[ind]).type(dtypes[ind])
            )
            eigenvalues.append(refreshed_eigenvalues.to(devices[ind]))

        if merge_dims:
            if self._data_format == "channels_last" and len(orig_shape) == 4:
                exp_avg_sq = exp_avg_sq.reshape(permuted_shape).permute(0, 2, 3, 1)
            else:
                exp_avg_sq = exp_avg_sq.reshape(orig_shape)

        state["exp_avg_sq"] = exp_avg_sq
        return bases, eigenvalues

    def precondition(self, grad, state, eps=1e-12, merge_dims=False, max_precond_dim=10000):
        """Apply the classic Shampoo inverse-root preconditioner."""
        if grad.ndim <= 1 or state.get("Q") is None:
            return grad
        if merge_dims:
            raise NotImplementedError(
                "Classic Shampoo preconditioning does not support merged dims."
            )

        grad_projected = self.project(
            grad,
            state,
            merge_dims=merge_dims,
            max_precond_dim=max_precond_dim,
        )
        preconditioner_scale = 1
        order = grad_projected.ndim
        for axis, eigvals in enumerate(state["eigenvalues"]):
            if len(eigvals) == 0:
                continue
            view_shape = [1] * order
            view_shape[axis] = eigvals.numel()
            factor = eigvals.clamp_min(eps).pow(0.5 / order).view(view_shape)
            preconditioner_scale = preconditioner_scale * factor

        return self.project_back(
            grad_projected / preconditioner_scale,
            state,
            merge_dims=merge_dims,
            max_precond_dim=max_precond_dim,
        )
