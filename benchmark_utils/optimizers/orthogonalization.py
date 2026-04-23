import torch


# Polar Express coefficients (num_iters=5, safety_factor=2e-2, cushion=2)
# From https://arxiv.org/pdf/2505.16932
# Amsel, Persson, Musco & Gower (2025)
POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def zeropower_via_polar_express(matrix):
    """Orthogonalize a 2D tensor with the Polar Express iteration."""
    if matrix.ndim != 2:
        raise ValueError("Polar Express expects a 2D tensor.")

    orthogonal = matrix.bfloat16()
    transposed = orthogonal.size(0) > orthogonal.size(1)
    if transposed:
        orthogonal = orthogonal.T

    # Normalize so the spectral radius stays inside the stable region.
    orthogonal = orthogonal / (orthogonal.norm() * (1 + 2e-2) + 1e-6)

    for a, b, c in POLAR_EXPRESS_COEFFS:
        gram = orthogonal @ orthogonal.T
        orthogonal = a * orthogonal + (b * gram + c * gram @ gram) @ orthogonal

    if transposed:
        orthogonal = orthogonal.T
    return orthogonal
