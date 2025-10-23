import torch

def svd_orthogonalize(matrix):
    U, _, _ = torch.linalg.svd(matrix, full_matrices=False)
    return U

def generate_trees_frames(ntrees, nlines, d, mean=0.0, std=1.0, device='cuda', gen_mode='gaussian_orthogonal'):
    """
    Generates frames (theta vectors and intercepts) for Partial TSW computation.

    Args:
        ntrees (int): Number of trees.
        nlines (int): Number of lines per tree.
        d (int): Dimension of the space.
        mean (float): Mean for Gaussian initialization. Default 0.0.
        std (float): Standard deviation for Gaussian initialization. Default 1.0.
        device (str): Torch device ('cuda', 'cpu').
        gen_mode (str): Method for generating theta: 'gaussian_raw' or 'gaussian_orthogonal'.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (theta, intercept)
            theta: Shape (ntrees, nlines, d)
            intercept: Shape (ntrees, 1, d)
    """
    assert gen_mode in ['gaussian_raw', 'gaussian_orthogonal'], f"Invalid gen_mode: {gen_mode}"

    # Generate Intercept
    intercept = torch.randn(ntrees, 1, d, device=device, dtype=torch.float32) * std + mean

    # Generate Theta
    if gen_mode == 'gaussian_raw':
        theta_init = torch.randn(ntrees, nlines, d, device=device, dtype=torch.float32)
        theta = theta_init / torch.linalg.norm(theta_init, dim=-1, keepdim=True).clamp(min=1e-12)
    elif gen_mode == 'gaussian_orthogonal':
        assert nlines <= d, "nlines must be <= d for gaussian_orthogonal"
        theta_init = torch.randn(ntrees, d, nlines, device=device, dtype=torch.float32)
        theta_ortho = svd_orthogonalize(theta_init)
        theta = theta_ortho.transpose(-2, -1)

    return theta.float(), intercept.float()