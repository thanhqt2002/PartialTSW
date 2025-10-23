import numpy as np
import torch
import math
from tqdm import tqdm
import torch.optim as optim
import os

import ot
from tsw import TSW, PartialTSW, generate_trees_frames
from baselines import sopt, spot, pawl, unbalanced_sliced_ot, sliced_unbalanced_ot

from cfg import parse_args, get_cfg_name
args = parse_args()
seed = args.seed
method = args.method
np.random.seed(args.seed)
torch.manual_seed(args.seed)
device='cuda'

def get_loss(X, Y, method, epoch):
    real_batch_latents = X
    fake_latents = Y
    if method == 'twd':
        # Use PartialTSW for linear (supports partial transport) or TSW for other ftypes
        if args.twd_ftype == 'linear':
            twd_calculator = PartialTSW(ntrees=args.twd_ntrees, nlines=args.twd_nlines, p=args.sw_p,
                                        delta=args.twd_delta, mass_division=args.twd_mass_division,
                                        device=device)
        else:
            twd_calculator = TSW(ntrees=args.twd_ntrees, nlines=args.twd_nlines, p=args.sw_p,
                                 delta=args.twd_delta, mass_division=args.twd_mass_division,
                                 ftype=args.twd_ftype, d=3, device=device)
        theta, intercept = generate_trees_frames(ntrees=args.twd_ntrees, nlines=args.twd_nlines, d=3, device=device)
        total_mass_X_tensor = torch.tensor(1.0, device=device); total_mass_Y_tensor = torch.tensor(1.0, device=device)
        if args.twd_unbalanced:
            current_step_float = float(epoch)
            total_steps_float = float(args.num_epoch)
            progress = max(0.0, min(1.0, current_step_float / total_steps_float))
            min_m, max_m = args.min_mass_generated, args.max_mass_generated; scheduled_mass_y = max_m
            if args.twd_unbalanced_scheduler == 'constant': scheduled_mass_y = max_m
            elif args.twd_unbalanced_scheduler == 'linear_increasing': scheduled_mass_y = min_m + progress * (max_m - min_m)
            elif args.twd_unbalanced_scheduler == 'linear_decreasing': scheduled_mass_y = max_m - progress * (max_m - min_m)
            total_mass_Y_tensor = torch.tensor(scheduled_mass_y, device=device)
        g_loss = twd_calculator(real_batch_latents, fake_latents, theta, intercept, total_mass_X=total_mass_X_tensor, total_mass_Y=total_mass_Y_tensor)
    elif method == 'usot':
        mass_X = torch.ones(real_batch_latents.size(0), device=device) / real_batch_latents.size(0)
        mass_Y = torch.ones(fake_latents.size(0), device=device) / fake_latents.size(0)
        g_loss, _, _, _, _, _ = unbalanced_sliced_ot(mass_X, mass_Y, real_batch_latents, fake_latents, num_projections=args.sw_projections, p=args.sw_p,
                                                        rho1=args.rho1, rho2=args.rho2, niter=10)#, mode='icdf')
    elif method == 'sw':
        g_loss = ot.sliced_wasserstein_distance(X, Y, n_projections=100, p=2)
    elif method == 'suot':
        mass_X = torch.ones(real_batch_latents.size(0), device=device)
        mass_Y = torch.ones(fake_latents.size(0), device=device)
        g_loss, _, _, _, _, _ = sliced_unbalanced_ot(mass_X, mass_Y, real_batch_latents, fake_latents, num_projections=args.sw_projections, p=args.sw_p,
                                                        rho1=args.rho1, rho2=args.rho2, niter=10)#, mode='icdf')
    elif method == 'sopt':
        g_loss = sopt(real_batch_latents.cpu(), fake_latents.cpu(), n_proj=args.sw_projections, reg=1) # SOPT/SPOT/PAWL might need CPU
    elif method == 'spot':
        g_loss = spot(real_batch_latents.cpu(), fake_latents.cpu(), n_proj=args.sw_projections)
    elif method == 'pawl':
        g_loss = pawl(real_batch_latents.cpu(), fake_latents.cpu(), n_proj=args.sw_projections, k=args.pawl_k)
    elif method == 'pot':
        # Ensure tensors are on the correct device and dtype for POT
        # a and b are marginals (weights for each sample). Uniform for now.
        a = torch.ones(real_batch_latents.size(0), device=device, dtype=real_batch_latents.dtype) / real_batch_latents.size(0)
        b = torch.ones(fake_latents.size(0), device=device, dtype=fake_latents.dtype) / fake_latents.size(0)

        # M is the cost matrix. Using squared Euclidean distance (L2^2).
        # ot.dist will use the PyTorch backend.
        # The power p for the cost metric. If args.sw_p = 2, this is squared Euclidean.
        # If args.sw_p = 1, this is L1 distance.
        # M = ot.dist(real_batch_latents, fake_latents, metric='euclidean').pow(args.sw_p) # M_ij = ||x_i - y_j||_2^p
        # Common choice: Squared Euclidean distance, equivalent to metric='sqeuclidean' if p=2
        if args.pot_cost_metric_p == 2:
            M = ot.dist(real_batch_latents, fake_latents, metric='sqeuclidean')
        elif args.pot_cost_metric_p == 1:
                M = ot.dist(real_batch_latents, fake_latents, metric='euclidean') # L1
        else: # Generic p-norm to the power p
            M = torch.cdist(real_batch_latents, fake_latents, p=args.pot_cost_metric_p).pow(args.pot_cost_metric_p)
        # Call the POT function
        # entropic_kl_uot_ti is the transport cost, which will be our loss
        _, log_uot_ti = ot.unbalanced.sinkhorn_unbalanced2(
            a, b, M,
            reg=args.pot_reg,
            reg_m=args.pot_reg_m_kl, # reg_m_kl in your snippet refers to reg_m here
            method="sinkhorn_translation_invariant",
            numItermax=args.pot_num_iter_max,
            stopThr=args.pot_stop_thr,
            log=True, # To get the log dictionary
            # reg_type="kl" # This seems to be default for sinkhorn_unbalanced if reg_m is for KL
        )
        g_loss = log_uot_ti['cost']
    else:
        raise ValueError(f"Unsupported loss type: {method}")
    
    return g_loss

if __name__ == "__main__":
    source_data = torch.load(f"shape_data_bai/dragon.pt")["T"]
    target_data = torch.load("shape_data_bai/bunny_noise.pt")
    print(method, source_data.shape, target_data.shape)

    X = torch.tensor(source_data, device=device, requires_grad=True)
    Y = torch.tensor(target_data, device=device)
    X_trans = np.zeros((args.num_epoch + 1, X.shape[0], X.shape[1]))
    X_trans[0] = X.detach().cpu().numpy()

    optimizer = optim.Adam([X], lr=args.lr)
    
    for i in range(args.num_epoch):
        optimizer.zero_grad()
        loss = get_loss(X=X, Y=Y, method=method, epoch=i)
        loss.backward()
        optimizer.step()
        if (i + 1) % 50 == 0:
            print(f"epoch {i + 1}: {loss.item()}")
        X_trans[i + 1] = X.detach().cpu().numpy()

    os.makedirs(f"results/{method}", exist_ok=True)
    np.savez(f"results/{method}/{get_cfg_name(args)}.npz",
            X=X_trans, Y=Y.detach().cpu().numpy())