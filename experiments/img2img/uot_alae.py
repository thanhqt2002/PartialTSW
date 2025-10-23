import os
import sys
import torch
from torch import optim
import numpy as np
import math
import matplotlib.pyplot as plt
import wandb
from func import *
sys.path.append("./ALAE")

from tsw import PartialTSW, generate_trees_frames
from baselines import sliced_wasserstein, sopt, spot, pawl, unbalanced_sliced_ot, sliced_unbalanced_ot
from baselines import ULightOT

from src.distributions import TensorSampler
try:
    from baselines.uot_fm import uot_fm
    HAS_UOT_FM = True
except ImportError:
    uot_fm = None
    HAS_UOT_FM = False
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

from torch.optim.lr_scheduler import MultiStepLR

from alae_ffhq_inference import load_config
from src.fid_score import calculate_frechet_distance
import time
import argparse

parser = argparse.ArgumentParser('img2img parameters')
parser.add_argument('--download', action='store_true', default=False, help='download data or not')
parser.add_argument('--plot', action='store_true', default=False, help='save figure or not')
parser.add_argument('--wandb', action='store_true', default=False, help='log to wandb or not')
parser.add_argument('--dim', type=int, default=512)
parser.add_argument('--input', type=str, choices=['YOUNG', 'MAN', 'WOMAN', 'ADULT'], default='YOUNG')
parser.add_argument('--target', type=str, choices=['YOUNG', 'MAN', 'WOMAN', 'ADULT'], default='ADULT')

parser.add_argument('--method', type=str, choices=['twd', 'sw', 'ulight', 'uot-fm', 'faster-uot', 'usot', 'suot', 'sopt', 'spot', 'pawl'], default='ulight')
parser.add_argument('--ulight_tau', type=float, default=100)
parser.add_argument('--faster_uot_reg', type=float, default=0.5)
parser.add_argument('--faster_uot_reg_m_kl', type=float, default=0.5)
parser.add_argument('--ufm_eps', type=float, default=0.5)
parser.add_argument('--L', type=int, default=100)
parser.add_argument('--twd_nlines', type=int, default=4)
parser.add_argument('--twd_min_total_mass_Y', type=float, default=1)
parser.add_argument('--twd_max_total_mass_Y', type=float, default=1.1)
parser.add_argument('--twd_unbalanced_scheduler', type=str, default='constant')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--num_epoch', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--rho1', type=float, default=0.01, help='First marginal relaxation term for USOT/SUOT')
parser.add_argument('--rho2', type=float, default=1.0, help='Second marginal relaxation term for USOT/SUOT')
parser.add_argument('--sopt_reg', type=float, default=1.0, help='Regularization parameter for SOPT')
parser.add_argument('--spot_k', type=int, default=10, help='Number of points for SPOT')
parser.add_argument('--pawl_k', type=int, default=10, help='Number of points for PAWL')

args = parser.parse_args()
device = args.device
use_cuda = device == 'cuda' and torch.cuda.is_available()
if device == 'cuda' and not use_cuda:
    print("Warning: CUDA device requested but not available. Switching to CPU.")
    device = 'cpu'

METHOD = args.method
DIM = args.dim
assert DIM > 1

INPUT_DATA = args.input
TARGET_DATA = args.target

OUTPUT_SEED = 42
BATCH_SIZE = 256
EPSILON = 0.05
DIVERGENCE = 'KL'#"KL" #balanced
K_POTENTIALS = L_POTENTIALS = 10
twd_ntrees = args.L // args.twd_nlines
Tau = args.ulight_tau
D_LR = 1
D_GRADIENT_MAX_NORM = 1e5 #float("inf")
SAMPLING_BATCH_SIZE = 128
INIT_BY_SAMPLES = True
IS_DIAGONAL = True
LR = args.lr

torch.manual_seed(OUTPUT_SEED)
np.random.seed(OUTPUT_SEED)

### Download data
if args.download:
    download_data()
    exit(0)

## Load classifers
# ... (classifier loading code remains the same) ...
if INPUT_DATA in ['ADULT', 'YOUNG']:
    mlp_classifier = BinaryClassifier().to(device)
    mlp_classifier.load_state_dict(torch.load('./checkpoints/male_female_classifier.pth', map_location=device))

    target_mlp_classifier = BinaryClassifier().to(device)
    target_mlp_classifier.load_state_dict(torch.load('./checkpoints/young_old_classifier.pth', map_location=device))
elif INPUT_DATA in ['MAN', 'WOMAN']:
    mlp_classifier = BinaryClassifier().to(device)
    mlp_classifier.load_state_dict(torch.load('./checkpoints/young_old_classifier.pth', map_location=device))

    target_mlp_classifier = BinaryClassifier().to(device)
    target_mlp_classifier.load_state_dict(torch.load('./checkpoints/male_female_classifier.pth', map_location=device))


## Logs
os.makedirs("logs", exist_ok=True)
EXP_NAME = f'{INPUT_DATA.lower()}_to_{TARGET_DATA.lower()}'
if METHOD == 'twd':
    RUN_NAME = f"TWD__trees_{twd_ntrees}__lines_{args.twd_nlines}__{args.twd_unbalanced_scheduler}__minY_{args.twd_min_total_mass_Y}__maxY_{args.twd_max_total_mass_Y}__ep_{args.num_epoch}__lr_{LR}"
elif METHOD == 'sw':
    RUN_NAME = f"SW_L_{args.L}__ep_{args.num_epoch}__lr_{args.lr}"
elif METHOD == 'ulight':
    RUN_NAME = f"ULight__tau_{args.ulight_tau}__ep_{args.num_epoch}"
elif METHOD == 'uot-fm':
    from baselines.uot_fm.config import get_config
    config = get_config()
    config.training.epsilon = args.ufm_eps
    config.training.num_steps = args.num_epoch
    RUN_NAME = f"UOT-FM__eps_{config.training.epsilon}__ep_{config.training.num_steps}"
elif METHOD == 'faster-uot':
    RUN_NAME = f"FasterUOT__reg_{args.faster_uot_reg}__reg_m_kl_{args.faster_uot_reg_m_kl}__ep_{args.num_epoch}"
elif METHOD == 'usot':
    RUN_NAME = f"USOT__rho1_{args.rho1}__rho2_{args.rho2}__ep_{args.num_epoch}"
elif METHOD == 'suot':
    RUN_NAME = f"SUOT__rho1_{args.rho1}__rho2_{args.rho2}__ep_{args.num_epoch}"
elif METHOD == 'sopt':
    RUN_NAME = f"SOPT__reg_{args.sopt_reg}__ep_{args.num_epoch}"
elif METHOD == 'spot':
    RUN_NAME = f"SPOT__k_{args.spot_k}__ep_{args.num_epoch}"
elif METHOD == 'pawl':
    RUN_NAME = f"PAWL__k_{args.pawl_k}__ep_{args.num_epoch}"

stats_file = open(f"logs/{EXP_NAME}.txt", "a", buffering=1)
print_stats(RUN_NAME, stats_file)

## Load data
# ... (data loading code remains the same) ...
train_size = 60000
test_size = 10000
latents = np.load("data/latents.npy")
gender = np.load("data/gender.npy")
age = np.load("data/age.npy")
_, test_gender = gender[:train_size], gender[train_size:]
_, test_age = age[:train_size], age[train_size:]
train_latents, test_latents = latents[:train_size], latents[train_size:]
x_inds_train, y_inds_train, x_inds_test, y_inds_test = load_inds(INPUT_DATA, TARGET_DATA, train_size, test_size)
x_data_train = train_latents[x_inds_train]
x_data_test = test_latents[x_inds_test]
x_data_test_gender = test_gender[x_inds_test]
x_data_test_age = test_age[x_inds_test]

y_data_train = train_latents[y_inds_train]
y_data_test = test_latents[y_inds_test]

### Stats X_train, Y_train, X_test, Y_test
# ... (stats calculation remains the same) ...
X_train = torch.tensor(x_data_train) # (38382, 512)
Y_train = torch.tensor(y_data_train) # (10869, 512)
X_test = torch.tensor(x_data_test) # (6544, 512)
Y_test = torch.tensor(y_data_test) # (1651, 512)
X_sampler = TensorSampler(X_train, device=device)
Y_sampler = TensorSampler(Y_train, device=device)

real_data = Y_sampler.sample(10000)
real_data = real_data.cpu().data.numpy().reshape(real_data.size(0), -1)
mu_data, sigma_data = np.mean(real_data, axis=0), np.cov(real_data, rowvar=False)

source_data = X_sampler.sample(10000)
source_data = source_data.cpu().data.numpy().reshape(source_data.size(0), -1)
mu_source, sigma_source = np.mean(source_data, axis=0), np.cov(source_data, rowvar=False)


# --- Model Setup ---
if METHOD == 'ulight':
    D = ULightOT(dim=DIM, k_potentials=K_POTENTIALS, l_potentials=L_POTENTIALS,epsilon=EPSILON,
                sampling_batch_size=SAMPLING_BATCH_SIZE, is_diagonal=IS_DIAGONAL).to(device)

    log_m = torch.zeros(1, requires_grad=True, device=device)

    if INIT_BY_SAMPLES:
        D.init_r_by_samples(Y_sampler.sample(K_POTENTIALS))

    D_opt = torch.optim.Adam(D.parameters(), lr=D_LR)
    m_opt = torch.optim.Adam([log_m], lr=1e-3)
    D_sch = MultiStepLR(D_opt, milestones=[500, 1000])
else: # twd or sw
    cfg = load_config("./ALAE/configs/ffhq.yaml", training_artifacts_dir="./ALAE/training_artifacts/ffhq/")
    D = ResidualMLP(dim=cfg.MODEL.LATENT_SPACE_SIZE, num_hidden_blocks=0, hidden_dim_multiplier=1).to(device)
    D_opt = optim.Adam(D.parameters(), lr=LR)
    if METHOD == 'twd':
        torch.set_float32_matmul_precision('high')
        # Use PartialTSW for linear (supports partial transport with total_mass_X and total_mass_Y)
        TWD_obj = torch.compile(PartialTSW(ntrees=twd_ntrees, nlines=args.twd_nlines,
                                            mass_division='distance_based',
                                            p=1, device=device))

### Training
start_time = time.time()
if use_cuda:
    torch.cuda.reset_peak_memory_stats(device=device)
    # Note: reset_peak_memory_stats affects the results of max_memory_allocated
if METHOD == 'uot-fm':
    model = uot_fm.train(X_sampler, Y_sampler, config)
else:
    for epoch in tqdm(range(args.num_epoch)):
        X, Y = X_sampler.sample(BATCH_SIZE), Y_sampler.sample(BATCH_SIZE)
        # training cycle
        if METHOD == 'ulight':
            D_opt.zero_grad()
            m_opt.zero_grad()

            tau = 1000 if epoch < 1000 else Tau
            log_V = D.get_potential(Y)
            psi = EPSILON * log_V + torch.norm(Y, p=2, dim=-1)**2/2
            if DIVERGENCE == 'KL':
                f_psi = tau * (torch.exp(-psi/tau) - 1)
            elif DIVERGENCE == 'Xi2':
                psi = -(F.relu(-psi + 2*tau) - (1+(-psi>-2*tau))*tau)
                f_psi = 0.25 * psi**2/tau - psi
            elif DIVERGENCE == 'balanced':
                f_psi = -psi
            log_C = D.get_C(X)
            log_U = D.get_marginal(X)
            phi = EPSILON * (log_U + log_m - log_C) + torch.norm(X, p=2, dim=-1)**2/2
            if DIVERGENCE == 'KL':
                f_phi = tau * (torch.exp(-phi/tau) - 1)
            elif DIVERGENCE == 'Xi2':
                phi = -(F.relu(-phi + 2*tau) - (1+(-phi>-2*tau))*tau)
                f_phi = 0.25 * phi**2/tau - phi
            elif DIVERGENCE == 'balanced':
                f_phi = -phi
            D_loss = EPSILON * torch.exp(log_m) + f_phi.mean() + f_psi.mean()
            D_loss.backward()

            D_gradient_norm = torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=D_GRADIENT_MAX_NORM)
            D_opt.step()
            m_opt.step()
            D_sch.step()
        elif METHOD == 'twd':
            total_mass_Y = 1.0  # Default value (balanced case, where mass ratio is 1.0)
            min_mass = args.twd_min_total_mass_Y      # The minimum total mass corresponds to the balanced case
            max_mass = args.twd_max_total_mass_Y # The target total mass for the generated distribution

            # Ensure float division for progress calculation, handle potential num_epoch=0
            current_epoch_float = float(epoch)
            total_epochs_float = float(args.num_epoch) if args.num_epoch > 0 else 1.0 # Avoid division by zero

            # Calculate progress, clamped between 0 and 1
            progress = max(0.0, min(1.0, current_epoch_float / total_epochs_float))

            # --- Existing Schedulers (Using new names) ---
            if args.twd_unbalanced_scheduler == 'constant':
                total_mass_Y = max_mass if max_mass != 1 else min_mass

            elif args.twd_unbalanced_scheduler == 'increasing': # Linear increasing
                # Interpolate linearly from min_mass up to max_mass
                total_mass_Y = min_mass + progress * (max_mass - min_mass)

            elif args.twd_unbalanced_scheduler == 'decreasing': # Linear decreasing
                # Interpolate linearly from max_mass down to min_mass
                total_mass_Y = min_mass + (1.0 - progress) * (max_mass - min_mass)

            # --- Added Cosine Schedulers (Using new names) ---
            elif args.twd_unbalanced_scheduler == 'cosine_increasing':
                # Increases from min_mass to max_mass following a cosine shape
                cosine_val = 0.5 * (1.0 - math.cos(math.pi * progress)) # Goes from 0 to 1
                total_mass_Y = min_mass + cosine_val * (max_mass - min_mass)

            elif args.twd_unbalanced_scheduler == 'cosine_decreasing':
                # Decreases from max_mass to min_mass following a cosine shape
                cosine_val = 0.5 * (1.0 + math.cos(math.pi * progress)) # Goes from 1 to 0
                total_mass_Y = min_mass + cosine_val * (max_mass - min_mass)

            elif args.twd_unbalanced_scheduler == 'cyclic_triangular':
                # triangle wave between min_mass and max_mass that repeats every cycle_len epochs
                cycle_len = 100                     # length of one full up-and-down cycle
                phase     = (epoch % cycle_len) / cycle_len   # ∈ [0,1)
                # triangle: 0→1 (ascending), 1→0 (descending)
                tri = 2*phase if phase < 0.5 else 2*(1-phase)
                total_mass_Y = min_mass + tri * (max_mass - min_mass)

            elif args.twd_unbalanced_scheduler == 'cyclic_cosine':
                # smooth cosine wave between the bounds, repeating every cycle_len epochs
                cycle_len = 100
                phase     = (epoch % cycle_len) / cycle_len               # ∈ [0,1)
                cos_val   = 0.5*(1 - math.cos(2*math.pi*phase))           # 0→1→0
                total_mass_Y = min_mass + cos_val * (max_mass - min_mass)

            elif args.twd_unbalanced_scheduler == 'reversed_cyclic_triangular':
                # triangle wave between min_mass and max_mass that repeats every cycle_len epochs
                cycle_len = 100                     # length of one full up-and-down cycle
                phase     = (epoch % cycle_len) / cycle_len   # ∈ [0,1)
                # triangle: 0→1 (ascending), 1→0 (descending)
                tri = 2*phase if phase < 0.5 else 2*(1-phase)
                total_mass_Y = min_mass + (1 - tri) * (max_mass - min_mass)
                
            elif args.twd_unbalanced_scheduler == 'reversed_cyclic_cosine':
                # smooth cosine wave between the bounds, repeating every cycle_len epochs
                cycle_len = 100
                phase     = (epoch % cycle_len) / cycle_len               # ∈ [0,1)
                cos_val   = 0.5*(1 - math.cos(2*math.pi*phase))           # 0→1→0
                total_mass_Y = min_mass + (1 - cos_val) * (max_mass - min_mass)

            else:
                # Optional: Add a warning or error for unknown scheduler types
                print(f"Warning: Unknown twd_unbalanced_scheduler '{args.twd_unbalanced_scheduler}'. Using constant value {max_mass}.")
                total_mass_Y = max_mass # Fallback to constant

            D_opt.zero_grad()
            X_transformed = D(X)
            theta_twd, intercept_twd = generate_trees_frames(
                ntrees=twd_ntrees,
                nlines=args.twd_nlines,
                d=X_transformed.shape[1],
                mean=torch.mean(X_transformed, dim=0),
                std=0.01,
                gen_mode='gaussian_raw',
                device=device
            )
            D_loss = TWD_obj(
                X_transformed, Y, theta_twd, intercept_twd,
                total_mass_X=torch.tensor([1.0], device=X.device),
                total_mass_Y=torch.tensor([total_mass_Y], device=X.device)
            )
            D_loss.backward()
            D_opt.step()

        elif METHOD == 'sw':
            D_opt.zero_grad()
            X_transformed = D(X)
            D_loss = sliced_wasserstein(X=X_transformed, Y=Y, num_projections=args.L, p=2, device=device)
            D_loss.backward()
            D_opt.step()
        elif METHOD == 'faster-uot':
            D_opt.zero_grad()
            X_transformed = D(X)
            a = torch.ones(X_transformed.size(0), device=device, dtype=X_transformed.dtype) / X_transformed.size(0)
            b = torch.ones(Y.size(0), device=device, dtype=Y.dtype) / Y.size(0)

            M = ot.dist(X_transformed, Y, metric='sqeuclidean')

            # Call the POT function
            # entropic_kl_uot_ti is the transport cost, which will be our loss
            _, log_uot_ti = ot.unbalanced.sinkhorn_unbalanced2(
                a, b, M,
                reg=args.faster_uot_reg,
                reg_m=args.faster_uot_reg_m_kl, # reg_m_kl in your snippet refers to reg_m here
                method="sinkhorn_translation_invariant",
                numItermax=100,
                stopThr=1e-9,
                log=True, # To get the log dictionary
                # reg_type="kl" # This seems to be default for sinkhorn_unbalanced if reg_m is for KL
            )
            D_loss = log_uot_ti['cost']

            D_loss.backward()
            D_opt.step()
        elif METHOD == 'usot':
            D_opt.zero_grad()
            X_transformed = D(X)
            mass_X = torch.ones(X_transformed.size(0), device=device) / X_transformed.size(0)
            mass_Y = torch.ones(Y.size(0), device=device) / Y.size(0)
            D_loss, _, _, _, _, _ = unbalanced_sliced_ot(
                mass_X, mass_Y, X_transformed, Y, 
                num_projections=args.L, p=2,
                rho1=args.rho1, rho2=args.rho2, 
                niter=10, mode='icdf'
            )
            D_loss.backward()
            D_opt.step()
        elif METHOD == 'suot':
            D_opt.zero_grad()
            X_transformed = D(X)
            mass_X = torch.ones(X_transformed.size(0), device=device)
            mass_Y = torch.ones(Y.size(0), device=device)
            D_loss, _, _, _, _, _ = sliced_unbalanced_ot(
                mass_X, mass_Y, X_transformed, Y,
                num_projections=args.L, p=2,
                rho1=args.rho1, rho2=args.rho2,
                niter=10, mode='icdf'
            )
            D_loss.backward()
            D_opt.step()
        elif METHOD == 'sopt':
            D_opt.zero_grad()
            X_transformed = D(X)
            D_loss = sopt(X_transformed.cpu(), Y.cpu(), n_proj=args.L, reg=args.sopt_reg)
            D_loss.backward()
            D_opt.step()
        elif METHOD == 'spot':
            D_opt.zero_grad()
            X_transformed = D(X)
            D_loss = spot(X_transformed.cpu(), Y[:args.spot_k].cpu(), n_proj=args.L)
            D_loss.backward()
            D_opt.step()
        elif METHOD == 'pawl':
            D_opt.zero_grad()
            X_transformed = D(X)
            D_loss = pawl(X_transformed.cpu(), Y.cpu(), n_proj=args.L, k=args.pawl_k)
            D_loss.backward()
            D_opt.step()
        
end_time = time.time()
training_time = end_time - start_time
peak_gpu_mem_bytes = 0
peak_gpu_mem_mb = 0
if use_cuda:
    peak_gpu_mem_bytes = torch.cuda.max_memory_allocated(device=device)
    peak_gpu_mem_mb = peak_gpu_mem_bytes / (1024**2) # Convert bytes to Megabytes


### Evaluation
X_test = X_test.to(device)
# Ensure model is in eval mode for consistent results if it has dropout/batchnorm etc.
if METHOD == 'uot-fm':
    D_test = uot_fm.evaluate(model, X_test, config)
else:
    D.eval() # Good practice, although ResidualMLP might not need it. ULightOT structure doesn't seem to have dropout/bn.
    with torch.no_grad(): # Disable gradient calculation for evaluation
        D_test = D(X_test) # shape (6544, 512)




mlp_classifier.eval()
target_mlp_classifier.eval()

with torch.no_grad(): # Disable gradient calculation for classifiers
    pred_labels = mlp_classifier(D_test)
    pred_labels = torch.round(pred_labels.squeeze())

    target_pred_labels = target_mlp_classifier(D_test)
    target_pred_labels = torch.round(target_pred_labels.squeeze())

pred_labels_np = pred_labels.detach().cpu().numpy()
target_pred_labels_np = target_pred_labels.detach().cpu().numpy()

## Load labels
# ... (label loading/comparison logic remains the same) ...
if INPUT_DATA == 'ADULT' or INPUT_DATA == 'YOUNG':
    actual_labels_np = np.where(x_data_test_gender == 'male', 1, 0)
    if INPUT_DATA == 'YOUNG':
        target_actual_labels_np = np.ones(len(x_data_test_gender)) # Use len for safety
    elif INPUT_DATA == 'ADULT':
        target_actual_labels_np = np.zeros(len(x_data_test_gender))
elif INPUT_DATA in ['MAN', 'WOMAN']:
    actual_labels_np = (x_data_test_age.reshape(-1) > 44)*1
    if INPUT_DATA == 'WOMAN':
        target_actual_labels_np = np.ones(len(x_data_test_age)) # Use len for safety
    elif INPUT_DATA == 'MAN':
        target_actual_labels_np = np.zeros(len(x_data_test_age))


accuracy = accuracy_score(actual_labels_np, pred_labels_np) # Swapped order for clarity (true, pred)
target_accuracy = accuracy_score(target_actual_labels_np, target_pred_labels_np) # Swapped order

print_stats(f"Accuracy: {accuracy:.4f}, Target accuracy: {target_accuracy:.4f}, Training Time: {training_time:.2f} s, Peak GPU Memory: {peak_gpu_mem_mb:.2f} MB", stats_file)

## W2 dist
# w2_S = w2(source_data, D_test)
# w2_T = w2(D_test, real_data)
# print_stats(f'W2(X_trans, Y): {w2_T}', stats_file)
# print_stats(f'W2(X, X_trans): {w2_S}', stats_file)

# model = load_model("ALAE/configs/ffhq.yaml", training_artifacts_dir="ALAE/training_artifacts/ffhq/")

# w2 = []
# target_size = 128 # Or 64, 256 - choose a size your memory can handle
# import torchvision.transforms.functional as TF

# for latent, mapped_latent in zip(X_test[:10], D_test[:10]):
#     # Decode original image
#     decoded_img_tensor = decode(model, latent.unsqueeze(0).cpu()) # Shape: (1, C, H, W)

#     # Decode mapped image
#     mapped_decoded_img_tensor = decode(model, mapped_latent.unsqueeze(0).cpu()) # Shape: (1, C, H, W)

#     # --- Resize the tensors ---
#     # Ensure tensors are float for interpolation
#     decoded_img_tensor = decoded_img_tensor.float()
#     mapped_decoded_img_tensor = mapped_decoded_img_tensor.float()

#     # Resize (using PyTorch functional transforms)
#     # Note: TF.resize expects (C, H, W) or (B, C, H, W)
#     decoded_img_resized = TF.resize(decoded_img_tensor, [target_size, target_size], interpolation=TF.InterpolationMode.BICUBIC)
#     mapped_decoded_img_resized = TF.resize(mapped_decoded_img_tensor, [target_size, target_size], interpolation=TF.InterpolationMode.BICUBIC)

#     # --- Convert to numpy arrays for ot.dist ---
#     # Process resized decoded image
#     decoded_img_np = ((decoded_img_resized * 0.5 + 0.5) * 255).type(torch.long).clamp(0, 255).cpu().type(torch.uint8).permute(0, 2, 3, 1).numpy().reshape(-1, 3)

#     # Process resized mapped image
#     mapped_decoded_img_np = ((mapped_decoded_img_resized * 0.5 + 0.5) * 255).type(torch.long).clamp(0, 255).cpu().type(torch.uint8).permute(0, 2, 3, 1).numpy().reshape(-1, 3)

#     # --- Calculate EMD on resized images ---
#     # Calculate distance matrix (now much smaller, e.g., (128*128, 128*128))
#     # Note: It's better practice to use float32/64 for distance calculations
#     M = ot.dist(mapped_decoded_img_np.astype(np.float64), decoded_img_np.astype(np.float64))

#     # Calculate EMD (assuming uniform distributions)
#     n_pixels_resized = target_size * target_size
#     a = np.ones((n_pixels_resized,)) / n_pixels_resized
#     b = np.ones((n_pixels_resized,)) / n_pixels_resized
#     w2_value = ot.emd2(a, b, M) # emd2 returns the EMD value directly

#     w2.append(w2_value)

# w2 = np.array(w2)
# print(f"Mean W2 (on {target_size}x{target_size} images): {np.mean(w2)}")


D_test = D_test.cpu().data.numpy().reshape(D_test.size(0), -1)
mu, sigma = np.mean(D_test, axis=0), np.cov(D_test, rowvar=False)
# Ensure sigma_data and sigma_source are calculated correctly earlier
FD_T = calculate_frechet_distance(mu, sigma, mu_data, sigma_data)
FD_S = calculate_frechet_distance(mu_source, sigma_source, mu, sigma)

print_stats(f'FD(X_trans, Y): {FD_T:.4f}, FD(X, X_trans): {FD_S:.4f}', stats_file)

## insert LPIPS calculation here
import lpips
lpips_alex = lpips.LPIPS(net='alex').to(device).eval()   # distance function
bs          = 32                                 # 32 in your script

# reuse the generator only once
gen = load_model("ALAE/configs/ffhq.yaml",
                 training_artifacts_dir="ALAE/training_artifacts/ffhq/").to(device).eval()

# ---------- LPIPS(source , transformed) -------------------
lp_keep_batches = []

with torch.no_grad():
    n = X_test.size(0)
    n = min(n, 100)             
    for i in tqdm(range(0, n, bs), desc="LPIPS keep", leave=False):
        # latents for this mini-batch
        z_src   = X_test[i:i+bs].to(device)                    # (bs,512)
        z_trans = D_test[i:i+bs].to(device) if torch.is_tensor(D_test) \
                   else torch.tensor(D_test[i:i+bs], device=device)

        # Option 1: decode -> RGB in [-1,1]
        img_src   = decode(gen, z_src)        # (bs,3,H,W)
        # Option 2: Load original images
        # img_src = torch.tensor(load_test_images(
        #     image_dir="./data/ffhq_test_images",
        #     indices_to_load=x_inds_test[i:i+bs],
        # ), device=device)
        img_trans = decode(gen, z_trans)

        # perceptual distance
        lp = lpips_alex(img_src, img_trans)   # (bs,1,1,1)
        lp_keep_batches.append(lp.squeeze())  # (bs,)

    # # Determine how many images to plot (up to 64 or batch size)
    # num_plot_samples = min(64, img_src.size(0))

    # # Plot comparison between source and transformed images
    # print(f"Plotting {num_plot_samples} source/transformed image pairs...")
    # # Create subplots ONCE before the loop
    # fig, axes = plt.subplots(2, num_plot_samples, figsize=(num_plot_samples * 2, 4))
    # # Handle case where num_plot_samples is 1 (axes is not a 2D array)
    # if num_plot_samples == 1:
    #     axes = np.array([[axes[0]], [axes[1]]])

    # for i in range(num_plot_samples):  # Iterate up to num_plot_samples
    #     # Source image (Top row)
    #     ax_src = axes[0, i]
    #     ax_src.imshow((img_src[i].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1))
    #     ax_src.set_title(f"Source {i}")
    #     ax_src.axis('off')

    #     # Transformed image (Bottom row)
    #     ax_trans = axes[1, i]
    #     ax_trans.imshow((img_trans[i].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1))
    #     ax_trans.set_title(f"Transformed {i}")
    #     ax_trans.axis('off')

    # plt.tight_layout()
    # plt.savefig(f'data/comparison_{METHOD}.png')
    # print(f"Saved comparison plot to comparison_{METHOD}.png")
    # plt.close(fig) # Close the specific figure

lpips_keep_avg = torch.cat(lp_keep_batches).mean().item()
print_stats(f"LPIPS (keep, avg): {lpips_keep_avg:.4f}", stats_file)

if args.plot:
    print("Saving Images")
    np.random.seed(1234)
    inds_to_map = np.random.choice(np.arange((x_inds_test < 300).sum()), size=10, replace=False)
    latent_to_map = torch.tensor(test_latents[x_inds_test[inds_to_map]], device=device)
    test_inp_images = np.load("data/test_images.npy")
    inp_images = test_inp_images[x_inds_test[inds_to_map]]
    with torch.no_grad():
        fig, _ = decode_and_plot(D, latent_to_map, inp_images, number_of_samples=1, device=device)
    base_dir = f"figures/{EXP_NAME}/{METHOD}"
    os.makedirs(base_dir, exist_ok=True)
    fig.savefig(f'{base_dir}/{RUN_NAME}.png')
    plt.close()

if args.wandb:
    print("Logging to wandb")
    run = wandb.init(project="uot-im2im", name=RUN_NAME, config=args) # Add run name and config
    # Build the table with new columns
    table_columns = ["method", "acc_keep", "acc_target", "fd(x, x_trans)", "fd(x_trans, y)", "lpips(x_trans, x)", "train_time_s", "peak_gpu_mem_mb"]
    if METHOD == 'twd':
        table_columns.extend(["ntrees", "nlines", "minY", "maxY", "scheduler"])
    elif METHOD == 'ulight':
        table_columns.extend(["tau"])
    elif METHOD == 'faster-uot':
        table_columns.extend(["reg", "reg_m_kl"])
    elif METHOD == 'usot':
        table_columns.extend(["rho1", "rho2"])
    elif METHOD == 'suot':
        table_columns.extend(["rho1", "rho2"])
    elif METHOD == 'sopt':
        table_columns.extend(["reg"])
    elif METHOD == 'spot':
        table_columns.extend(["k"])
    elif METHOD == 'pawl':
        table_columns.extend(["k"])

    table = wandb.Table(columns=table_columns)

    # Prepare data row
    data_row = [METHOD, accuracy, target_accuracy, FD_S, FD_T, lpips_keep_avg, training_time, peak_gpu_mem_mb]
    if METHOD == 'twd':
        data_row.extend([twd_ntrees, args.twd_nlines, args.twd_min_total_mass_Y, args.twd_max_total_mass_Y, args.twd_unbalanced_scheduler])
    elif METHOD == 'ulight':
        data_row.extend([args.ulight_tau])
    elif METHOD == 'faster-uot':
        data_row.extend([args.faster_uot_reg, args.faster_uot_reg_m_kl])
    elif METHOD == 'usot':
        data_row.extend([args.rho1, args.rho2])
    elif METHOD == 'suot':
        data_row.extend([args.rho1, args.rho2])
    elif METHOD == 'sopt':
        data_row.extend([args.sopt_reg])
    elif METHOD == 'spot':
        data_row.extend([args.spot_k])
    elif METHOD == 'pawl':
        data_row.extend([args.pawl_k])

    table.add_data(*data_row)

    # Log the table and individual summary metrics for easier viewing
    wandb.log({
        f"{EXP_NAME}_summary_table": table, # Log table under a specific key
        "accuracy_keep": accuracy,
        "accuracy_target": target_accuracy,
        "frechet_distance_source_transformed": FD_S,
        "frechet_distance_transformed_target": FD_T,
        "lpips_keep": lpips_keep_avg,
        "training_time_seconds": training_time,
        "peak_gpu_memory_mb": peak_gpu_mem_mb
    })
    run.finish()

# Close the stats file properly
stats_file.close()