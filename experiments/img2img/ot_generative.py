# import os
# import sys
# import time
# import traceback
# import argparse

# import numpy as np
# import torch
# import torch.distributed as dist
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import optim
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data import TensorDataset, DataLoader
# from torch.utils.data.distributed import DistributedSampler
# from tqdm import tqdm

# import wandb

# # --- Project imports ---
# sys.path.append("./")
# sys.path.append("./ALAE")

# from src.ot import SW  # Sliced Wasserstein
# from tsw import TSW, generate_trees_frames  # Tree Wasserstein
# from src.distributions import TensorSampler
# from src.fid_score import calculate_frechet_distance

# try:
#     from alae_ffhq_inference import load_model, decode
#     ALAE_AVAILABLE = True
# except ImportError:
#     print("Warning: ALAE components not found. Visualization will be skipped.")
#     ALAE_AVAILABLE = False
#     load_model = None
#     decode = None


# # --- Generator Definition ---
# class ResidualMLPBlock(nn.Module):
#     def __init__(self, dim, activation=nn.LeakyReLU(0.2, inplace=True)):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Linear(dim, dim),
#             nn.BatchNorm1d(dim),
#             activation,
#             nn.Linear(dim, dim),
#             nn.BatchNorm1d(dim),
#         )
#         self.final_activation = activation

#     def forward(self, x):
#         res = x
#         out = self.block(x)
#         out += res
#         return self.final_activation(out)


# class GeneratorResNetMLP(nn.Module):
#     def __init__(self, noise_dim, latent_dim, num_blocks=4, hidden_scale=2):
#         super().__init__()
#         working_dim = latent_dim * hidden_scale
#         activation = nn.LeakyReLU(0.2, inplace=True)

#         self.initial_layer = nn.Sequential(
#             nn.Linear(noise_dim, working_dim),
#             nn.BatchNorm1d(working_dim),
#             activation
#         )
#         self.blocks = nn.Sequential(
#             *[ResidualMLPBlock(working_dim, activation=activation) for _ in range(num_blocks)]
#         )
#         self.final_layer = nn.Linear(working_dim, latent_dim)

#         print(f"Initialized GeneratorResNetMLP: noise_dim={noise_dim}, latent_dim={latent_dim}, "
#               f"blocks={num_blocks}, hidden_scale={hidden_scale} -> working_dim={working_dim}")

#     def forward(self, z):
#         h = self.initial_layer(z)
#         h = self.blocks(h)
#         return self.final_layer(h)


# # --- Distributed helpers ---
# def setup(rank, world_size, backend='nccl'):
#     os.environ.setdefault('MASTER_ADDR', 'localhost')
#     os.environ.setdefault('MASTER_PORT', '12355')
#     print(f"Rank {rank}: initializing process group (backend={backend}, world_size={world_size})")
#     dist.init_process_group(backend, rank=rank, world_size=world_size)
#     torch.cuda.set_device(rank)
#     print(f"Rank {rank}: process group initialized, using CUDA device {rank}")


# def cleanup():
#     if dist.is_initialized():
#         print(f"Rank {dist.get_rank()}: destroying process group")
#         dist.destroy_process_group()


# def print_stats(msg, file_handle, rank):
#     if rank == 0:
#         print(msg)
#         if file_handle and not file_handle.closed:
#             file_handle.write(msg + "\n")
#             file_handle.flush()


# # --- FID calculation (rank 0 only) ---
# def calculate_and_log_fid(generator_model, noise_dim, n_samples, batch_size_fid,
#                           mu_real, sigma_real, device, stats_file_handle, current_epoch, use_wandb, rank):
#     if rank != 0:
#         return float('inf')

#     print_stats(f"\n--- Calculating FID at Epoch {current_epoch} ---", stats_file_handle, rank)

#     # unwrap DDP if needed
#     model = generator_model.module if isinstance(generator_model, DDP) else generator_model
#     model.eval()

#     # generate latents
#     all_gen = []
#     with torch.no_grad():
#         for i in range(0, n_samples, batch_size_fid):
#             bs = min(batch_size_fid, n_samples - i)
#             z = torch.randn(bs, noise_dim, device=device)
#             fake = model(z)
#             all_gen.append(fake.cpu().numpy())
#     gen_np = np.concatenate(all_gen, axis=0)

#     mu_gen = np.mean(gen_np, axis=0)
#     sigma_gen = np.cov(gen_np, rowvar=False)

#     try:
#         fid_score = calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
#         print_stats(f"FID Score (Epoch {current_epoch}): {fid_score:.4f}", stats_file_handle, rank)
#         if use_wandb:
#             wandb.log({"FID Score": fid_score, "epoch": current_epoch})
#     except Exception as e:
#         print_stats(f"Error computing FID Score at Epoch {current_epoch}: {e}", stats_file_handle, rank)
#         print_stats(traceback.format_exc(), stats_file_handle, rank)
#         fid_score = float('inf')

#     model.train()
#     print_stats(f"--- FID calculation complete (Epoch {current_epoch}) ---\n", stats_file_handle, rank)
#     return fid_score


# # --- Main training ---
# def main_worker(rank, world_size, args):
#     print(f"Starting main_worker on Rank {rank}/{world_size}")
#     setup(rank, world_size, args.dist_backend)
#     device = torch.device(f'cuda:{rank}')

#     # set seeds
#     torch.manual_seed(args.output_seed + rank)
#     np.random.seed(args.output_seed + rank)

#     # rank 0: setup logging dir and stats file
#     stats_file = None
#     OUTPUT_PATH = ''
#     full_exp_name = f'{args.exp_name}_{args.loss_type}_delta{args.delta}_std{args.std}_e{args.epochs}_lr{args.lr}_ws{world_size}'
#     if rank == 0:
#         OUTPUT_PATH = os.path.join('./data  ', full_exp_name)
#         os.makedirs(OUTPUT_PATH, exist_ok=True)
#         stats_file = open(os.path.join(OUTPUT_PATH, "results.txt"), "a", buffering=1)
#         print_stats(f"--- Experiment: {full_exp_name} ---", stats_file, rank)
#         print_stats(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}", stats_file, rank)
#         print_stats(f"Args: {vars(args)}", stats_file, rank)
#         print_stats(f"World Size: {world_size}", stats_file, rank)

#     # initialize wandb (rank 0 only)
#     if rank == 0 and args.use_wandb:
#         try:
#             wandb.init(
#                 project=args.wandb_project,
#                 entity=args.wandb_entity,
#                 name=args.wandb_run_name if args.wandb_run_name else full_exp_name,
#                 config=vars(args),
#                 save_code=True,
#                 reinit=True
#             )
#             print_stats("WandB initialized successfully.", stats_file, rank)
#         except Exception as e:
#             print_stats(f"Error initializing WandB: {e}. Disabling WandB logging.", stats_file, rank)
#             args.use_wandb = False
#     elif rank != 0:
#         args.use_wandb = False

#     # setup loss-specific objects
#     if args.loss_type in ('twd', 'twd_rp'):
#         if args.L % args.nlines != 0:
#             raise ValueError(f"L ({args.L}) must be divisible by nlines ({args.nlines}) for TWD.")
#         ntrees = args.L // args.nlines
#         TWD_obj = TSW(ntrees=ntrees, nlines=args.nlines, d=args.dim, delta=args.delta, ftype='linear', mass_division='distance_based', device=device)
#         print_stats(f"Using TWD with ntrees={ntrees}, nlines={args.nlines}", stats_file, rank)
#     else:
#         TWD_obj = None
#         print_stats(f"Using SW with L={args.L}", stats_file, rank)

#     # load data
#     try:
#         latents = np.load("data/latents.npy")
#         train_latents = latents[:60000]
#         print_stats(f"Loaded {train_latents.shape[0]} training latents", stats_file, rank)
#     except Exception as e:
#         print_stats(f"Error loading data: {e}", stats_file, rank)
#         if rank == 0 and args.use_wandb: wandb.finish(exit_code=1)
#         cleanup()
#         sys.exit(1)

#     # compute real stats for FID (rank 0 only)
#     mu_real = sigma_real = None
#     if rank == 0:
#         print_stats("Calculating real data stats for FID", stats_file, rank)
#         sampler = TensorSampler(torch.tensor(train_latents, dtype=torch.float32), device='cpu')
#         real_np = sampler.sample(min(args.fid_samples, train_latents.shape[0])).cpu().numpy()
#         mu_real = real_np.mean(axis=0)
#         sigma_real = np.cov(real_np, rowvar=False)
#         print_stats(f"Real data stats computed using {real_np.shape[0]} samples", stats_file, rank)
#         del sampler, real_np

#     # create DataLoader with DistributedSampler
#     X_train = torch.tensor(train_latents, dtype=torch.float32)
#     dataset = TensorDataset(X_train)
#     train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
#                                        shuffle=True, seed=args.output_seed)
#     batch_per_gpu = args.batch_size // world_size
#     train_loader = DataLoader(dataset,
#                               batch_size=batch_per_gpu,
#                               sampler=train_sampler,
#                               num_workers=args.num_workers,
#                               pin_memory=True,
#                               drop_last=True)
#     print_stats(f"DataLoader created (batch per GPU: {batch_per_gpu})", stats_file, rank)

#     # initialize model and optimizer
#     generator = GeneratorResNetMLP(
#         noise_dim=args.noise_dim,
#         latent_dim=args.dim,
#         num_blocks=args.gen_blocks,
#         hidden_scale=args.gen_hidden_scale
#     ).to(device)
#     generator = DDP(generator, device_ids=[rank], output_device=rank)
#     optimizer = optim.Adam(generator.parameters(), lr=args.lr)

#     # --- Cosine LR Scheduler ---
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#         optimizer,
#         T_max=args.epochs,
#         eta_min=0.0
#     )

#     if rank == 0 and args.use_wandb:
#         wandb.watch(generator, log='all', log_freq=args.log_interval * 10)

#     # --- Training Loop ---
#     print_stats(f"\n--- Starting Training: {args.epochs} epochs ---", stats_file, rank)
#     for epoch in range(args.start_epoch, args.epochs):
#         train_sampler.set_epoch(epoch)
#         print_stats(f"--- Epoch {epoch+1}/{args.epochs} ---", stats_file, rank)

#         data_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=(rank!=0))
#         epoch_losses = []
#         generator.train()

#         for i, (real_batch,) in enumerate(data_iter):
#             step = epoch * len(train_loader) + i + 1
#             real = real_batch.to(device, non_blocking=True)

#             optimizer.zero_grad()
#             z = torch.randn(real.size(0), args.noise_dim, device=device)
#             fake = generator(z)

#             if args.loss_type == 'sw':
#                 loss = SW(X=fake, Y=real, L=args.L, p=2, device=device)
#             elif args.loss_type == 'twd':
#                 mean_local = real.mean(dim=0)
#                 theta, intercept = generate_trees_frames(
#                     ntrees=ntrees, nlines=args.nlines, d=args.dim,
#                     mean=mean_local, std=args.std,
#                     gen_mode='gaussian_raw', device=device
#                 )
#                 loss = TWD_obj(fake, real, theta, intercept)
#             else:  # twd_rp
#                 progress = step / (args.epochs * len(train_loader))
#                 kappa = (1 - progress) ** 10 * (100.0 - 1.0) + 1.0
#                 mean_local = real.mean(dim=0)
#                 theta, intercept = generate_trees_frames(
#                     ntrees=ntrees, nlines=args.nlines, d=args.dim,
#                     mean=mean_local, std=args.std,
#                     gen_mode='random_path', device=device,
#                     kappa=kappa, X=fake.detach(), Y=real.detach()
#                 )
#                 loss = TWD_obj(fake, real, theta, intercept)

#             loss.backward()
#             optimizer.step()
#             epoch_losses.append(loss.item())

#             if rank == 0 and step % args.log_interval == 0:
#                 avg_loss = np.mean(epoch_losses[-args.log_interval:])
#                 print_stats(f"Step {step}: loss={loss.item():.4f}, avg_loss={avg_loss:.4f}", stats_file, rank)
#                 if args.use_wandb:
#                     wandb.log({
#                         "step_loss": loss.item(),
#                         "avg_loss_interval": avg_loss,
#                         "epoch": epoch+1
#                     }, step=step)

#         # --- End of Epoch: Logging, FID, Checkpoint ---
#         avg_epoch_loss = np.mean(epoch_losses)
#         if rank == 0:
#             print_stats(f"Epoch {epoch+1} summary: avg_loss={avg_epoch_loss:.4f}", stats_file, rank)
#             if args.use_wandb:
#                 wandb.log({"epoch_loss": avg_epoch_loss, "epoch": epoch+1}, step=step)

#             is_last = (epoch+1 == args.epochs)
#             do_fid = (args.fid_interval > 0 and (epoch+1) % args.fid_interval == 0)
#             do_save = (args.save_interval > 0 and (epoch+1) % args.save_interval == 0)

#             if do_fid or is_last:
#                 calculate_and_log_fid(
#                     generator_model=generator,
#                     noise_dim=args.noise_dim,
#                     n_samples=args.fid_samples,
#                     batch_size_fid=args.batch_size,
#                     mu_real=mu_real,
#                     sigma_real=sigma_real,
#                     device=device,
#                     stats_file_handle=stats_file,
#                     current_epoch=epoch+1,
#                     use_wandb=args.use_wandb,
#                     rank=rank
#                 )

#             if do_save or is_last:
#                 ckpt_path = os.path.join(OUTPUT_PATH, f'generator_epoch_{epoch+1}.pth')
#                 ckpt = {
#                     'epoch': epoch+1,
#                     'model_state_dict': generator.module.state_dict(),
#                     'optimizer_state_dict': optimizer.state_dict(),
#                     'loss': epoch_losses[-1],
#                     'args': vars(args)
#                 }
#                 torch.save(ckpt, ckpt_path)
#                 print_stats(f"Checkpoint saved: {ckpt_path}", stats_file, rank)
#                 if args.use_wandb:
#                     artifact = wandb.Artifact(f'{full_exp_name}-ckpt-epoch-{epoch+1}', type='model')
#                     artifact.add_file(ckpt_path)
#                     wandb.log_artifact(artifact)

#         # --- Step Cosine LR scheduler ---
#         scheduler.step()
#         lr_now = scheduler.get_last_lr()[0]
#         print_stats(f"Learning rate updated to: {lr_now:.6e}", stats_file, rank)
#         if rank == 0 and args.use_wandb:
#             wandb.log({'learning_rate': lr_now}, step=step+1)

#     # --- Final Visualization (rank 0 only) ---
#     if rank == 0:
#         if args.plot and ALAE_AVAILABLE:
#             print_stats("Generating sample images...", stats_file, rank)
#             try:
#                 cfg = "./ALAE/configs/ffhq.yaml"
#                 arts = "./ALAE/training_artifacts/ffhq/"
#                 alae_model = load_model(cfg, training_artifacts_dir=arts)
#                 alae_model.eval().to('cpu')

#                 generator.module.eval().to(device)
#                 n_vis = 16
#                 z_vis = torch.randn(n_vis, args.noise_dim, device=device)
#                 with torch.no_grad():
#                     latents_vis = generator.module(z_vis).cpu()

#                 decoded = []
#                 for i in range(n_vis):
#                     img_t = decode(alae_model, latents_vis[i].unsqueeze(0))
#                     img = img_t.squeeze(0).permute(1,2,0).numpy()
#                     img = (img * 0.5 + 0.5).clip(0,1)
#                     decoded.append(img)

#                 import matplotlib.pyplot as plt
#                 cols = 4
#                 rows = (n_vis + cols - 1) // cols
#                 fig, axes = plt.subplots(rows, cols, figsize=(cols*2.5, rows*2.5), dpi=150)
#                 axes = axes.flatten()
#                 for i, im in enumerate(decoded):
#                     axes[i].imshow(im)
#                     axes[i].axis('off')
#                 for j in range(i+1, len(axes)):
#                     axes[j].axis('off')
#                 plt.tight_layout(pad=0.1)
#                 img_path = os.path.join(OUTPUT_PATH, f'{full_exp_name}_samples_epoch{args.epochs}.png')
#                 fig.savefig(img_path)
#                 plt.close(fig)
#                 print_stats(f"Saved sample grid to {img_path}", stats_file, rank)
#                 if args.use_wandb:
#                     wandb.log({"Generated Samples": wandb.Image(img_path)}, step=step)
#             except Exception as e:
#                 print_stats(f"Visualization error: {e}", stats_file, rank)
#                 print_stats(traceback.format_exc(), stats_file, rank)
#         else:
#             print_stats("Skipping visualization (ALAE not available)", stats_file, rank)

#         if args.use_wandb:
#             wandb.finish()
#         if stats_file and not stats_file.closed:
#             print_stats(f"--- Script finished: {full_exp_name} ---", stats_file, rank)
#             stats_file.close()

#     cleanup()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser('Unconditional Latent Generation using SW/TWD')
#     # General Args
#     parser.add_argument('--plot', action='store_true', default=False, help='Save generated image grid (rank 0 only)')
#     parser.add_argument('--dim', type=int, default=512, help='Latent dimension')
#     parser.add_argument('--noise_dim', type=int, default=512, help='Dimension of input noise')
#     parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for generator')
#     parser.add_argument('--epochs', type=int, default=1800, help='Number of training epochs')
#     parser.add_argument('--start_epoch', type=int, default=0, help='Epoch to start training from (for resuming)')
#     parser.add_argument('--batch_size', type=int, default=256, help='Total training batch size across all GPUs')
#     parser.add_argument('--output_seed', type=int, default=42, help='Base random seed')
#     parser.add_argument('--exp_name', type=str, default='gen_dist', help='Base experiment name')
#     parser.add_argument('--save_interval', type=int, default=-1, help='Save model checkpoint every N epochs (rank 0 only)')
#     parser.add_argument('--fid_interval', type=int, default=25, help='Calculate FID every N epochs (rank 0 only)')
#     parser.add_argument('--fid_samples', type=int, default=60000, help='Number of samples for FID calculation (rank 0 only)')
#     parser.add_argument('--log_interval', type=int, default=1000, help='Log step loss every N steps (rank 0 only)')
#     parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader workers per GPU')

#     # Loss Specific Args
#     parser.add_argument('--loss_type', type=str, choices=['sw', 'twd', 'twd_rp'], default='sw', help='Type of loss function')
#     parser.add_argument('--L', type=int, default=100, help='Projections (SW) or total lines (TWD)')
#     parser.add_argument('--nlines', type=int, default=4, help='Lines per tree for TWD')
#     parser.add_argument('--delta', type=float, default=1, help='Delta for TWD')
#     parser.add_argument('--std', type=float, default=0.01, help='Standard deviation for TWD')

#     # Generator Architecture Args
#     parser.add_argument('--gen_blocks', type=int, default=1, help='Number of residual blocks in Generator')
#     parser.add_argument('--gen_hidden_scale', type=int, default=2, help='Hidden scale factor in Generator')

#     # WandB Args
#     parser.add_argument('--use_wandb', action='store_true', default=False, help='Enable WandB logging (rank 0 only)')
#     parser.add_argument('--wandb_project', type=str, default='img2img-distributed', help='WandB project name')
#     parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity')
#     parser.add_argument('--wandb_run_name', type=str, default=None, help='Custom WandB run name')

#     # Distributed Args
#     parser.add_argument('--local_rank', type=int, default=int(os.environ.get('LOCAL_RANK', 0)),
#                         help='Local rank. Passed by torchrun.')
#     parser.add_argument('--dist_backend', type=str, default='nccl', choices=['nccl', 'gloo'],
#                         help='Distributed backend')

#     args = parser.parse_args()

#     world_size = int(os.environ.get('WORLD_SIZE', 1))
#     rank = int(os.environ.get('RANK', 0))

#     print(f"Launcher detected: world_size={world_size}, rank={rank}")

#     if world_size > 1 and not torch.cuda.is_available():
#         print("Error: Distributed training requested but CUDA is not available.")
#         sys.exit(1)
#     if world_size > 1 and torch.cuda.device_count() < world_size:
#         print(f"Error: World size ({world_size}) exceeds available GPUs ({torch.cuda.device_count()})")

#     if world_size > 1:
#         main_worker(args.local_rank, world_size, args)
#     else:
#         print("Running in single-process mode")
#         main_worker(0, 1, args)
