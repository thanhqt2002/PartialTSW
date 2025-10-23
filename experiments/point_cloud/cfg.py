import argparse

def parse_args():
    parser = argparse.ArgumentParser('point cloud parameters')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--method', default='twd')
    parser.add_argument('--num_epoch', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)

    # --- MODIFIED choices for loss_type ---
    parser.add_argument('--loss_type', type=str, default='sw',
                        choices=['sw', 'twd', 'usot', 'suot', 'sopt', 'spot', 'pawl', 'pot'])
    parser.add_argument('--sw_projections', type=int, default=10)
    parser.add_argument('--sw_p', type=int, default=2, help="Power for SW and also for p-norm in POT cost if not sqeuclidean")
    # TWD args
    parser.add_argument('--twd_ntrees', type=int, default=5)
    parser.add_argument('--twd_nlines', type=int, default=2)
    parser.add_argument('--twd_ftype', type=str, default='linear', choices=['linear', 'poly', 'circular', 'pow', 'circular_concentric'])
    parser.add_argument('--twd_mass_division', type=str, default='distance_based', choices=['uniform', 'distance_based'])
    parser.add_argument('--twd_delta', type=float, default=2.0)
    parser.add_argument('--twd_unbalanced', action='store_true', default=False, help='Use unbalanced TWD')
    parser.add_argument('--min_mass_generated', type=float, default=1.0)
    parser.add_argument('--max_mass_generated', type=float, default=1.0)
    parser.add_argument('--twd_unbalanced_scheduler', type=str, default='constant', choices=['constant', 'linear_increasing', 'linear_decreasing', 'cosine_increasing', 'cosine_decreasing', 'cyclic_cosine', 'reversed_cyclic_cosine'])
    parser.add_argument('--twd_num_cycles', type=int, default=0)
    # USOT/SUOT args
    parser.add_argument('--rho1', type=float, default=0.01)
    parser.add_argument('--rho2', type=float, default=1)
    # SOPT args
    parser.add_argument('--sopt_reg', type=float, default=1)
    # SPOT args
    parser.add_argument('--spot_k', type=int, default=10)
    # --- POT ARGS ---
    parser.add_argument('--pot_reg', type=float, default=0.1, help="Entropic regularization for POT Sinkhorn")
    parser.add_argument('--pot_reg_m_kl', type=float, default=1.0, help="Marginal KL regularization for POT Unbalanced Sinkhorn")
    parser.add_argument('--pot_num_iter_max', type=int, default=100, help="Max iterations for POT Sinkhorn") 
    parser.add_argument('--pot_stop_thr', type=float, default=1e-9, help="Stop threshold for POT Sinkhorn")
    parser.add_argument('--pot_cost_metric_p', type=int, default=2, help="Power p for the cost metric ||x-y||^p. p=2 is sq Euclidean if metric='sqeuclidean', p=1 is L1 if metric='euclidean'.")
    # PAWL args
    parser.add_argument('--pawl_k', type=int, default=10, help="Number of points for PAWL")

    parser.add_argument('--plot_style', type=int, default=0)
    args = parser.parse_args()

    return args

def get_cfg_name(args):
    meta = f"seed-{args.seed}_ep-{args.num_epoch}_lr-{args.lr}"
    if args.method == 'twd':
        name = f"mmass-{args.max_mass_generated}"
    elif args.method == 'pot':
        name = f"reg-{args.pot_reg}_reg_mkl-{args.pot_reg_m_kl}"
    elif args.method == 'pawl':
        name = f"k-{args.pawl_k}"
    elif args.method == 'usot':
        name = f"rho1-{args.rho1}_rho2-{args.rho2}"
    elif args.method == 'suot':
        name = f"rho1-{args.rho1}_rho2-{args.rho2}"
    elif args.method == 'sopt':
        name = f"reg-{args.sopt_reg}"
    elif args.method == 'spot':
        name = f"k-{args.spot_k}"
    elif args.method == 'sw':
        name = "nproj-100"
        
    return name + "_" + meta