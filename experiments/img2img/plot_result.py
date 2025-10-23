import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# Data (same)

uot_eps = [0.05, 0.1, 0.2, 0.25]
uot_acc = [76.773, 82.381, 80.043, 78.515]
uot_lpips = [0.44468, 0.47127, 0.49235, 0.49785]

ulight_tau   = [50, 100, 250, 1000, 10000]
ulight_acc   = [70.39, 77.51, 82.7, 85.27, 86.064]
ulight_lpips = [0.3706, 0.391, 0.4086, 0.4198, 0.42328]

ptsw_param = [1.2, 1.1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
ptsw_acc   = [84.8716381, 85.7121027, 91.0605134, 94.00978, 95.9810513,
              96.5922983, 97.0507335, 98.5330073, 99.1136919, 99.4804401]
ptsw_lpips = [0.40088, 0.40466, 0.41834, 0.4282, 0.43937,
              0.44925, 0.45895, 0.47712, 0.49022, 0.50780]

sw_acc, sw_lpips = [89.884], [0.40741]
db_acc, db_lpips = [89.96], [0.40675]

plt.figure(figsize=(6,6))

s = 60                                # slightly larger
lw = 1.3                              # line-width for the 'x'

plt.scatter(sw_acc,   sw_lpips,   s=s, marker='x', linewidths=lw,
            label='SW',         color='purple',   alpha=0.85)
plt.scatter(db_acc,   db_lpips,  s=s, marker='x', linewidths=lw,
            label='Db-TSW',     color='green',    alpha=0.85)
plt.scatter(uot_acc,  uot_lpips, s=s, marker='x', linewidths=lw,
            label='UOT-FM',     color='goldenrod',alpha=0.85)
plt.scatter(ulight_acc, ulight_lpips, s=s, marker='x', linewidths=lw,
            label='ULight',     color='tab:blue', alpha=0.85)
plt.scatter(ptsw_acc,  ptsw_lpips, s=s, marker='x', linewidths=lw,
            label='PartialTSW', color='tab:orange',alpha=0.85)

x_off = 0.3
for x,y,eps in zip(uot_acc, uot_lpips, uot_eps):
    plt.text(x+x_off, y, rf'$\lambda={eps}$', fontsize=6, va='center', ha='left', color='goldenrod')

for x, y, nu in zip(ptsw_acc, ptsw_lpips, ptsw_param):
    if nu < 1:                                 # 0.2 … 0.9  ➜ below
        plt.text(x, y-0.002,                  # a small downward offset
                 rf'$\nu(\mathcal{{T}})={nu}$',
                 fontsize=6, ha='center', va='top', color='tab:orange')
    else:                                      # 1.1 and 1.2  ➜ keep right
        plt.text(x + x_off, y,
                 rf'$\nu(\mathcal{{T}})={nu}$',
                 fontsize=6, ha='left', va='center', color='tab:orange')


for x,y,tau in zip(ulight_acc, ulight_lpips, ulight_tau):
    if tau==10000:
        plt.text(x, y+0.006, rf'$\tau={tau}$', fontsize=6, ha='center', va='bottom', color='tab:blue')
    else:
        plt.text(x+x_off, y, rf'$\tau={tau}$', fontsize=6, va='center', ha='left', color='tab:blue')

plt.xlabel(r'Accuracy ($\%$) $\uparrow$')
plt.ylabel(r'LPIPS $\downarrow$')
plt.xlim(70,100)
plt.ylim(0.525,0.35)
plt.gca().invert_yaxis()
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax = plt.gca()
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

# legend at top center inside
order = ['SW','Db-TSW','UOT-FM','ULight','PartialTSW']
handles, labels = plt.gca().get_legend_handles_labels()
handles = [handles[labels.index(l)] for l in order]
plt.legend(handles, order, frameon=False, loc='upper center', bbox_to_anchor=(0.5,1.03), ncol=3)

plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
plt.tight_layout()
plt.savefig('figures/uot_alae_results.pdf')
plt.show()
