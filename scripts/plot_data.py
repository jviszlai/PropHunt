import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

import sys
sys.path.append('..')

BASE_PATH = 'data'

intro_data = pkl.load(open(f'{BASE_PATH}/surface/d_5/intro_figure_data.pkl', 'rb'))
p_range = [1e-3, 3e-3, 5e-3, 7e-3]

plt.figure(figsize=(3, 2))
x_lim = (8*10**-4, 10**-2)
y_lim = (5*10**-6, 5*10**-2)
plt.plot(p_range, intro_data['good_depth'], marker='d', color='tab:green')
plt.plot(p_range, intro_data['bad_depth'], marker='d', color='tab:red')
plt.loglog()
plt.xlim(*x_lim)
plt.ylim(*y_lim)
plt.text(2e-3, 3e-5, 'depth=8', color='tab:green')
plt.text(1.2e-3, 2e-3, 'depth=4', color='tab:red')
plt.xlabel('Physical Error Rate')
plt.ylabel('Logical Error Rate')
plt.savefig('figures/intro_a_depth.pdf', bbox_inches='tight')

plt.figure(figsize=(3, 2))
x_lim = (8*10**-4, 10**-2)
y_lim = (5*10**-6, 5*10**-2)
plt.plot(p_range, intro_data['good_deff'], marker='d', color='tab:green')
plt.plot(p_range, intro_data['bad_deff'], marker='d', color='tab:red')
plt.loglog()
plt.xlim(*x_lim)
plt.ylim(*y_lim)
plt.text(2e-3, 3e-5, 'd_eff=5', color='tab:green')
plt.text(1.2e-3, 8e-4, 'd_eff=5', color='tab:red')
plt.xlabel('Physical Error Rate')
plt.ylabel('Logical X Error Rate')
plt.savefig('figures/intro_b_d_eff.pdf', bbox_inches='tight')

benchmarks = [
    ('surface', 3),
    ('surface', 5),
    ('surface', 7),
    ('surface', 9),
    ('lp', 3),
    ('rqt', 6),
    ('rqt_di_54', 4),
    ('rqt_di_108', 4),
]

p_range_lookup = {
    benchmark[0]: [5e-4, 1e-3, 3e-3, 5e-3, 7e-3] if benchmark[0] != 'surface' else [1e-3, 3e-3, 5e-3, 7e-3, 9e-3, 1e-2]
    for benchmark in benchmarks
}

max_iter_lookup = {
    ('surface', 3): 5,
    ('surface', 5): 5,
    ('surface', 7): 5,
    ('surface', 9): 16,
    ('lp', 3): 5,
    ('rqt', 6): 5,
    ('rqt_di_54', 4): 9,
    ('rqt_di_108', 4): 16,
}

benchmark_name_lookup = {
    ('surface', 3): '[[9, 1, 3]]',
    ('surface', 5): '[[25, 1, 5]]',
    ('surface', 7): '[[49, 1, 7]]',
    ('surface', 9): '[[81, 1, 9]]',
    ('lp', 3): '[[39, 3, 3]]',
    ('rqt', 6): '[[60, 2, 6]]',
    ('rqt_di_54', 4): '[[54, 11, 4]]',
    ('rqt_di_108', 4): '[[108, 18, 4]]',
}

opt_data_lookup = {
    ('surface', d): pkl.load(open(f'{BASE_PATH}/surface/d_{d}/hand_opt_data.pkl', 'rb')) for d in [3, 5, 7, 9]
}

idle_data_start_lookup = {
    (name, d): pkl.load(open(f'{BASE_PATH}/{name}/d_{d}/idle_data_start.pkl', 'rb')) for name, d in benchmarks
}

idle_data_end_lookup = {
    (name, d): pkl.load(open(f'{BASE_PATH}/{name}/d_{d}/idle_data_end.pkl', 'rb')) for name, d in benchmarks
}

data_lookup = {
    benchmark: pkl.load(open(f'{BASE_PATH}/{benchmark[0]}/d_{benchmark[1]}/iter_data.pkl', 'rb'))
    for benchmark in benchmarks
}

graph_lookup = {
    benchmark: pkl.load(open(f'{BASE_PATH}/{benchmark[0]}/d_{benchmark[1]}/checkpoint_graphs.pkl', 'rb'))
    for benchmark in benchmarks
}


plt.rcParams.update({'font.size': 12})
fig, axes = plt.subplots(2, 4, figsize = (3 * 4, 2 * 2))
plt.subplots_adjust(hspace=0.6,wspace=0.3)
i = 0
for (name, d) in benchmarks:
    idx_0 = i // 4
    idx_1 = i % 4
    max_iter = max_iter_lookup[(name, d)]
    alpha_range = np.logspace(0, -1.5, max_iter)
    dataset = data_lookup[(name, d)]
    p_range = p_range_lookup[name]
    for iter, data in dataset.items():
        if iter == max_iter:
            break
        if iter == 0:
            axes[idx_0][idx_1].plot(p_range, dataset[iter], marker = 'o', markerfacecolor='tab:red', color='tab:red', label=f'Coloration Circuit\n(Optimization Start)')
        else:
            axes[idx_0][idx_1].fill_between(p_range, dataset[iter], dataset[iter - 1], alpha = 0.5 * alpha_range[iter], color='tab:green', label=f'Intermediate\nSM Circuits' if iter == 1 else None)
            if iter == max_iter - 1:
                axes[idx_0][idx_1].plot(p_range, dataset[iter], marker = '*',markersize=8, markerfacecolor=to_rgba('tab:green', 0.8), linewidth=0.5, color='tab:green', zorder=3, label=f'PropHunt\n(Optimization End)')
    if name == 'surface':
        axes[idx_0][idx_1].plot(p_range, opt_data_lookup[(name, d)], marker='s',markersize=8, markerfacecolor='none', linewidth=0.5, markeredgewidth=1.5, color='tab:blue', label='Hand-Designed Circuit')
    axes[idx_0][idx_1].set_title(benchmark_name_lookup[(name, d)])
    axes[idx_0][idx_1].loglog()
    i += 1
axes[0, 3].legend(loc='lower left', bbox_to_anchor=(1.1, -1.1))
plt.suptitle('QEC Code Benchmark Performance', y=1.02, fontsize=14)
axes[1, 2].set_xlabel('Physical Error Rate')
axes[0, 0].set_ylabel('Logical Error Rate')
axes[0, 0].yaxis.set_label_coords(-0.3, -0.25)
plt.savefig('figures/benchmarks.pdf', bbox_inches='tight')

marker_lookup = {
    ('surface', 3): 'o',
    ('surface', 5): 'd',
    ('surface', 7): 'X',
    ('surface', 9): 's',
    ('lp', 3): 'P',
    ('rqt', 6): '*',
    ('rqt_di_54', 4): '^',
    ('rqt_di_108', 4): 'v',
}

idle_strength_range = [3e-8, 7e-8, 3e-7, 7e-7, 3e-6, 7e-6, 3e-5, 7e-5, 3e-4, 7e-4, 3e-3]

fig, axes = plt.subplots(5, 1, sharex=True, figsize=(13, 11), height_ratios=[4, 1, 1, 1, 1])

for ax in axes:
    ax.axvline(300e-9 / 10, linestyle='dashed', color='black') # 300 ns gate, 10 s coherence
    ax.axvline(1e-3 / 10, linestyle='dashed', color='black') # 1 ms gate, 10 s coherence
    ax.axvline(36e-9 / 80e-6, linestyle='dashed', color='black') # 36 ns gate, 80 us coherence

axes[0].text(200e-9 / 10, 3e-2, 'Neutral Atoms')
axes[0].text(7e-5 / 10, 3e-2, 'Neutral Atoms with Movement')
axes[0].text(35e-9 / 80e-6, 3e-2, 'Superconducting')

axes[0].plot([], [], linewidth=2, color='tab:red', label='Coloration\nCircuit')
axes[0].plot([], [], linewidth=2, color='tab:green', label='PropHunt')

for i, (name, d) in enumerate(benchmarks):
    idle_data_start = idle_data_start_lookup[(name, d)]
    idle_data_end = idle_data_end_lookup[(name, d)]
    if name == 'surface':
        idx = 0
    else:
        idx = i - 3
    axes[idx].plot([], [], marker=marker_lookup[(name, d)], linestyle='none', color='black', label=benchmark_name_lookup[(name, d)])
    axes[idx].plot(idle_strength_range, idle_data_start, color='tab:red', marker=marker_lookup[(name, d)], markersize=8, markerfacecolor='tab:red', linewidth=0.5)
    axes[idx].plot(idle_strength_range, idle_data_end, color='tab:green', marker=marker_lookup[(name, d)], markersize=9, markerfacecolor='tab:green', linewidth=0.5)

    axes[idx].loglog()

axes[0].legend(loc='upper left')
axes[1].legend(loc='upper left')
axes[2].legend(loc='upper left')
axes[3].legend(loc='upper left')
axes[4].legend(loc='upper left')

axes[-1].set_xlabel('Idle Error Strength')
axes[1].set_ylabel('Logical Error Rate')
fig.suptitle('Idle Sensitivity Study', y=0.92, fontsize=16)


plt.savefig('figures/idle_sensitivity.pdf', bbox_inches='tight')

plt.figure()

zne_data = pkl.load(open('data/zne/zne_expectation_data.pkl', 'rb'))

d_range = [13, 11, 9]

for i in range(3):
    if i == 0:
        bar_container = plt.bar(i - 0.2, abs(1 - zne_data['baseline'][0][i]), 0.4, align='center', color='tab:blue', hatch='//', label='DS-ZNE')
    else:
        bar_container = plt.bar(i - 0.2, abs(1 - zne_data['baseline'][0][i]), 0.4, align='center', color='tab:blue', hatch='//')
    plt.bar_label(bar_container, fmt='{:.3f}')
for i in range(3):
    if i == 0:
        bar_container = plt.bar(i + 0.2, abs(1 - zne_data['hook'][0][i]), 0.4, align='center', color='tab:orange', hatch='\\\\', label='Hook-ZNE')
    else:
        bar_container = plt.bar(i + 0.2, abs(1 - zne_data['hook'][0][i]), 0.4, align='center', color='tab:orange', hatch='\\\\')
    plt.bar_label(bar_container, fmt='{:.3f}')
plt.xticks([0, 1, 2], [13, 11, 9])
plt.xlabel('Maximum Code Distance')
plt.ylabel('Error in Estimated Expection \n (L1-norm)')
plt.legend(loc='upper left')
plt.grid(which='major', axis='y')

plt.savefig('figures/zne_expectations.pdf', bbox_inches='tight')
