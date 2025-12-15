import logging
import numpy as np
import collections
import os
import pickle as pkl
import sinter
from sympy.abc import x, y
from qldpc import abstract
from qldpc import codes
from qldpc import decoders
from qldpc.objects import Pauli

import sys
sys.path.append('..')

if __name__ == '__main__':
    benchmarks = [
        ('lp', 3),
        ('rqt', 6),
        ('rqt_di_54', 4),
        ('rqt_di_108', 4),
        ('surface', 3),
        ('surface', 5),
        ('surface', 7),
        ('surface', 9),
    ]
    max_iters = {
        ('lp', 3): 5,
        ('rqt', 6): 5,
        ('rqt_di_54', 4): 9,
        ('rqt_di_108', 4): 16,
        ('surface', 3): 5,
        ('surface', 5): 5,
        ('surface', 7): 5,
        ('surface', 9): 16,
    }
    p = 1e-3
    idle_strength_range = [3e-8, 7e-8, 3e-7, 7e-7, 3e-6, 7e-6, 3e-5, 7e-5, 3e-4, 7e-4, 3e-3]

    for name, d in benchmarks:
        tasks = []
        graphs = pkl.load(open(f'data/{name}/d_{d}/checkpoint_graphs.pkl', 'rb'))
        for idle_strength in idle_strength_range:
            tasks.append(sinter.Task(circuit=graphs[0].stim_circ(p, d, basis='Z', idle_strength=idle_strength), json_metadata={'idle': idle_strength, 'start': True, 'basis': 'Z'}))
            tasks.append(sinter.Task(circuit=graphs[0].stim_circ(p, d, basis='X', idle_strength=idle_strength), json_metadata={'idle': idle_strength, 'start': True, 'basis': 'X'}))
            final_idx = max_iters[(name, d)] - 1
            tasks.append(sinter.Task(circuit=graphs[final_idx].stim_circ(p, d, basis='Z', idle_strength=idle_strength), json_metadata={'idle': idle_strength, 'start': False, 'basis': 'Z'}))
            tasks.append(sinter.Task(circuit=graphs[final_idx].stim_circ(p, d, basis='X', idle_strength=idle_strength), json_metadata={'idle': idle_strength, 'start': False, 'basis': 'X'}))

        if name == 'surface':
            results = sinter.collect(
                num_workers=os.cpu_count(),
                tasks=tasks,
                decoders=['pymatching'],
                max_shots=20_000_000,
                max_errors=1_000,
                print_progress=False,)
        else:
            decoder_kwargs = {'with_BP_LSD': True, 'max_iter': 30, 'bp_method': "ms", 'lsd_method': 'lsd_cs', 'lsd_order': 0}
            results = sinter.collect(
                num_workers=os.cpu_count(),
                tasks=tasks,
                decoders=['custom'],
                max_shots=10_000_000,
                max_errors=100,
                custom_decoders={'custom': decoders.SinterDecoder(**decoder_kwargs)},)
            

        idle_data_start = np.zeros(len(idle_strength_range))
        idle_data_end = np.zeros(len(idle_strength_range))
        X_data_start = [(0, 0) for _ in range(len(idle_strength_range))]
        Z_data_start = [(0, 0) for _ in range(len(idle_strength_range))]
        X_data_end = [(0, 0) for _ in range(len(idle_strength_range))]
        Z_data_end = [(0, 0) for _ in range(len(idle_strength_range))]
        for result in results:
            idle = result.json_metadata['idle']
            basis = result.json_metadata['basis']
            start = result.json_metadata['start']
            idle_idx = idle_strength_range.index(idle)
            if basis == 'X':
                if start:
                    X_data_start[idle_idx] = (result.errors, result.shots)
                else:
                    X_data_end[idle_idx] = (result.errors, result.shots)
            else:
                if start:
                    Z_data_start[idle_idx] = (result.errors, result.shots)
                else:
                    Z_data_end[idle_idx] = (result.errors, result.shots)
        for idle_idx in range(len(idle_strength_range)):
            idle_data_start[idle_idx] = (X_data_start[idle_idx][0] + Z_data_start[idle_idx][0]) / (d * (X_data_start[idle_idx][1] + Z_data_start[idle_idx][1]))
            idle_data_end[idle_idx] = (X_data_end[idle_idx][0] + Z_data_end[idle_idx][0]) / (d * (X_data_end[idle_idx][1] + Z_data_end[idle_idx][1]))

        pkl.dump(idle_data_start, open(f'data/{name}/d_{d}/idle_data_start.pkl', 'wb'))
        pkl.dump(idle_data_end, open(f'data/{name}/d_{d}/idle_data_end.pkl', 'wb'))