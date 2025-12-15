import pickle as pkl
import sinter
import numpy as np
import os

import sys
sys.path.append('..')

from prop_hunt.prop_graph import PropagationGraph

d_range = [3, 5, 7, 9]
p_range = [1e-3, 3e-3, 5e-3, 7e-3, 9e-3, 1e-2]

if __name__ == '__main__':
    tasks = []
    opt_surface_codes = pkl.load(open('hand_opt_surface_codes.pkl', 'rb'))
    for d in d_range:
        patch = opt_surface_codes[d]
        prop_graph = PropagationGraph({anc.idx: [data.idx for data in anc.data_qubits if data] for anc in patch.z_ancilla},
                                {anc.idx: [data.idx for data in anc.data_qubits if data] for anc in patch.x_ancilla},
                                [[data.idx for data in patch.logical_z_qubits]],
                                [[data.idx for data in patch.logical_x_qubits]])

        for p in p_range:
            tasks.append(sinter.Task(circuit=prop_graph.stim_circ(p, d, basis='Z'), json_metadata={'p': p, 'basis': 'Z'}))
            tasks.append(sinter.Task(circuit=prop_graph.stim_circ(p, d, basis='X'), json_metadata={'p': p, 'basis': 'X'}))

        results = sinter.collect(
            num_workers=os.cpu_count(),
            tasks=tasks,
            decoders=['pymatching'],
            max_shots=20_000_000,
            max_errors=1_000,
            print_progress=False,)

        iter_data = np.zeros(len(p_range))
        X_data = [(0, 0) for _ in range(len(p_range))]
        Z_data = [(0, 0) for _ in range(len(p_range))]
        for result in results:
            p = result.json_metadata['p']
            basis = result.json_metadata['basis']
            p_idx = p_range.index(p)
            if basis == 'X':
                X_data[p_idx] = (result.errors, result.shots)
            else:
                Z_data[p_idx] = (result.errors, result.shots)

        for p_idx in range(len(p_range)):
            if X_data[p_idx][1] == 0:
                continue
            iter_data[p_idx] = (X_data[p_idx][0] + Z_data[p_idx][0]) / (d * (X_data[p_idx][1] + Z_data[p_idx][1]))

        os.makedirs(f'data/surface/d_{d}', exist_ok=True)

        pkl.dump(iter_data, open(f'data/surface/d_{d}/hand_opt_data.pkl', 'wb'))