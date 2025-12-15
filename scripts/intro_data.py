import sys
sys.path.append('..')

from prop_hunt.surface_code import CustomScheduleMemoryPatch
from prop_hunt.prop_graph import PropagationGraph
import sinter
import numpy as np
import pickle as pkl

def surface_code(d: int, try_depth=False, bad=False):
    patch = CustomScheduleMemoryPatch(d, d, d, apply_idle_during_gates=False)
    if bad:
        for anc in patch.z_ancilla:
            anc.data_qubits = [anc.data_qubits[0], anc.data_qubits[2], anc.data_qubits[1], anc.data_qubits[3]]
        for anc in patch.x_ancilla:
            anc.data_qubits = [anc.data_qubits[0], anc.data_qubits[2], anc.data_qubits[1], anc.data_qubits[3]]
    
    if try_depth:
        sched = {}
        for t in range(4):
            gates = []
            for anc in patch.z_ancilla:
                if anc.data_qubits[t]:
                    gates.append((anc.data_qubits[t].idx, anc.idx))
            for anc in patch.x_ancilla:
                if anc.data_qubits[t]:
                    gates.append((anc.idx, anc.data_qubits[t].idx))
            sched[t] = ('CX', gates)
    else:
        sched = {}
        for t in range(8):
            gates = []
            if t < 4:
                for anc in patch.z_ancilla:
                    if anc.data_qubits[t]:
                        gates.append((anc.data_qubits[t].idx, anc.idx))
            else:
                for anc in patch.x_ancilla:
                    if anc.data_qubits[t - 4]:
                        gates.append((anc.idx, anc.data_qubits[t - 4].idx))
            sched[t] = ('CX', gates)
    patch.set_custom_schedule(sched)
    return patch


if __name__ == '__main__':
    p_range = [1e-3, 3e-3, 5e-3, 7e-3]

    bad_patch = surface_code(5, True, True)
    good_patch = surface_code(5, False, False)

    bad_deff = pkl.load(open('intro_deff_graph.pkl', 'rb'))
    opt_surface = pkl.load(open('hand_opt_surface_codes.pkl', 'rb'))[5]
    good_deff = PropagationGraph({anc.idx: [data.idx for data in anc.data_qubits if data] for anc in opt_surface.z_ancilla},
                            {anc.idx: [data.idx for data in anc.data_qubits if data] for anc in opt_surface.x_ancilla},
                            [[data.idx for data in opt_surface.logical_z_qubits]],
                            [[data.idx for data in opt_surface.logical_x_qubits]])
    
    tasks = []

    for p in p_range:
        bad_patch.set_error_vals_normal({'T1': 10**7, 'T2': 10**7, 'gate1_err': p, 'gate2_err': p, 'readout_err': p})
        good_patch.set_error_vals_normal({'T1': 10**7, 'T2': 10**7, 'gate1_err': p, 'gate2_err': p, 'readout_err': p})
        tasks.append(sinter.Task(circuit=bad_patch.get_stim(), 
                                json_metadata={'p': p, 'opt': False, 'basis': 'XZ'}))
        tasks.append(sinter.Task(circuit=good_patch.get_stim(), 
                                json_metadata={'p': p, 'opt': True, 'basis': 'XZ'}))
        
        tasks.append(sinter.Task(circuit=bad_deff.stim_circ(p, 5, basis='X'), 
                                json_metadata={'p': p, 'opt': False, 'basis': 'X'}))
        tasks.append(sinter.Task(circuit=good_deff.stim_circ(p, 5, basis='X'), 
                                json_metadata={'p': p, 'opt': True, 'basis': 'X'}))

    results = sinter.collect(
        num_workers=1,
        tasks=tasks,
        decoders=['pymatching'],
        max_shots=20_000_000,
        max_errors=1_000,
        print_progress=False,)

    bad_depth_data = np.zeros(len(p_range))
    good_depth_data = np.zeros(len(p_range))

    bad_d_eff_data = np.zeros(len(p_range))
    good_d_eff_data = np.zeros(len(p_range))

    for result in results:
        p = result.json_metadata['p']
        basis = result.json_metadata['basis']
        p_idx = p_range.index(p)
        if result.json_metadata['opt']:
            if basis == 'X':
                good_d_eff_data[p_idx] = result.errors / (5 * result.shots)
            else:
                good_depth_data[p_idx] = result.errors / (5 * result.shots)
        else:
            if basis == 'X':
                bad_d_eff_data[p_idx] = result.errors / (5 * result.shots)
            else:
                bad_depth_data[p_idx] = result.errors / (5 * result.shots)

    intro_data = {
        'bad_depth': bad_depth_data,
        'good_depth': good_depth_data,
        'bad_deff': bad_d_eff_data,
        'good_deff': good_d_eff_data
    }
    pkl.dump(intro_data, open('data/surface/d_5/intro_figure_data.pkl', 'wb'))