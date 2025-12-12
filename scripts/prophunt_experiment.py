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

from prop_hunt.prop_graph import PropagationGraph, prop_graph_from_code
from prop_hunt.prop_hunt import PropHuntCompiler

p_range = [5e-4, 1e-3, 3e-3, 5e-3, 7e-3]

def lp_code(d: int):
    # Lifted Product Code
    if d == 3:
        group = abstract.CyclicGroup(3)
        zero = abstract.Element(group)
        x0, x1, x2 = [abstract.Element(group, member) for member in group.generate()]
        base_matrix = [[x1 + x2, x0, zero], [zero, x0 + x1, x1]]
        protograph = abstract.Protograph(base_matrix)
        return codes.LPCode(protograph)
    else:
        raise NotImplementedError

def rqt_code(d: int):
    # Random Quantum Tanner Graph
    # Fix RQT seed for reproducability of results (different seeds can create codes with varied code parameters)
    if d == 6:
        group = abstract.CyclicGroup(15)
        subcode = codes.RepetitionCode(2)
        return codes.QTCode.random(group, subcode, seed=8020)

def rqt_di_code(d: int, seed: int):
    # Random Quantum Tanner Graph based on Dihedral group 
    if d == 4:
        group = abstract.DihedralGroup(6)
        subcode = codes.RepetitionCode(3)
        return codes.QTCode.random(group, subcode, seed=seed)

def surface_code(d: int):
    return codes.SurfaceCode(d, d)

'''
Example parameters:
max_iter: 25
samples: 1_000
num_workers: 36
'''
if __name__ == '__main__':
    code_type = str(sys.argv[1])
    d = int(sys.argv[2])
    max_iter = int(sys.argv[3])
    samples = int(sys.argv[4])
    num_workers = int(sys.argv[5])

    get_codes = {
        'surface': lambda distance: surface_code(distance),
        'lp': lambda distance: lp_code(distance),
        'rqt': lambda distance: rqt_code(distance),
        'rqt_di_156': lambda distance: rqt_di_code(distance, 156),
        'rqt_di_8020': lambda distance: rqt_di_code(distance, 8020),
    }

    if code_type not in get_codes:
        raise NotImplementedError('Code type not supported')

    os.makedirs(f'data/{code_type}/d_{d}', exist_ok=True)
    log_path = f'data/{code_type}/d_{d}/prophunt.log'
    if os.path.exists(log_path):
        os.remove(log_path)

    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=log_path, encoding='utf-8', level=logging.INFO)

    num_rounds = d

    decoder_kwargs = {}
    if 'lp' in code_type or 'rqt' in code_type:
        decoder_kwargs = {'with_BP_LSD': True, 'max_iter': 30, 'bp_method': "ms", 'lsd_method': 'lsd_cs', 'lsd_order': 0}
        sinter_decoders = ['custom']
    elif code_type == 'surface':
        sinter_decoders = ['pymatching']

    code = get_codes[code_type](d)

    prop_graph = prop_graph_from_code(code)
    compiler: PropHuntCompiler = PropHuntCompiler(prop_graph)
    prop_graph_history: list[PropagationGraph] = compiler.compile(max_iter, d, samples, num_workers, logger)

    max_iter = len(prop_graph_history)

    pkl.dump(prop_graph_history, open(f'data/{code_type}/d_{d}/checkpoint_graphs.pkl', 'wb'))

    tasks = []
    for p in p_range:
        for i, graph in enumerate(prop_graph_history):   
            tasks.append(sinter.Task(circuit=graph.stim_circ(p, d, basis='Z'), json_metadata={'p': p, 'iteration': i, 'basis': 'Z'}))
            tasks.append(sinter.Task(circuit=graph.stim_circ(p, d, basis='X'), json_metadata={'p': p, 'iteration': i, 'basis': 'X'}))
    
    results = sinter.collect(
        num_workers=os.cpu_count(),
        tasks=tasks,
        decoders=sinter_decoders,
        max_shots=10_000_000,
        max_errors=500,
        custom_decoders={'custom': decoders.SinterDecoder(**decoder_kwargs)},)

    iter_data = {i: np.zeros(len(p_range)) for i in range(max_iter)}
    X_data = {i: [(0, 0) for _ in range(len(p_range))] for i in range(max_iter)}
    Z_data = {i: [(0, 0) for _ in range(len(p_range))] for i in range(max_iter)}
    for result in results:
        p = result.json_metadata['p']
        i = result.json_metadata['iteration']
        basis = result.json_metadata['basis']
        p_idx = p_range.index(p)
        if basis == 'X':
            X_data[i][p_idx] = (result.errors, result.shots)
        else:
            Z_data[i][p_idx] = (result.errors, result.shots)
    for i in range(max_iter):
        for p_idx in range(len(p_range)):
            iter_data[i][p_idx] = (X_data[i][p_idx][0] + Z_data[i][p_idx][0]) / (d * (X_data[i][p_idx][1] + Z_data[i][p_idx][1]))
    
    pkl.dump(iter_data, open(f'data/{code_type}/d_{d}/iter_data.pkl', 'wb'))


