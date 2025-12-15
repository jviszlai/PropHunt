import logging
import numpy as np
import os
import pickle as pkl
import sinter
from qldpc import decoders

import sys
sys.path.append('..')

from prop_hunt.prop_graph import PropagationGraph
from prop_hunt.prop_hunt import PropHuntCompiler


'''
Example parameters:
samples: 500
num_workers: 36
'''
if __name__ == '__main__':
    code_type = str(sys.argv[1])
    d = int(sys.argv[2])
    samples = int(sys.argv[3])
    num_workers = int(sys.argv[4])

    benchmark_codes = pkl.load(open('benchmark_codes.pkl', 'rb'))

    if code_type == 'surface':
        p_range = [1e-3, 3e-3, 5e-3, 7e-3, 9e-3, 1e-2]
    else:
        p_range = [5e-4, 1e-3, 3e-3, 5e-3, 7e-3]

    if (code_type, d) not in benchmark_codes:
        raise NotImplementedError('Code type not supported')
    
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
    max_iter = max_iters[(code_type, d)]

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

    code = benchmark_codes[(code_type, d)]
    prop_graph = PropagationGraph(code[0], code[1], code[2], code[3])
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


