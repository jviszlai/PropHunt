import logging
import numpy as np
import collections
import os
import pickle as pkl
import sinter
import time
from sympy.abc import x, y
from qldpc import abstract
from qldpc import codes
from qldpc.objects import Pauli

import sys
sys.path.append('..')

from prop_hunt.ambiguous_error import error_sampler, get_ambiguous_error

if __name__ == '__main__':
    benchmarks = [
        ('lp', 3),
        ('rqt', 6),
        ('surface', 7),
    ]

    for name, d in benchmarks:
        graphs = pkl.load(open(f'data/{name}/d_{d}/checkpoint_graphs.pkl', 'rb'))
        ambiguous_errors, solve_times =  error_sampler(graphs[0], d, 25, 1, return_solve_times=True)
        start_time = time.time()
        try:
            logical_error_indices = get_ambiguous_error(graphs[0], timeout=1800)
        except TypeError:
            logical_error_indices = [] # Indicates timeout
        end_time = time.time()
        global_d_eff = len(logical_error_indices)
        global_time = end_time - start_time
        timing_results = {
            'local': ([len(errors) for errors in ambiguous_errors], solve_times),
            'global': (global_d_eff, global_time)
        }
        pkl.dump(timing_results, open(f'data/{name}/d_{d}/timing_results.pkl', 'wb'))


