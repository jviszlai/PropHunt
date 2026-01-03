
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
from cirq import(
    NoiseModel,
    BitFlipChannel,
    PhaseFlipChannel,
)

import pickle as pkl
from mitiq import zne

p_err=0.0045
num_trials = 100
scale_factors = [1, 3, 5, 7]

data_folder = 'data/zne'

def gen_noise_model(PERR, distance):
    """Create sweepable Pauli noise model."""
    PTH = 0.009
    LERR = 0.03 * (PERR / PTH) ** float((distance + 1) / 2)
    return LERR # model as single-qubit errors remaining after correction


class PauliNoiseModel(NoiseModel):

    def __init__(self, error_rate):
        self.error_rate = error_rate

    def noisy_operation(self, op):
        error_rate = self.error_rate
        channel = BitFlipChannel(error_rate).on_each(op.qubits)
        channel += PhaseFlipChannel(error_rate).on_each(op.qubits)
        return [op, channel]

def distance_extrapolation(distance_scale_factors, ds_expectation_values):
    fac = zne.PolyFactory(scale_factors=distance_scale_factors, order=3)
    for s, v in zip(distance_scale_factors, ds_expectation_values):
        fac.push({"scale_factor": s}, v)
    result = fac.reduce()
    return result

def calculate_ds_exp_vals_linear(d_array, distance_indices, exp_vals, num_trials):
    """Perform extrapolation to zero noise limit on expectation values obtained
    with noise scaling by unitary folding. Return mean and standard deviation
    of the ZNE expectation values.
    """
    ds_values = np.zeros((num_trials, len(distance_indices)))

    for count, d_ind in enumerate(distance_indices):
        distance_scale_factors = [gen_noise_model(p_err, d_array[di]) / gen_noise_model(p_err, d_array[d_ind[0]]) for di in d_ind]
        for trial in range(num_trials):
            ds_values[trial, count] = distance_extrapolation(distance_scale_factors, exp_vals[trial, d_ind])

    return [np.mean(ds_values, axis=0), np.std(ds_values, axis=0)]

if __name__ == '__main__':

    ### Baseline - DS-ZNE

    d_array = [13, 11, 9, 7,5,3]
    exp_vals_depth20 = np.zeros((num_trials, len(scale_factors) + 1, len(d_array)))

    distance_indices = [[0, 1, 2, 3], [1, 2, 3, 4],[2, 3, 4,5]]

    for d_ind in range(len(d_array) - 3):
        exp_vals_depth20[:, :, d_ind] = np.loadtxt(os.path.join(f"{data_folder}/data_scale_shots_perr_0045", f"depth50_distance{d_array[d_ind]}.txt"))

    for d_ind in range(len(d_array)-3, len(d_array)):
        exp_val = np.loadtxt(os.path.join(f"{data_folder}/data_scale_shots_perr_0045", f"depth50_distance{d_array[d_ind]}.txt"))
        exp_vals_depth20[:, 0, d_ind] = exp_val[:, 0]

    baseline_ds_exp_vals = calculate_ds_exp_vals_linear(d_array, distance_indices, exp_vals_depth20[:, 0, :], num_trials)


    ### Hook-ZNE: Optimal Config --  used in the evaluation

    d_array = [13,12.5,12.0,11.5,11,10.5,10,9.5,9,8.5,8,7.5,7]

    exp_vals_depth20 = np.zeros((num_trials, len(scale_factors) + 1, len(d_array)))

    ## Note -- higher dynamic range in noise amplification helps

    distance_indices = [[0, 2, 3, 4], [4,  6, 7, 8],[8, 10, 11, 12]]
    for d_ind in range(len(d_array) - 3):
        exp_vals_depth20[:, :, d_ind] = np.loadtxt(os.path.join(f"{data_folder}/data_scale_shots_perr_0045", f"depth50_distance{d_array[d_ind]}.txt"))

    for d_ind in range(len(d_array)-3, len(d_array)):
        exp_val = np.loadtxt(os.path.join(f"{data_folder}/data_scale_shots_perr_0045", f"depth50_distance{d_array[d_ind]}.txt"))
        exp_vals_depth20[:, 0, d_ind] = exp_val[:, 0]
    hook_zne_ds_exp_vals = calculate_ds_exp_vals_linear(d_array, distance_indices, exp_vals_depth20[:, 0, :], num_trials)

    zne_data = {
        'baseline': baseline_ds_exp_vals,
        'hook': hook_zne_ds_exp_vals
    }

    pkl.dump(zne_data, open(f'{data_folder}/zne_expectation_data.pkl', 'wb'))