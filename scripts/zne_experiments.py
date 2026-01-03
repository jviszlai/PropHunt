
from functools import partial
import os
import numpy as np
import cirq
from cirq import(
    I,
    Circuit,
    NoiseModel,
    DensityMatrixSimulator,
    BitFlipChannel,
    PhaseFlipChannel,
    measure,
)

from mitiq import zne
from mitiq.benchmarks import generate_rb_circuits
from mitiq.zne.scaling import fold_global

from pathlib import Path
import os, numpy as np

# Change this folder name if you want a different location in Drive
OUTPUT_DIR = Path("data/zne")


# Handy helper to wrap any iterator with tqdm while keeping code readable.
def with_progress(iterable, **tqdm_kwargs):
    return tqdm(iterable, **tqdm_kwargs)

def execute(circuit, shots, correct_bitstring):
    """Executes the input circuit(s) and returns ⟨A⟩, where
    A = |correct_bitstring⟩⟨correct_bitstring| for each circuit.
    """
    circuit_to_run = circuit.copy()

    circuit_to_run += measure(*sorted(circuit.all_qubits()), key="m")
    backend = DensityMatrixSimulator()

    result = backend.run(circuit_to_run, repetitions=shots)
    expval = result.measurements["m"].tolist().count(correct_bitstring) / shots
    return expval

def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def save_txt(rel_path: str, arr, fmt="%.8f"):
    """Save text file under OUTPUT_DIR/rel_path"""
    path = ensure_parent(OUTPUT_DIR / rel_path)
    np.savetxt(path, arr, fmt=fmt)
    return path

def load_txt(rel_path: str):
    """Load text file from OUTPUT_DIR/rel_path"""
    path = OUTPUT_DIR / rel_path
    return np.loadtxt(path)

def save_npz(rel_path: str, **arrays):
    """Save compressed numpy archive under OUTPUT_DIR/rel_path"""
    path = ensure_parent(OUTPUT_DIR / rel_path)
    np.savez_compressed(path, **arrays)
    return path

def load_npz(rel_path: str):
    """Load compressed numpy archive from OUTPUT_DIR/rel_path"""
    path = OUTPUT_DIR / rel_path
    return np.load(path)

def gen_noise_model(PERR, distance):
    """Create sweepable Pauli noise model."""
    PTH = 0.009
    LERR = 0.03 * (PERR / PTH) ** float((distance + 1) / 2)
    return LERR # model as single-qubit errors remaining after correction

def merge_func(op1, op2):
    return True

def noisy_execute(circ, noise_level, shots, correct_bitstring):
    qubits = circ.all_qubits()
    copy = Circuit()
    for moment in circ.moments:
        idle = False
        for q in qubits:
            # every moment every qubit gets a single-qubit noise op
            if not moment.operates_on_single_qubit(q):
                idle = True
                op_to_circ = Circuit(PauliNoiseModel(noise_level).noisy_operation(cirq.I(q)))
                merged_op = cirq.merge_operations_to_circuit_op(op_to_circ, merge_func)
                copy.append(moment.with_operations(merged_op.all_operations()))
                break
        if not idle:
            copy.append(moment)
    noisy_circ = copy.with_noise(PauliNoiseModel(noise_level))
    return execute(noisy_circ, shots=shots, correct_bitstring=correct_bitstring)

def distance_scaled_execute(circ, distance, base_noise_level, shots, correct_bitstring):
    LERR = gen_noise_model(base_noise_level, distance)
    return noisy_execute(circ, LERR, shots, correct_bitstring)

def scale_shots(num_device_qubits, scaled_distance, base_shots, n_qubits_circuit):
    used_qubits = n_qubits_circuit * scaled_distance ** 2
    return base_shots * int(num_device_qubits / used_qubits)


class PauliNoiseModel(NoiseModel):

    def __init__(self, error_rate):
        self.error_rate = error_rate

    def noisy_operation(self, op):
        error_rate = self.error_rate
        channel = BitFlipChannel(error_rate).on_each(op.qubits)
        channel += PhaseFlipChannel(error_rate).on_each(op.qubits)
        return [op, channel]

if __name__ == '__main__':
    # Create the base output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # === Performance & Progress Utilities ===
    # Parallel map and tqdm progress that "just works" (even if tqdm isn't installed).
    try:
        from tqdm.auto import tqdm
    except Exception:
        # graceful fallback if tqdm is unavailable
        def tqdm(x, **kwargs):
            return x

    # Joblib-based parallel map (process-based by default)
    try:
        from joblib import Parallel, delayed
        def pmap(func, iterable, n_jobs=-1, backend="loky", prefer=None, batch_size="auto", require=None, **kwargs):
            """
            Parallel map over `iterable` applying `func` to each element.
            Shows a tqdm progress bar automatically.
            - n_jobs=-1 uses all available cores.
            - backend="loky" is process-based; use backend="threading" for I/O-bound tasks.
            Any extra **kwargs are forwarded to joblib.Parallel.
            """
            iterable_list = list(iterable)  # so tqdm can compute total
            return Parallel(n_jobs=n_jobs, backend=backend, prefer=prefer, batch_size=batch_size, require=require, **kwargs)(
                delayed(func)(x) for x in tqdm(iterable_list, desc="pmap", leave=False)
            )
    except Exception as _e:
        # graceful fallback to sequential execution if joblib isn't present
        def pmap(func, iterable, **kwargs):
            iterable_list = list(iterable)
            out = []
            for x in tqdm(iterable_list, desc="pmap (seq)", leave=False):
                out.append(func(x))
            return out
    
    num_trials = 100

    base_shots = 20000
    device_size = 1200
    p_err=0.0045
    n_qubits = 2
    correct_bitstring = [0] * n_qubits
    depth = 50

    scale_factors = [1, 3, 5, 7]
    fac = zne.PolyFactory(scale_factors, order=3)

    d_array = [13,12.5,12.0,11.5, 11, 10.5, 10, 9.5, 9, 8.5, 8, 7.5, 7, 6.5, 6, 5.5, 5, 4.5, 4, 3.5, 3]

    circuits = generate_rb_circuits(n_qubits, depth, trials=num_trials)
    trial_results = np.zeros((num_trials, 5, len(d_array))) # row: trial, column: scaling technique, page: distance (high to low)

    for d_ind in tqdm(range(len(d_array))):
        print(f"On distance {d_array[d_ind]}")
        for trial in tqdm(range(num_trials)):
            if trial in np.linspace(0, num_trials, 11):
                print(f"    On trial {trial}")
            executor = partial(distance_scaled_execute, distance=d_array[d_ind], base_noise_level=p_err, shots=scale_shots(device_size, d_array[d_ind], base_shots, n_qubits), correct_bitstring=correct_bitstring)
            fac.run(circuits[trial], executor, scale_noise=fold_global)
            trial_results[trial, :-1, d_ind] = fac.get_expectation_values()
            trial_results[trial, -1, d_ind] = distance_scaled_execute(circuits[trial], d_array[d_ind], p_err, 4 * scale_shots(device_size, d_array[d_ind], base_shots, n_qubits), correct_bitstring)
        p_err_string = str(p_err).replace(".", "")[1:]
        save_txt(f"data_scale_shots_perr_{p_err_string}/depth{depth}_distance{d_array[d_ind]}.txt", trial_results[:, :, d_ind])

