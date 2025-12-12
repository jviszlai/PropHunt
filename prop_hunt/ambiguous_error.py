from prop_hunt.prop_graph import (
    FlippedCheck,
    LogicalOperator,
    PropagationGraph,
    TwoQubitError,
)

from galois import GF2
import numpy as np
import os
import parse
from multiprocessing import Pool
import random
import string
import subprocess
import z3

SOLVER = "/home/viszlai/slurm/loandra/loandra"
TMP_FILE = "/home/viszlai/tmp/ambiguous_global2.wdimacs"
TIMEOUT = 3 * 60


def _generate_xor(var_list: list[z3.BoolRef]) -> z3.BoolRef:
    """
    Generates a tree of single or two variable XOR clauses for multivariate XOR functions

    Arguments:
        var_list -- List of z3 boolean variables

    Returns:
        Nested z3 boolean expression
    """
    if len(var_list) == 0:
        return z3.BoolVal(False)
    elif len(var_list) == 1:
        return var_list[0]
    elif len(var_list) == 2:
        return z3.Xor(var_list[0], var_list[1])
    else:
        return z3.Xor(
            _generate_xor(var_list[: len(var_list) // 2]),
            _generate_xor(var_list[len(var_list) // 2 :]),
        )


def _solve_maxsat(wdimacs: str, timeout: int, new_tmp_file: str = None) -> str:
    """
    Makes an external call to loandra to solve a MaxSAT problem in wdimacs format

    Arguments:
        wdimacs -- wdimacs string of MaxSAT problem to be solved
        timeout -- solver timeout

    Keyword Arguments:
        new_tmp_file -- temp file for wdimacs problem if different than global temp file

    Returns:
        Variable assignments output by solver
    """
    use_tmp_file = TMP_FILE if not new_tmp_file else new_tmp_file
    with open(use_tmp_file, "w") as f:
        f.write(wdimacs)
    if timeout == None:
        out_str: str = subprocess.run(
            f"{SOLVER} -print-model {use_tmp_file}",
            capture_output=True,
            text=True,
            shell=True,
        ).stdout
    else:
        out_str: str = subprocess.run(
            f"timeout {timeout} {SOLVER} -print-model {use_tmp_file}",
            capture_output=True,
            text=True,
            shell=True,
        ).stdout
    os.remove(use_tmp_file)
    return parse.parse("{}\nv {}", out_str)[1][:-1]


def _find_logical_error(
    error_map: dict[TwoQubitError, tuple[list[FlippedCheck], list[LogicalOperator]]],
    check_list: list[int],
    obs_list: list[LogicalOperator],
    timeout: int,
    new_tmp_file: str = None,
) -> list[int]:
    """
    Finds a minimum weight logical error for a given circuit-level error subproblem

    Arguments:
        error_map -- circuit-level model mapping gate errors to flipped syndromes and flipped logical observables
        check_list -- list of checks involved in subproblem
        obs_list -- list of observables involved in subproblem
        timeout -- timeout for MaxSAT solver

    Keyword Arguments:
        new_tmp_file -- temp file for MaxSAT solving if different than global temp file

    Returns:
        indices of circuit-level errors involved in the found min weight logical error
    """
    error_list: list[TwoQubitError] = list(error_map.keys())

    s = z3.Goal()
    error_vars: list[z3.BoolRef] = [
        z3.Bool(f"err_idx_{i}") for i in range(len(error_list))
    ]
    obs_flipped_vars: list[z3.BoolRef] = [
        z3.Bool(f"obs_flipped_idx_{i}") for i in range(len(obs_list))
    ]
    check_flips: set[FlippedCheck] = set()
    for _, (fcheck, fop) in error_map.items():
        check_flips |= set(fcheck)
    check_flips: dict[tuple[FlippedCheck], list[int]] = {
        fcheck: [] for fcheck in check_flips
    }
    obs_flips: dict[int, list[int]] = {i: [] for i in range(len(obs_list))}
    for i, error in enumerate(error_list):
        s.add(error_vars[i] == False)
        flipped_checks, flipped_obs = error_map[error]
        for flipped_check in flipped_checks:
            check_flips[flipped_check].append(i)
        for obs in flipped_obs:
            obs_idx = obs_list.index(obs)
            obs_flips[obs_idx].append(i)

    for i, obs in enumerate(obs_list):
        s.add(
            obs_flipped_vars[i] == _generate_xor([error_vars[j] for j in obs_flips[i]])
        )

    for fcheck in check_flips:
        s.add(_generate_xor([error_vars[j] for j in check_flips[fcheck]]) == False)

    s.add(z3.Or(obs_flipped_vars))

    cnf_tactic: z3.Tactic = z3.Then("simplify", "tseitin-cnf")
    dimacs: str = cnf_tactic(s)[0].dimacs()
    wdimacs: str = ""
    var_mapping: dict[int, str] = {}
    for i, line in enumerate(dimacs.split("\n")):
        if line.startswith("p cnf"):
            continue
        elif line.startswith("c"):
            _, cnf_id, var_id = line.split()
            var_mapping[int(cnf_id)] = var_id
        elif i <= len(error_vars):
            wdimacs += f"1 {line}\n"
        else:
            wdimacs += f"h {line}\n"

    assignments: str = _solve_maxsat(wdimacs, timeout, new_tmp_file)
    var_assignments: dict[str, bool] = {
        var: assignments[i - 1] == "1" for i, var in var_mapping.items()
    }
    error_assignments: list[bool] = [False] * len(error_vars)
    for var, val in var_assignments.items():
        if var.startswith("err_idx_"):
            error_assignments[int(var.split("_")[-1])] = val

    return np.where(error_assignments)[0]


def get_ambiguous_error(prop_graph: PropagationGraph, timeout: int = TIMEOUT):
    """
    Note: Not in use in PropHunt, but used for scaling analysis compared to global solver

    Performs a global solving for a minimum weight logical error for a circuit-level model described by the PropagationGraph

    Arguments:
        prop_graph -- circuit-level model of a CSS code

    Keyword Arguments:
        timeout -- timeout for MaxSAT solver

    Returns:
        indicies of circuit-level errors involved in min weight logical error
    """
    error_map: dict[TwoQubitError, tuple[list[FlippedCheck], list[LogicalOperator]]] = (
        prop_graph.generate_errors()
    )
    check_list: list[int] = list(prop_graph.z_checks.keys()) + list(
        prop_graph.x_checks.keys()
    )
    obs_list: list[LogicalOperator] = prop_graph.operators
    logical_error_indices: list[int] = _find_logical_error(
        error_map, check_list, obs_list, timeout
    )

    return logical_error_indices


def check_ambiguity(
    check_mat: np.ndarray,
    obs_mat: np.ndarray,
    syn_idx: np.ndarray,
    err_idx: np.ndarray,
    return_ambiguous_errors: bool = False,
) -> bool:
    """
    Checks if a set of syndromes corresponds to a subproblem that can decoded ambiguously or
    if a set of errors corresponds to a subproblem that can be decoded ambiguously

    Arguments:
        check_mat -- Global circuit-level check matrix
        obs_mat -- Global circuit-level observable matrix
        syn_idx -- Some set of syndrome indices
        err_idx -- Some set of error indicies

    Keyword Arguments:
        return_ambiguous_errors -- If syndromes contain ambiguity, return subproblem immediately

    Returns:
        If return_ambiguous_errors then returns ambiguous subproblem (used in ambiguous subgraph finding)
        else returns True if syn_idx or err_idx correspond to a subproblem containing ambiguity (used in ambiguity pruning of candidate changes)
    """
    # First check if old syndromes, new errors can be deocded ambiguously

    # Identify subset of errors (columns) in check_mat only supported on rows in syn_idx
    non_syn_idx: np.ndarray = np.array(
        [i for i in range(check_mat.shape[0]) if i not in syn_idx]
    )
    invalid_cols: np.ndarray = np.where((check_mat[non_syn_idx] == 1).any(axis=0))[0]
    ambiguous_errors: np.ndarray = np.array(
        [
            col
            for col in np.where((check_mat[syn_idx] == 1).any(axis=0))[0]
            if col not in invalid_cols
        ]
    )
    s_check_submat: np.ndarray = check_mat[syn_idx][:, ambiguous_errors].astype(
        np.uint8
    )
    s_obs_submat: np.ndarray = obs_mat[:, ambiguous_errors].astype(np.uint8)
    # If obs_submat in rowspace of check_submat, no ambiguity
    syn_ambiguity: bool = GF2._matrix_rank(GF2(s_check_submat)) != GF2._matrix_rank(
        GF2(np.vstack((s_check_submat, s_obs_submat)))
    )

    if return_ambiguous_errors:
        return syn_ambiguity, ambiguous_errors

    # Second check if new errors are still a logical error
    ambiguous_syndromes: np.ndarray = np.where(
        (check_mat[:, err_idx] == 1).any(axis=1)
    )[0]
    e_check_submat: np.ndarray = check_mat[ambiguous_syndromes][:, err_idx].astype(
        np.uint8
    )
    e_obs_submat: np.ndarray = obs_mat[:, err_idx].astype(np.uint8)
    err_ambiguity: bool = GF2._matrix_rank(GF2(e_check_submat)) != GF2._matrix_rank(
        GF2(np.vstack((e_check_submat, e_obs_submat)))
    )
    # error_vector: np.ndarray = np.array([1 if i in err_idx else 0 for i in range(check_mat.shape[1])])
    # err_ambiguity: bool = np.count_nonzero((check_mat @ error_vector) % 2) == 0 and np.count_nonzero((obs_mat @ error_vector) % 2) != 0

    return syn_ambiguity or err_ambiguity


def _sample_ambiguous_errors(args) -> list[np.ndarray]:
    prop_graph: PropagationGraph
    check_mat: np.ndarray
    obs_mat: np.ndarray
    weight_cap: int
    num_samples: int
    worker_idx: int
    prop_graph, check_mat, obs_mat, weight_cap, num_samples, worker_idx = args
    """
    Randomly samples ambiguous subgraphs and corresponding minimum weight logical errors.
    May be run on parallel threads, so numpy random seed must be regenerated

    Arguments:
        prop_graph -- circuit-level model for a CSS code
        check_mat -- circuit-level check matrix
        obs_mat -- circuit-level observable matrix
        weight_cap -- cap on number of errors in subproblem. If cap is reached, graph expansion halts.
        num_samples -- number of random ambiguous subgraphs to try and generate
        worker_idx -- parallel worker ID

    Returns:
        At most num_samples minimum weight logical errors corresponding to ambiguous subgraphs
    """
    ambiguous_errors: list[np.ndarray] = []
    error_map = prop_graph.generate_errors()
    error_list: list[TwoQubitError] = list(error_map.keys())
    tmp_file_hash: str = "".join(random.choice(string.ascii_letters) for _ in range(5))
    new_tmp_file = f"/home/viszlai/tmp/ambiguous_{worker_idx}_{tmp_file_hash}.wdimacs"
    for i in range(num_samples):
        np.random.seed()
        error_idx: list[int] = [np.random.randint(0, check_mat.shape[1])]
        a_error_idx: np.ndarray = None
        error_found: bool = False
        while len(error_idx) < weight_cap:
            np.random.seed()
            syn_idx: int = np.random.choice(np.where(check_mat[:, error_idx[-1]])[0])
            new_errors: list[int] = [
                idx for idx in np.where(check_mat[syn_idx])[0] if idx not in error_idx
            ]
            if len(new_errors) == 0:
                break
            np.random.seed()
            new_err: int = np.random.choice(new_errors)
            error_idx.append(new_err)

            syn_idx: np.ndarray = np.where(
                (check_mat[:, np.array(error_idx)] == 1).any(axis=1)
            )[0]
            ambiguous, a_error_idx = check_ambiguity(
                check_mat,
                obs_mat,
                syn_idx,
                np.array(error_idx),
                return_ambiguous_errors=True,
            )
            if ambiguous:
                error_found = True
                break
        if not error_found:
            continue
        reduced_error_map = {}
        check_list = set()
        obs_list = []
        for idx in a_error_idx:
            reduced_error_map[error_list[idx]] = error_map[error_list[idx]]
            for check in reduced_error_map[error_list[idx]][0]:
                check_list.add(check.check_id)
            for obs in reduced_error_map[error_list[idx]][1]:
                if obs not in obs_list:
                    obs_list.append(obs)
        reduced_error_list = list(reduced_error_map.keys())
        logical_error: list[int] = _find_logical_error(
            reduced_error_map, list(check_list), obs_list, TIMEOUT, new_tmp_file
        )
        ambiguous_errors.append(
            np.array([error_list.index(reduced_error_list[i]) for i in logical_error])
        )

    return ambiguous_errors


def error_sampler(
    prop_graph: PropagationGraph,
    weight_cap: int,
    num_samples: int = 500,
    num_workers: int = 4,
):
    """
    Parallel sampling of ambiguous subgraphs and minimum weight logical errors

    Arguments:
        prop_graph -- circuit-level model of a CSS code
        weight_cap -- cap on number of errors in subproblem during sampling. If cap is reached, graph expansion halts.

    Keyword Arguments:
        num_samples -- total number of subgraphs to try and sample
        num_workers -- number of parallel workers

    Returns:
        At most num_samples minimum weight logical errors corresponding to ambiguous subgraphs
    """
    check_mat, obs_mat = prop_graph.error_map_to_mats()
    samples_per_worker: int = num_samples // num_workers
    found_errors: list[np.ndarray] = []
    arg_list = [
        (prop_graph, check_mat, obs_mat, weight_cap, samples_per_worker, i)
        for i in range(num_workers)
    ]
    with Pool(processes=num_workers) as pool:
        results = pool.map(_sample_ambiguous_errors, arg_list)
        for ambiguous_errors in results:
            found_errors.extend(ambiguous_errors)
    return found_errors
