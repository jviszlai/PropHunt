import cirq
import copy
import networkx as nx
import numpy as np
import z3

from prop_hunt.ambiguous_error import (
    check_ambiguity,
    TwoQubitError,
    _solve_maxsat,
    error_sampler,
)
from prop_hunt.prop_graph import PropagationGraph


class PropHuntCompiler:
    def __init__(self, prop_graph: PropagationGraph):
        """
        Optimizes the CNOT schedule of a CSS code represented by a PropagationGraph

        Arguments:
            prop_graph -- circuit-level model of a CSS code
        """
        self.prop_graph = prop_graph

    def _circuit_validity(self, prop_graph: PropagationGraph) -> bool:
        """
        Verifies the CNOT schedule represented by the PropagationGraph is valid - no cyclic dependencies and all checks commute

        Arguments:
            prop_graph -- circuit-level model of a CSS code

        Returns:
            True if the schedule is valid, false otherwise
        """
        # All data wires should be acyclic
        for data_id in prop_graph.data_list:
            subgraph: nx.DiGraph = nx.DiGraph(
                [
                    (u, v)
                    for u, v, data_attr in prop_graph.graph.edges(data="data")
                    if data_attr == data_id
                ]
            )
            if not nx.is_directed_acyclic_graph(subgraph):
                return False
        # All opposite type checks should commute
        z_check_idx: dict[int, int] = {
            z_check: i for i, z_check in enumerate(prop_graph.z_checks.keys())
        }
        x_check_idx: dict[int, int] = {
            x_check: i for i, x_check in enumerate(prop_graph.x_checks.keys())
        }
        out_commutivity: np.ndarray = np.zeros(
            (len(prop_graph.z_checks), len(prop_graph.x_checks)), dtype=bool
        )
        in_commutivity: np.ndarray = np.zeros(
            (len(prop_graph.z_checks), len(prop_graph.x_checks)), dtype=bool
        )
        for check_id in prop_graph.z_checks.keys():
            out_edges = prop_graph.graph.out_edges(check_id)
            in_edges = prop_graph.graph.in_edges(check_id)
            for _, check2 in out_edges:
                if check2 in prop_graph.x_checks:
                    out_commutivity[z_check_idx[check_id]][x_check_idx[check2]] ^= 1
            for check2, _ in in_edges:
                if check2 in prop_graph.x_checks:
                    in_commutivity[z_check_idx[check_id]][x_check_idx[check2]] ^= 1
        return np.count_nonzero(out_commutivity) + np.count_nonzero(in_commutivity) == 0

    def _generate_candidate_changes(
        self, errors: list[TwoQubitError]
    ) -> list[PropagationGraph]:
        """
        Generates a set of candidate CNOT schedules for a list of circuit-level errors

        Arguments:
            errors -- list of circuit-level errors to modify the error propagation of

        Returns:
            a list of candidate PropagationGraphs
        """
        mutate_ordering: list[(int, int, int)] = []
        mutate_schedule: list[(int, int)] = []
        check_basis: dict[int, str] = {
            z_check: "Z" for z_check in self.prop_graph.z_checks.keys()
        } | {x_check: "X" for x_check in self.prop_graph.x_checks.keys()}
        for error in errors:
            # Hook errors
            if (error.pauli1 == "Z" or error.pauli1 == "Y") and check_basis[
                error.qubit1
            ] == "Z":
                if error.pauli2 == "Z" or error.pauli2 == "Y":
                    mutate_ordering.append((error.qubit1, error.qubit2, 1))
                else:
                    mutate_ordering.append((error.qubit1, error.qubit2, 0))
            elif (error.pauli1 == "X" or error.pauli1 == "Y") and check_basis[
                error.qubit1
            ] == "X":
                if error.pauli2 == "X" or error.pauli2 == "Y":
                    mutate_ordering.append((error.qubit1, error.qubit2, 1))
                else:
                    mutate_ordering.append((error.qubit1, error.qubit2, 0))
            # Data scheduling errors
            if error.pauli2 != "I":
                mutate_schedule.append((error.qubit1, error.qubit2))

        candidates: list[PropagationGraph] = []

        for check_id, qbit_id, back in mutate_ordering:
            basis: str = check_basis[check_id]
            check_dict: dict[int, list[int]] = (
                self.prop_graph.z_checks if basis == "Z" else self.prop_graph.x_checks
            )
            qbit_index = check_dict[check_id].index(qbit_id) - back
            if qbit_index < 0:
                qbit_index = 0
            i_range: list[int] = [
                i for i in range(len(check_dict[check_id])) if i != qbit_index
            ]
            for i in i_range:
                new_graph: PropagationGraph = copy.deepcopy(self.prop_graph)
                new_graph_check_dict: dict[int, list[int]] = (
                    new_graph.z_checks if basis == "Z" else new_graph.x_checks
                )
                pivot: int = new_graph_check_dict[check_id][i]
                new_graph_check_dict[check_id].remove(pivot)
                new_graph_check_dict[check_id].insert(qbit_index, pivot)
                candidates.append(new_graph)

        for check_id, qbit_id in mutate_schedule:
            subgraph: nx.DiGraph = nx.DiGraph(
                [
                    (u, v)
                    for u, v, data_attr in self.prop_graph.graph.edges(data="data")
                    if data_attr == qbit_id
                ]
            )
            for _, check2 in subgraph.out_edges(check_id):
                new_graph: PropagationGraph = copy.deepcopy(self.prop_graph)
                if check_basis[check2] == check_basis[check_id]:
                    new_graph.graph.remove_edge(check_id, check2, qbit_id)
                    new_graph.graph.add_edge(
                        check2, check_id, data=qbit_id, key=qbit_id
                    )
                else:
                    all_edges: list[tuple[int, int, int]] = list(
                        self.prop_graph.graph.in_edges(check_id, data="data")
                    ) + list(self.prop_graph.graph.out_edges(check_id, data="data"))
                    all_edges.remove((check_id, check2, qbit_id))
                    other_edge: tuple[int, int, int] = all_edges[0]
                    for u, v, data_id in [other_edge, (check_id, check2, qbit_id)]:
                        new_graph.graph.remove_edge(u, v, data_id)
                        new_graph.graph.add_edge(v, u, data=data_id, key=data_id)
                candidates.append(new_graph)

            for check2, _ in subgraph.in_edges(check_id):
                new_graph: PropagationGraph = copy.deepcopy(self.prop_graph)
                if check_basis[check2] == check_basis[check_id]:
                    new_graph.graph.remove_edge(check2, check_id, qbit_id)
                    new_graph.graph.add_edge(
                        check_id, check2, data=qbit_id, key=qbit_id
                    )
                else:
                    all_edges: list[tuple[int, int, int]] = list(
                        self.prop_graph.graph.in_edges(check_id, data="data")
                    ) + list(self.prop_graph.graph.out_edges(check_id, data="data"))
                    all_edges.remove((check2, check_id, qbit_id))
                    other_edge: tuple[int, int, int] = all_edges[0]
                    for u, v, data_id in [other_edge, (check2, check_id, qbit_id)]:
                        new_graph.graph.remove_edge(u, v, data_id)
                        new_graph.graph.add_edge(v, u, data=data_id, key=data_id)
                candidates.append(new_graph)

        return candidates

    def compile(
        self,
        max_iter: int,
        distance: int,
        num_samples: int,
        num_workers: int,
        logger=None,
    ) -> list[PropagationGraph]:
        """
        Optimizes the CNOT schedule of the PropagationGraph

        Arguments:
            max_iter -- maximum number of iterations during optimization
            distance -- the distance of the CSS code
            num_samples -- number of ambiguous subgraph samples per iteration
            num_workers -- number of parallel workers for subgraph sampling

        Keyword Arguments:
            logger -- optional logger, otherwise logs are printed to the console

        Returns:
            history of the CNOT schedule, represented as a PropagationGraph, for each iteration
        """
        prop_graph_history: list[PropagationGraph] = []
        self.prop_graph.d = distance
        successive_no_changes: int = 0
        for i in range(max_iter):
            prop_graph_history.append(self.prop_graph)
            check_mat, obs_mat = self.prop_graph.error_map_to_mats()
            found_errors = error_sampler(
                self.prop_graph, 2 * distance, num_samples, num_workers
            )
            min_weight = distance
            min_weight_errors: list[np.ndarray] = []
            for error in found_errors:
                error_vector = np.array(
                    [1 if i in error else 0 for i in range(check_mat.shape[1])]
                )
                if (
                    np.count_nonzero((check_mat @ error_vector) % 2) == 0
                    and np.count_nonzero((obs_mat @ error_vector) % 2) > 0
                ):
                    if len(error) <= min_weight:
                        min_weight = len(error)
                        min_weight_errors.append(error)
            if logger:
                logger.info(f"Iteration: {i}, d_eff={min_weight}")
            else:
                print((f"Iteration: {i}, d_eff={min_weight}"))
            modified_syn_idx = set()
            changes_made = 0
            if logger:
                logger.info(f"Number Min Weight Errors: {len(min_weight_errors)}")
            else:
                print((f"Number Min Weight Errors: {len(min_weight_errors)}"))
            for logical_err_id in min_weight_errors:
                error_map = self.prop_graph.generate_errors()
                logical_err_list: list[TwoQubitError] = [
                    list(error_map.keys())[i] for i in logical_err_id
                ]
                err_list: list[TwoQubitError] = [
                    list(error_map.keys())[i] for i in logical_err_id
                ]
                check_mat, obs_mat = self.prop_graph.error_map_to_mats(
                    error_map=error_map
                )
                syn_idx: np.ndarray = np.where(
                    (check_mat[:, logical_err_id] == 1).any(axis=1)
                )[0]
                if set(syn_idx) & modified_syn_idx:
                    # Already modified
                    continue
                candidate_changes: list[PropagationGraph] = (
                    self._generate_candidate_changes(err_list)
                )

                valid_changes: list[PropagationGraph] = []
                for candidate in candidate_changes:
                    new_error_map = candidate.generate_errors()
                    new_check_mat, new_obs_mat = candidate.error_map_to_mats(
                        error_map=new_error_map
                    )
                    new_err_idx: np.ndarray = np.array(
                        [list(error_map.keys()).index(err) for err in logical_err_list]
                    )
                    if self._circuit_validity(candidate) and not check_ambiguity(
                        new_check_mat, new_obs_mat, syn_idx, new_err_idx
                    ):
                        valid_changes.append(candidate)
                if len(valid_changes) > 0:
                    change_scheds = [change.to_sched() for change in valid_changes]
                    valid_changes = [
                        valid_changes[i]
                        for i in range(len(valid_changes))
                        if change_scheds[i]
                    ]
                    circuit_depths: list[int] = [
                        len(sched) for sched in change_scheds if sched
                    ]
                    if len(circuit_depths) > 0:
                        self.prop_graph = valid_changes[np.argmin(circuit_depths)]
                        modified_syn_idx |= set(syn_idx)
                        changes_made += 1
            if logger:
                logger.info(f"Number Changes Made: {changes_made}")
            else:
                print((f"Number Changes Made: {changes_made}"))
            if changes_made == 0:
                successive_no_changes += 1
            if successive_no_changes == 3:
                if logger:
                    logger.info(
                        f"No changes applied for 3 iterations in a row. Ending search."
                    )
                else:
                    print(
                        (f"No changes applied for 3 iterations in a row. Ending search")
                    )
                return prop_graph_history
        return prop_graph_history
