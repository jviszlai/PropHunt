from dataclasses import dataclass
import networkx as nx
import numpy as np
import collections
import cirq
from qldpc.codes import CSSCode
from qldpc.objects import Pauli
import stim


@dataclass(frozen=True)
class TwoQubitError:
    # 2Q Depolarizing Error
    qubit1: int
    qubit2: int
    pauli1: str
    pauli2: str
    time: int


@dataclass(frozen=True)
class FlippedCheck:
    check_id: int
    time: int


@dataclass(frozen=True)
class LogicalOperator:
    op_id: int
    basis: str
    data_list: list[int]


class PropagationGraph:
    def __init__(
        self,
        z_checks: dict[int, list[int]],
        x_checks: dict[int, list[int]],
        z_operators: list[list[int]],
        x_operators: list[list[int]],
        try_depth: bool = False,
    ):
        """
        A circuit-level model for a CSS code defined by parity checks and logical operators.
        Performs static propagation of unweighted 2Q depolarizing errors over a code distance d round schedule

        Arguments:
            z_checks -- Z parity checks - mapping of check id to list of data qubit ids
            x_checks -- X parity checks - mapping of check id to list of data qubit ids
            z_operators -- Z operators - each defined as a list of data qubit ids
            x_operators -- X operators - each defined as a list of data qubit ids

        Keyword Arguments:
            try_depth -- If false, initial schedule has all x checks occur before all z checks.
                         If true, initial schedule is according to the data qubit id order in z_checks and x_checks
                         with no guarantee of commutation preservation
        """

        self.z_checks: dict[int, list[int]] = z_checks
        self.x_checks: dict[int, list[int]] = x_checks
        self.operators: list[LogicalOperator] = [
            LogicalOperator(i, "Z", op) for i, op in enumerate(z_operators)
        ] + [
            LogicalOperator(i + len(z_operators), "X", op)
            for i, op in enumerate(x_operators)
        ]
        self.data_list: dict[int, set[int]] = {}
        for check_id, qubits in list(z_checks.items()) + list(x_checks.items()):
            for data in qubits:
                self.data_list.setdefault(data, set()).add(check_id)
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self.try_depth = try_depth
        self._build_init_graph()

    def _build_init_graph(self) -> None:
        """
        Builds an initial graph structure to represent a CNOT schedule
        """
        z_checks: list[int] = list(self.z_checks.keys())
        x_checks: list[int] = list(self.x_checks.keys())
        for check_id, qubits in self.z_checks.items():
            self.graph.add_node(check_id, basis="Z", data_list=qubits)
            for node in self.graph.nodes:
                if node == check_id:
                    continue
                data_overlap: set[int] = set(qubits) & set(
                    self.graph.nodes[node]["data_list"]
                )
                for data in data_overlap:
                    data_idx1: int = qubits.index(data)
                    data_idx2: int = self.graph.nodes[node]["data_list"].index(data)
                    if data_idx1 <= data_idx2:
                        self.graph.add_edge(check_id, node, data=data, key=data)
                    else:
                        self.graph.add_edge(node, check_id, data=data, key=data)

        for check_id, qubits in self.x_checks.items():
            self.graph.add_node(check_id, basis="X", data_list=qubits)
            for node in self.graph.nodes:
                if node == check_id:
                    continue
                data_overlap: set[int] = set(qubits) & set(
                    self.graph.nodes[node]["data_list"]
                )
                for data in data_overlap:
                    data_idx1: int = qubits.index(data)
                    data_idx2: int = self.graph.nodes[node]["data_list"].index(data)
                    if self.try_depth:
                        if data_idx1 <= data_idx2:
                            self.graph.add_edge(check_id, node, data=data, key=data)
                        else:
                            self.graph.add_edge(node, check_id, data=data, key=data)
                    else:
                        if node in z_checks or data_idx1 <= data_idx2:
                            self.graph.add_edge(check_id, node, data=data, key=data)
                        else:
                            self.graph.add_edge(node, check_id, data=data, key=data)

    def _deploarize_2q(self, qubit1: int, qubit2: int) -> list[TwoQubitError]:
        """
        Generates a list of 2Q depolarizing error components

        Arguments:
            qubit1 -- qubit 1 id
            qubit2 -- qubit 2 id

        Returns:
            list of 15 TwoQubitErrors for 2Q depolarizing components
        """

        errors = []
        for pauli1 in ["I", "X", "Y", "Z"]:
            for pauli2 in ["I", "X", "Y", "Z"]:
                if pauli1 == "I" and pauli2 == "I":
                    continue  # Skip the identity error
                errors.append(TwoQubitError(qubit1, qubit2, pauli1, pauli2, 0))
        return errors

    def _propagate_check_error(
        self,
        reachable_checks: dict[tuple[int, int], set[int]],
        check_id: int,
        check_basis: str,
        data_idx: int,
        pauli: str,
    ) -> tuple[set[int], set[int]]:
        """
        Propagate pauli error on a check qubit occuring after a CNOT

        Arguments:
            reachable_checks -- set of check qubits reachable in the same round for a given CNOT (check, data)
            check_id -- check qubit involved
            check_basis -- pauli basis of check qubit
            data_idx -- index in check_id's list of data qubits of the data qubit involved
            pauli -- pauli error to propagate

        Returns:
            tuple of set of flipped checks this round and set of flipped checks next round
        """
        if pauli == "I":
            return set(), set()
        if (check_basis == "Z" and pauli == "X") or (
            check_basis == "X" and pauli == "Z"
        ):
            return {check_id}, {check_id}

        basis_checks: dict[int, list[int]] = (
            self.x_checks if check_basis == "Z" else self.z_checks
        )
        data_qubits: list[int] = (
            self.z_checks[check_id] if check_basis == "Z" else self.x_checks[check_id]
        )
        t0_checks: set[int] = {check_id} if pauli == "Y" else set()
        t1_checks: set[int] = {check_id} if pauli == "Y" else set()
        for j in range(data_idx + 1, len(data_qubits)):
            possible_checks: set[int] = set(
                [
                    check
                    for check in self.data_list[data_qubits[j]]
                    if check in basis_checks
                ]
            )
            propagations: set[int] = set(
                [
                    check
                    for check in reachable_checks[(check_id, data_qubits[j])]
                    if check in basis_checks
                ]
            )
            t0_checks ^= propagations
            t1_checks ^= possible_checks - propagations - {check_id}
        return t0_checks, t1_checks

    def _propagate_data_error(
        self,
        reachable_checks: dict[tuple[int, int], set[int]],
        check_id: int,
        data_id: int,
        pauli: str,
    ) -> tuple[set[int], set[int]]:
        """
        Propagate pauli error on a data qubit occuring after a CNOT

        Arguments:
            reachable_checks -- set of check qubits reachable in the same round for a given CNOT (check, data)
            check_id -- check qubit involved
            data_id -- data qubit involved
            pauli -- pauli error to propagate

        Returns:
            tuple of set of flipped checks this round and set of flipped checks next round
        """
        affected_checks: list[int] = []
        if pauli == "Z" or pauli == "Y":
            affected_checks += list(self.x_checks.keys())
        if pauli == "X" or pauli == "Y":
            affected_checks += list(self.z_checks.keys())

        possible_checks: set[int] = set(
            [check for check in self.data_list[data_id] if check in affected_checks]
        )
        t0_checks: set[int] = set(
            [
                check
                for check in reachable_checks[(check_id, data_id)]
                if check in affected_checks
            ]
        )
        t1_checks: set[int] = possible_checks - t0_checks
        return t0_checks, t1_checks

    def generate_errors(
        self,
    ) -> dict[TwoQubitError, tuple[list[FlippedCheck], list[LogicalOperator]]]:
        """
        Generates mapping from 2Q depolarizing errors to the list of flipped checks and logical operators

        Returns:
            dictionary mapping 2Q errors to flipped checks and flipped logical operators
        """

        error_map: dict[
            TwoQubitError, tuple[list[FlippedCheck], list[LogicalOperator]]
        ] = {}
        reachable_checks: dict[tuple[int, int], set[int]] = {}
        for data, check_ids in self.data_list.items():
            data_subgraph: nx.DiGraph = nx.DiGraph(
                [
                    (u, v)
                    for u, v, data_attr in self.graph.edges(data="data")
                    if data_attr == data
                ]
            )
            for check_id in check_ids:
                reachable_checks[(check_id, data)] = set(
                    nx.descendants(data_subgraph, check_id)
                )

        for check_id, data_qubits in self.graph.nodes(data="data_list"):
            check_basis: str = "Z" if check_id in self.z_checks else "X"
            for idx, data_id in enumerate(data_qubits):
                two_q_errs: list[TwoQubitError] = self._deploarize_2q(check_id, data_id)
                for two_q_err in two_q_errs:
                    flipped_checks: list[FlippedCheck] = []
                    check_err_t0, check_err_t1 = self._propagate_check_error(
                        reachable_checks, check_id, check_basis, idx, two_q_err.pauli1
                    )
                    data_err_t0, data_err_t1 = self._propagate_data_error(
                        reachable_checks, check_id, data_id, two_q_err.pauli2
                    )
                    t0_errs: set[int] = check_err_t0 ^ data_err_t0
                    t1_errs: set[int] = check_err_t1 ^ data_err_t1
                    flipped_checks.extend(
                        [FlippedCheck(check_id, 0) for check_id in t0_errs]
                    )
                    flipped_checks.extend(
                        [FlippedCheck(check_id, 1) for check_id in t1_errs]
                    )

                    hook_data: list[int] = data_qubits[idx + 1 :]
                    z_flip_data: list[int] = (
                        hook_data
                        if check_basis == "Z"
                        and (two_q_err.pauli1 == "Z" or two_q_err.pauli1 == "Y")
                        else []
                    ) + (
                        [data_id]
                        if two_q_err.pauli2 == "Z" or two_q_err.pauli2 == "Y"
                        else []
                    )
                    x_flip_data: list[int] = (
                        hook_data
                        if check_basis == "X"
                        and (two_q_err.pauli1 == "X" or two_q_err.pauli1 == "Y")
                        else []
                    ) + (
                        [data_id]
                        if two_q_err.pauli2 == "X" or two_q_err.pauli2 == "Y"
                        else []
                    )
                    flipped_operators: list[LogicalOperator] = []

                    for op in self.operators:
                        if op.basis == "X" and np.logical_xor.reduce(
                            [data in op.data_list for data in z_flip_data]
                        ):
                            flipped_operators.append(op)
                        elif op.basis == "Z" and np.logical_xor.reduce(
                            [data in op.data_list for data in x_flip_data]
                        ):
                            flipped_operators.append(op)

                    if len(flipped_checks) > 0:
                        for t in range(self.d):
                            two_q_err_t = TwoQubitError(
                                two_q_err.qubit1,
                                two_q_err.qubit2,
                                two_q_err.pauli1,
                                two_q_err.pauli2,
                                t,
                            )
                            flipped_checks_t = [
                                FlippedCheck(fcheck.check_id, t + fcheck.time)
                                for fcheck in flipped_checks
                            ]
                            error_map[two_q_err_t] = (
                                flipped_checks_t,
                                flipped_operators,
                            )

        return error_map

    def error_map_to_mats(
        self,
        error_map: dict[
            TwoQubitError, tuple[list[FlippedCheck], list[LogicalOperator]]
        ] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Converts an error map dictionary to circuit-level matrices

        Keyword Arguments:
            error_map -- dictionary mapping 2Q errors over d rounds to flipped checks and flipped logical operators.
                         if None, error_map is generated using generate_errors

        Returns:
            a tuple of the circuit-level check matrix and the circuit-level logical observable matrix
        """
        if not error_map:
            error_map: dict[
                TwoQubitError, tuple[list[FlippedCheck], list[LogicalOperator]]
            ] = self.generate_errors()
        num_checks: int = len(self.z_checks) + len(self.x_checks)
        check_id_offset: int = min(
            list(self.z_checks.keys()) + list(self.x_checks.keys())
        )
        num_observables: int = len(self.operators)

        check_flips: np.ndarray = np.zeros(((self.d + 1) * num_checks, len(error_map)))
        obs_flips: np.ndarray = np.zeros((num_observables, len(error_map)))
        for i, (_, (flipped_checks, flipped_obs)) in enumerate(error_map.items()):
            for flipped_check in flipped_checks:
                check_flips[
                    flipped_check.check_id
                    - check_id_offset
                    + num_checks * flipped_check.time,
                    i,
                ] = 1
            for obs in flipped_obs:
                obs_flips[obs.op_id, i] = 1
        return check_flips, obs_flips

    def to_sched(self) -> dict[int, tuple[str, list[tuple[int, int]]]]:
        """
        Converts the PropagationGraph to a CNOT schedule for a single round of syndrome measurement

        Returns:
           A time-indexed dictionary. Each time step contains a list of parallel gates to perform
        """
        scheduled_cx: tuple[int, int] = set()
        to_schedule_cx: tuple[int, int] = set()
        for check, data_list in (self.z_checks | self.x_checks).items():
            for data in data_list:
                to_schedule_cx.add((check, data))
        cx_sched: dict[int, tuple[str, list[tuple[int, int]]]] = {}
        active: set[int] = set()
        t: int = 0
        while len(to_schedule_cx) > 0:
            cx_list: list[tuple[int, int]] = []
            active = set()
            for check, data_list in (self.z_checks | self.x_checks).items():
                for data in data_list:
                    if (
                        (check, data) in scheduled_cx
                        or check in active
                        or data in active
                    ):
                        continue
                    schedulable: bool = True
                    for check2, _, data2 in self.graph.in_edges(check, data="data"):
                        if data2 == data and (check2, data) not in scheduled_cx:
                            schedulable = False
                    if schedulable:
                        if check in self.z_checks:
                            cx_list.append((data, check))
                        else:
                            cx_list.append((check, data))
                        scheduled_cx.add((check, data))
                        to_schedule_cx.remove((check, data))
                        active.add(check)
                        active.add(data)
                    break
            if len(active) == 0:
                # Invalid schedule
                return None
            cx_sched[t] = ("CX", cx_list)
            t += 1
        return cx_sched

    def cirq_circ(self):
        """
        Converts the PropagationGraph to a cirq circuit

        Returns:
            A cirq circuit containing CX gates for a single round of syndrome measurement
        """
        sched = self.to_sched()
        circ: cirq.Circuit = cirq.Circuit()
        qubits: dict[int, cirq.LineQubit] = {}
        for _, (gate, qubit_pairs) in sched.items():
            operations: list[cirq.Operation] = []
            for q1, q2 in qubit_pairs:
                if q1 not in qubits:
                    qubits[q1] = cirq.LineQubit(q1)
                if q2 not in qubits:
                    qubits[q2] = cirq.LineQubit(q2)
                operations.append(cirq.CX(qubits[q1], qubits[q2]))
            circ.append(cirq.Moment(operations))
        return circ

    def stim_circ(
        self, p: float, num_rounds=1, basis="Z", idle_strength=0.0
    ):  # Idle strength = t / T (t: idle time, T: coherence time)
        """
        Generates a stim experiment for the PropagationGraph

        Arguments:
            p -- uniform physical error rate

        Keyword Arguments:
            num_rounds -- number of measurement rounds to include
            basis -- logical operator basis measured at the end
            idle_strength -- Strength of idle errors in between gate moments

        Returns:
            constructed stim circuit
        """
        meas_record = []

        sched = self.to_sched()

        z_ancilla: list[int] = list(self.z_checks.keys())
        x_ancilla: list[int] = list(self.x_checks.keys())
        data: list[int] = list(self.data_list.keys())

        all_ancilla: list[int] = z_ancilla + x_ancilla
        all_qubits: list[int] = all_ancilla + data
        det_ancilla: list[int] = z_ancilla if basis == "Z" else x_ancilla

        logical_qubits: list[tuple[LogicalOperator, LogicalOperator]] = [
            (self.operators[i], self.operators[i + len(self.operators) // 2])
            for i in range(len(self.operators) // 2)
        ]

        def apply_idle(circ, qubits):
            if idle_strength == 0.0:
                return
            p_x = 0.25 * (1 - np.exp(-idle_strength))
            p_y = p_x
            p_z = 0.5 * (1 - np.exp(-idle_strength)) - p_x
            circ.append("PAULI_CHANNEL_1", qubits, (p_x, p_y, p_z))

        def apply_1gate(circ, gate, qubits):
            circ.append(gate, qubits)
            circ.append("DEPOLARIZE1", qubits, p)
            circ.append("TICK")

        def apply_2gate(circ, gate, qubit_pairs):
            err_qubits = []
            for q1, q2 in qubit_pairs:
                circ.append(gate, [q1, q2])
                err_qubits += [q1, q2]

            if len(err_qubits) > 0:
                circ.append("DEPOLARIZE2", err_qubits, p)
                circ.append("TICK")

        def meas_qubits(circ, op, qubits, perfect=False):
            if not perfect:
                circ.append("X_ERROR", qubits, p)
            circ.append(op, qubits)
            circ.append("TICK")

            # Update measurement record indices
            meas_round = {}
            for i in range(len(qubits)):
                q = qubits[-(i + 1)]
                meas_round[q] = -(i + 1)
            for round in meas_record:
                for q, idx in round.items():
                    round[q] = idx - len(qubits)
            meas_record.append(meas_round)

        def get_meas_rec(round_idx, qubit):
            return stim.target_rec(meas_record[round_idx][qubit])

        def stabilizer_circ(circ):
            apply_1gate(circ, "H", [anc for anc in x_ancilla])

            for _, (gate, qubit_pairs) in sched.items():
                apply_2gate(circ, gate, qubit_pairs)
                apply_idle(circ, all_qubits)

            apply_1gate(circ, "H", [anc for anc in x_ancilla])

            # Readout syndromes
            meas_qubits(circ, "MR", [anc for anc in all_ancilla])

        def repeated_stabilizers(circ, repetitions):
            repeat_circ = stim.Circuit()
            stabilizer_circ(repeat_circ)
            for anc in det_ancilla:
                repeat_circ.append(
                    "DETECTOR", [get_meas_rec(-1, anc), get_meas_rec(-2, anc)], (anc, 0)
                )
            repeat_circ.append("SHIFT_COORDS", [], (0, 1))

            circ.append(stim.CircuitRepeatBlock(repetitions, repeat_circ))

        obs_ancillas = [len(all_qubits) + i for i in range(len(self.operators) // 2)]

        circ = stim.Circuit()

        # Coords
        for qubit in all_qubits:
            circ.append("QUBIT_COORDS", qubit, qubit)

        # Measure ZZ and XX
        obs_list = []
        for i, (z_obs, x_obs) in enumerate(logical_qubits):
            ZZ_obs = [stim.target_z(obs_ancillas[i]), stim.target_combiner()] + [
                targ
                for targ_pair in [
                    [stim.target_z(qubit), stim.target_combiner()]
                    for qubit in z_obs.data_list
                ]
                for targ in targ_pair
            ][:-1]
            XX_obs = [stim.target_x(obs_ancillas[i]), stim.target_combiner()] + [
                targ
                for targ_pair in [
                    [stim.target_x(qubit), stim.target_combiner()]
                    for qubit in x_obs.data_list
                ]
                for targ in targ_pair
            ][:-1]
            obs_list.append((ZZ_obs, XX_obs, z_obs))

        # Reset
        circ.append(
            "R",
            [obs_ancilla for obs_ancilla in obs_ancillas]
            + [qubit for qubit in all_qubits],
        )

        # Init Stabilizers
        meas_round = {}
        anc_left = len(all_ancilla)
        for anc in z_ancilla:
            mpp_stab = [
                targ
                for targ_pair in [
                    [stim.target_z(qubit), stim.target_combiner()]
                    for qubit in self.z_checks[anc]
                ]
                for targ in targ_pair
            ][:-1]
            circ.append("MPP", mpp_stab)
            meas_round[anc] = -2 * len(logical_qubits) - anc_left
            anc_left -= 1

        for anc in x_ancilla:
            mpp_stab = [
                targ
                for targ_pair in [
                    [stim.target_x(qubit), stim.target_combiner()]
                    for qubit in self.x_checks[anc]
                ]
                for targ in targ_pair
            ][:-1]
            circ.append("MPP", mpp_stab)
            meas_round[anc] = -2 * len(logical_qubits) - anc_left
            anc_left -= 1

        assert anc_left == 0

        meas_record.append(meas_round)

        for ZZ_obs, XX_obs, z_obs in obs_list:
            circ.append("MPP", ZZ_obs)
            circ.append("MPP", XX_obs)
            for qubit in z_obs.data_list:
                circ.append("CZ", [stim.target_rec(-1), qubit])

        stabilizer_circ(circ)

        for anc in det_ancilla:
            circ.append(
                "DETECTOR", [get_meas_rec(-1, anc), get_meas_rec(-2, anc)], (anc, 0)
            )

        # Stabilizers
        repeated_stabilizers(circ, num_rounds)

        meas_round = {}
        anc_left = len(all_ancilla)
        for anc in z_ancilla:
            mpp_stab = [
                targ
                for targ_pair in [
                    [stim.target_z(qubit), stim.target_combiner()]
                    for qubit in self.z_checks[anc]
                ]
                for targ in targ_pair
            ][:-1]
            circ.append("MPP", mpp_stab)
            meas_round[anc] = -anc_left
            anc_left -= 1

        for anc in x_ancilla:
            mpp_stab = [
                targ
                for targ_pair in [
                    [stim.target_x(qubit), stim.target_combiner()]
                    for qubit in self.x_checks[anc]
                ]
                for targ in targ_pair
            ][:-1]
            circ.append("MPP", mpp_stab)
            meas_round[anc] = -anc_left
            anc_left -= 1

        assert anc_left == 0
        for round in meas_record:
            for q, idx in round.items():
                round[q] = idx - len(all_ancilla)
        meas_record.append(meas_round)

        for anc in det_ancilla:
            circ.append(
                "DETECTOR", [get_meas_rec(-1, anc), get_meas_rec(-2, anc)], (anc, 0)
            )

        for i, (ZZ_obs, XX_obs, _) in enumerate(obs_list):
            circ.append("MPP", ZZ_obs)
            circ.append("MPP", XX_obs)

            if basis == "Z":
                circ.append("OBSERVABLE_INCLUDE", stim.target_rec(-2), 2 * i)  # ZZ
            else:
                circ.append("OBSERVABLE_INCLUDE", stim.target_rec(-1), 2 * i + 1)  # XX

        return circ


def prop_graph_from_code(code: CSSCode) -> PropagationGraph:
    """
    Generates a PropagationGraph from a qldpc.CSSCode. Uses a coloration circuit as the initial schedule

    Arguments:
        code -- qldpc.CSSCode

    Returns:
        PropagationGraph for code using a coloration circuit as the initial schedule
    """
    Hx = code.get_stabilizer_ops(Pauli.X)
    Hz = code.get_stabilizer_ops(Pauli.Z)

    Lx = code.get_logical_ops(Pauli.X)
    Lz = code.get_logical_ops(Pauli.Z)

    x_coloring = nx.coloring.greedy_color(
        nx.line_graph(code.code_x.graph.to_undirected()), "largest_first"
    )
    x_colors: dict[int, list[tuple[str, int, int]]] = collections.defaultdict(list)
    for edge, color in x_coloring.items():
        x_colors[color].append(sorted(edge))
    z_coloring = nx.coloring.greedy_color(
        nx.line_graph(code.code_z.graph.to_undirected()), "largest_first"
    )
    z_colors: dict[int, list[tuple[str, int, int]]] = collections.defaultdict(list)
    for edge, color in z_coloring.items():
        z_colors[color].append(sorted(edge))

    z_checks = {j + Hx.shape[0]: [] for j in range(Hz.shape[0])}
    x_checks = {i: [] for i in range(Hx.shape[0])}

    for color, edge_list in x_colors.items():
        for data_node, check_node in edge_list:
            x_checks[check_node.index].append(
                data_node.index + Hx.shape[0] + Hz.shape[0]
            )

    for color, edge_list in z_colors.items():
        for data_node, check_node in edge_list:
            z_checks[check_node.index + Hx.shape[0]].append(
                data_node.index + Hx.shape[0] + Hz.shape[0]
            )

    return PropagationGraph(
        z_checks,
        x_checks,
        [
            [k + Hx.shape[0] + Hz.shape[0] for k in np.where(Lz[i])[0]]
            for i in range(Lz.shape[0])
        ],
        [
            [k + Hx.shape[0] + Hz.shape[0] for k in np.where(Lx[i])[0]]
            for i in range(Lx.shape[0])
        ],
    )
