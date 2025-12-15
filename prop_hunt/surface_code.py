from stim_surface_code.memory import MemoryPatch
import stim


class CustomScheduleMemoryPatch(MemoryPatch):
    # Simulates X and Z error rates simultaneously with Stim for surface codes

    def set_custom_schedule(self, custom_schedule: dict[int, tuple[str, list[tuple[int,int]]]]) -> None:
        self.custom_schedule = custom_schedule

    def syndrome_round(self, circ: stim.Circuit, deterministic_detectors: list[int] = [], inactive_detectors: list[int] = []) -> stim.Circuit:
        self.apply_reset(
            circ, 
            [measure.idx for measure in self.ancilla
             if self.qubits_active[measure.idx]],
        )

        # Gates
        self.apply_1gate(circ, 'H', [measure.idx 
                                     for measure in self.x_ancilla
                                     if self.qubits_active[measure.idx]])
        for t, (gate, err_qubits) in self.custom_schedule.items():
            self.apply_2gate(circ, gate ,err_qubits)
            
        self.apply_1gate(circ, 'H', [measure.idx 
                                     for measure in self.x_ancilla
                                     if self.qubits_active[measure.idx]])

        # Measure
        self.apply_meas(
            circ, 
            [measure.idx for measure in self.ancilla
             if self.qubits_active[measure.idx]],
        )

        for ancilla in self.ancilla:
            if (self.qubits_active[ancilla.idx] 
                and ancilla.idx not in inactive_detectors):
                if ancilla.idx in deterministic_detectors:
                    # no detector history to compare
                    circ.append(
                        'DETECTOR', 
                        self.get_meas_rec(-1, ancilla.idx), 
                        ancilla.coords + (0,)
                    )
                else:
                    # compare detector to a previous round
                    circ.append(
                        'DETECTOR', 
                        [self.get_meas_rec(-1, ancilla.idx),
                        self.get_meas_rec(-2, ancilla.idx)],
                        ancilla.coords + (0,)
                    )

        circ.append('TICK')
        circ.append('SHIFT_COORDS', [], [0.0, 0.0, 1.0])

        return circ

    def get_stim(self) -> stim.Circuit:

        assert self.error_vals_initialized

        obs_ancilla = len(self.all_qubits) + 1

        self.meas_record: list[dict[int, int]] = []

        circ = stim.Circuit()

        # Coords
        for qubit in self.all_qubits:
            circ.append('QUBIT_COORDS', qubit.idx, qubit.coords)

        # Measure ZZ and XX
        ZZ_obs = [stim.target_z(obs_ancilla), stim.target_combiner()] + [targ 
                                                            for targ_pair in [[stim.target_z(qubit.idx), stim.target_combiner()] 
                                                            for qubit in self.logical_z_qubits] for targ in targ_pair][:-1]
        XX_obs = [stim.target_x(obs_ancilla), stim.target_combiner()] + [targ 
                                                            for targ_pair in [[stim.target_x(qubit.idx), stim.target_combiner()] 
                                                            for qubit in self.logical_x_qubits] for targ in targ_pair][:-1]
        circ.append('R', [obs_ancilla] + [qubit.idx for qubit in self.data])

        meas_round = {}
        anc_left = len(self.ancilla)
        for anc in self.z_ancilla:
            mpp_stab = [targ for targ_pair in [[stim.target_z(qubit.idx), stim.target_combiner()] 
                             for qubit in anc.data_qubits if qubit] for targ in targ_pair][:-1]
            circ.append("MPP", mpp_stab)
            meas_round[anc.idx] = -2 - anc_left
            anc_left -= 1
        
        for anc in self.x_ancilla:
            mpp_stab = [targ for targ_pair in [[stim.target_x(qubit.idx), stim.target_combiner()] 
                             for qubit in anc.data_qubits if qubit] for targ in targ_pair][:-1]
            circ.append("MPP", mpp_stab)
            meas_round[anc.idx] = -2 - anc_left
            anc_left -= 1
            
        assert anc_left == 0

        self.meas_record.append(meas_round)

        circ.append("MPP", ZZ_obs)
        circ.append("MPP", XX_obs)
        for qubit in self.logical_z_qubits:
            circ.append("CZ", [stim.target_rec(-1), qubit.idx])
        
        self.syndrome_round(circ)

        circ.append(stim.CircuitRepeatBlock(self.dm - 1, self.syndrome_round(stim.Circuit())))


        meas_round = {}
        anc_left = len(self.ancilla)
        for anc in self.z_ancilla:
            mpp_stab = [targ for targ_pair in [[stim.target_z(qubit.idx), stim.target_combiner()] 
                             for qubit in anc.data_qubits if qubit] for targ in targ_pair][:-1]
            circ.append("MPP", mpp_stab)
            meas_round[anc.idx] = -anc_left
            anc_left -= 1
        
        for anc in self.x_ancilla:
            mpp_stab = [targ for targ_pair in [[stim.target_x(qubit.idx), stim.target_combiner()] 
                             for qubit in anc.data_qubits if qubit] for targ in targ_pair][:-1]
            circ.append("MPP", mpp_stab)
            meas_round[anc.idx] = -anc_left
            anc_left -= 1
            
        assert anc_left == 0
        for round in self.meas_record:
            for q, idx in round.items():
                round[q] = idx - len(self.ancilla)
        self.meas_record.append(meas_round)

        for anc in self.ancilla:
            circ.append("DETECTOR", [self.get_meas_rec(-1, anc.idx), self.get_meas_rec(-2, anc.idx)], anc.coords + (0,))

        circ.append("MPP", ZZ_obs)
        circ.append("MPP", XX_obs)

        # Observable is observables of qubits along logical operator
        circ.append('OBSERVABLE_INCLUDE', stim.target_rec(-2), 0) # ZZ
        circ.append('OBSERVABLE_INCLUDE', stim.target_rec(-1), 1) # XX

        return circ