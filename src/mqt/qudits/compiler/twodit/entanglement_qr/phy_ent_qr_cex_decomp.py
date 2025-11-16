from __future__ import annotations

import gc
from typing import TYPE_CHECKING, cast

from mqt.qudits.compiler import CompilerPass
from mqt.qudits.compiler.twodit.entanglement_qr import EntangledQRCEX
from mqt.qudits.core.custom_python_utils import append_to_front
from mqt.qudits.quantum_circuit.components.extensions.gate_types import GateTypes

if TYPE_CHECKING:
    from mqt.qudits.quantum_circuit import QuantumCircuit
    from mqt.qudits.quantum_circuit.gate import Gate
    from mqt.qudits.simulation.backends.backendv2 import Backend


class PhyEntQRCEXPass(CompilerPass):
    def __init__(self, backend: Backend) -> None:
        super().__init__(backend)
        from mqt.qudits.quantum_circuit import QuantumCircuit

        self.circuit = QuantumCircuit()

    def __transpile_local_ops(self, gate: Gate) -> list[Gate]:
        energy_graph_i = self.backend.energy_level_graphs[cast("int", gate.target_qudits)]
        """pi_pulses_routing = []
        pi_backs = []
        from mqt.qudits.quantum_circuit.gates import R, Rz, Rh, VirtRz
        if isinstance(gate, (R, Rz, Rh)):
            pi_pulses_routing, physical_rotation, pi_backs = ghost_routing(gate, energy_graph_i)
            if isinstance(gate, R):
                new_physical = R(gate.parent_circuit,
                                 "new_physical_R",
                                 gate.target_qudits,
                                 [physical_rotation.lev_a, physical_rotation.lev_b, physical_rotation.theta,
                                  physical_rotation.phi],
                                 gate.dimensions)
                if physical_rotation.theta * physical_rotation.phi * gate.theta * gate.phi < 0:
                    new_physical.dag()
            elif isinstance(gate, Rz):
                new_physical = Rz(gate.parent_circuit, "new_physical_Rz", gate.target_qudits,
                                  [physical_rotation.lev_a, physical_rotation.lev_b, gate.phi],
                                  gate.dimensions)
                if physical_rotation.theta * physical_rotation.phi * gate.phi < 0:
                    new_physical.dag()
            elif isinstance(gate, Rh):
                new_physical = Rh(gate.parent_circuit, "new_physical_Rh", gate.target_qudits,
                                  [physical_rotation.lev_a, physical_rotation.lev_b],
                                  gate.dimensions)
                if (physical_rotation.theta * physical_rotation.phi) < 0:
                    new_physical.dag()

        elif isinstance(gate, VirtRz):
            new_physical = VirtRz(gate.parent_circuit,
                                  "phyVrz",
                                  gate.target_qudits,
                                  [energy_graph_i.log_phy_map[gate.lev_a], gate.phi],
                                  gate.dimensions)
        decomp = [*pi_pulses_routing, new_physical, *pi_backs]"""

        from mqt.qudits.compiler.onedit.mapping_aware_transpilation import PhyQrDecomp
        qr = PhyQrDecomp(gate, energy_graph_i)
        decomp, _algorithmic_cost, _total_cost = qr.execute()


        ################################
        target = gate.to_matrix(1).copy()
        gdgdg0 = target.round(2)
        for rotation in decomp:
            target = rotation.to_matrix() @ target
            gdgdg0 = target.round(2)
        ################################

        return decomp

    @staticmethod
    def __transpile_two_ops(backend: Backend, gate: Gate) -> tuple[bool, list[Gate]]:
        assert gate.gate_type == GateTypes.TWO
        from mqt.qudits.compiler.twodit.transpile.phy_two_control_transp import PhyEntSimplePass

        phy_two_simple = PhyEntSimplePass(backend)
        transpiled = phy_two_simple.transpile_gate(gate)

        ################################
        target = gate.to_matrix(2).copy()
        for rotation in transpiled:
            target = rotation.to_matrix(identities=2) @ target
        dd = target.round(2)
        ################################

        return (len(transpiled) > 0), transpiled

    def transpile_gate(self, orig_gate: Gate) -> list[Gate]:
        simple_gate, simple_gate_decomp = self.__transpile_two_ops(self.backend, orig_gate)
        if simple_gate:
            return simple_gate_decomp

        eqr = EntangledQRCEX(orig_gate)
        decomp, _countcr, _countpsw = eqr.execute()

        # Full sequence of logical operations to be implemented to reconstruct
        # the logical operation on the device
        full_logical_sequence = [op.dag() for op in reversed(decomp)]

        # Actual implementation of the gate in the device based on the mapping
        physical_sequence: list[Gate] = []
        for gate in reversed(full_logical_sequence):
            if gate.gate_type == GateTypes.SINGLE:
                loc_gate = self.__transpile_local_ops(gate)
                append_to_front(physical_sequence, [op.dag() for op in reversed(loc_gate)])
            elif gate.gate_type == GateTypes.TWO:
                _, ent_gate = self.__transpile_two_ops(self.backend, gate)
                append_to_front(physical_sequence, ent_gate)
            elif gate.gate_type == GateTypes.MULTI:
                msg = "Multi not supposed to be in decomposition!"
                raise RuntimeError(msg)

        return physical_sequence

    def transpile(self, circuit: QuantumCircuit) -> QuantumCircuit:
        self.circuit = circuit
        instructions = circuit.instructions
        new_instructions: list[Gate] = []

        for gate in reversed(instructions):
            if gate.gate_type == GateTypes.TWO:
                gate_trans = self.transpile_gate(gate)
                append_to_front(new_instructions, gate_trans)
                # new_instructions.extend(gate_trans)
                gc.collect()
            else:
                append_to_front(new_instructions, gate)
                # new_instructions.append(gate)
        transpiled_circuit = self.circuit.copy()
        return transpiled_circuit.set_instructions(new_instructions)
