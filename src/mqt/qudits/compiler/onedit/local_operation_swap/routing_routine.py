from __future__ import annotations

from typing import cast

from mqt.qudits.quantum_circuit.gates import R, Rh, Rz
from mqt.qudits.core import LevelGraph


def ghost_routing(gate: R | Rz | Rh, graph: LevelGraph) -> tuple[list[R], R, list[R]]:
    from mqt.qudits.compiler.onedit.local_operation_swap import cost_calculator, gate_chain_condition
    circuit = gate.parent_circuit
    phi = gate.phi
    _, pi_pulses_routing, temp_placement, _, _ = cost_calculator(gate, graph, 0)

    if temp_placement.nodes[gate.lev_a]["lpmap"] > temp_placement.nodes[gate.lev_b]["lpmap"]:
        phi *= -1

    physical_rotation = R(
            circuit,
            "R",
            cast("int", gate.target_qudits),
            [temp_placement.nodes[gate.lev_a]["lpmap"], temp_placement.nodes[gate.lev_b]["lpmap"], gate.theta, phi],
            gate.dimensions,
    )

    physical_rotation = gate_chain_condition(pi_pulses_routing, physical_rotation)
    pi_backs = [
        R(
                circuit,
                "R",
                cast("int", gate.target_qudits),
                [pi_g.lev_a, pi_g.lev_b, pi_g.theta, -pi_g.phi],
                gate.dimensions,
        )
        for pi_g in reversed(pi_pulses_routing)
    ]

    return pi_pulses_routing, physical_rotation, pi_backs