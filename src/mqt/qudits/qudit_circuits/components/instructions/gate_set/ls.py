from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import expm

from mqt.qudits.qudit_circuits.components.instructions.gate import Gate
from mqt.qudits.qudit_circuits.components.instructions.gate_extensions.gate_types import GateTypes
from mqt.qudits.qudit_circuits.components.instructions.mini_tools.matrix_factory_tools import from_dirac_to_basis

if TYPE_CHECKING:
    from mqt.qudits.qudit_circuits.circuit import QuantumCircuit
    from mqt.qudits.qudit_circuits.components.instructions.gate_extensions.controls import ControlData


class LS(Gate):
    def __init__(
        self,
        circuit: QuantumCircuit,
        name: str,
        target_qudits: list[int] | int,
        parameters: list | None,
        dimensions: list[int] | int,
        controls: ControlData | None = None,
    ):
        super().__init__(
            circuit=circuit,
            name=name,
            gate_type=GateTypes.TWO,
            target_qudits=target_qudits,
            dimensions=dimensions,
            control_set=controls,
        )
        if self.validate_parameter(parameters):
            self.theta = parameters[0]
            self._params = parameters
        self.qasm_tag = "ls"

    def __array__(self, dtype: str = "complex") -> np.ndarray:
        dimension_0 = self._dimensions[0]
        dimension_1 = self._dimensions[1]

        exp_matrix = np.zeros((dimension_0 * dimension_1, dimension_0 * dimension_1), dtype="complex")
        d_min = min(dimension_0, dimension_1)
        for i in range(d_min):
            exp_matrix += np.outer(
                np.array(from_dirac_to_basis([i, i], self._dimensions)),
                np.array(from_dirac_to_basis([i, i], self._dimensions)),
            )

        return expm(-1j * self.theta * exp_matrix)

    def validate_parameter(self, parameter):
        assert 0 <= parameter[0] <= 2 * np.pi, f"Angle should be in the range [0, 2*pi]: {parameter[0]}"
        return True

    def __str__(self):
        # TODO
        pass
