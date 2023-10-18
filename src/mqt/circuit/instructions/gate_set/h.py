from typing import Any

from mqt.circuit.instructions.gate import Gate


class H(Gate):
    def __qasm__(self) -> str:
        pass

    def __array__(self, dtype: str = "complex") -> Any:
        pass

    def control(self, num_ctrl_qudits: int = 1, label: str | None = None, ctrl_state: int | str | None = None):
        pass

    def validate_parameter(self, parameter):
        pass

    def __init__(self, name: str, num_qudits: int, params: list):
        super().__init__(name, num_qudits, params)