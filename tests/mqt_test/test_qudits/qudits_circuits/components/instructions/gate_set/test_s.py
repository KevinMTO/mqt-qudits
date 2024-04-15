from unittest import TestCase

import numpy as np
from mqt.qudits.qudit_circuits.circuit import QuantumCircuit

from tests.mqt_test.test_qudits.qudits_circuits.components.instructions.gate_set.utils import omega_d


class TestS(TestCase):
    def setUp(self):
        self.circuit_23 = QuantumCircuit(1, [2, 3], 0)

    def test___array__(self):
        s_0 = self.circuit_23.s(0)
        matrix_0 = s_0.to_matrix(identities=0)
        assert np.allclose(np.array([[1, 0], [0, 1j]]), matrix_0)

        s_1 = self.circuit_23.s(1)
        matrix_1 = s_1.to_matrix(identities=0)
        assert np.allclose(np.array([[1, 0, 0], [0, omega_d(3), 0], [0, 0, 1]]), matrix_1)

    def test_validate_parameter(self):
        s = self.circuit_23.s(0)
        assert s.validate_parameter()
