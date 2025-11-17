#!/usr/bin/env python3
# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Demo script for the improved ASCII circuit visualization.

This script demonstrates the new ASCII art circuit drawer with PDF export.
"""

from __future__ import annotations

from mqt.qudits.quantum_circuit import QuantumCircuit
from mqt.qudits.quantum_circuit.components.quantum_register import QuantumRegister
from mqt.qudits.visualisation import (
    draw_circuit_to_console,
    export_circuit_to_pdf,
    export_circuit_to_text,
)


def demo_simple_circuit() -> None:
    """Demo with a simple 2-qudit circuit."""
    qreg = QuantumRegister("simple", 2, [3, 3])
    circuit = QuantumCircuit(qreg)

    circuit.h(0)
    circuit.x(0)
    circuit.cx([0, 1])
    circuit.x(1)
    circuit.s(0)

    # Display to console
    draw_circuit_to_console(circuit)

    # Export to text file
    export_circuit_to_text(circuit, "simple_circuit.txt")

    # Export to PDF
    export_circuit_to_pdf(
        circuit,
        "simple_circuit.pdf",
        title="Simple 2-Qudit Circuit",
    )


def demo_complex_circuit() -> None:
    """Demo with a more complex circuit."""
    qreg = QuantumRegister("complex", 4, [3, 3, 4, 2])
    circuit = QuantumCircuit(qreg)

    # Layer 1: Single-qudit gates
    circuit.h(0)
    circuit.h(1)
    circuit.x(2)
    circuit.z(3)

    # Layer 2: More single-qudit gates
    circuit.s(0)
    circuit.s(1)
    circuit.z(2)

    # Layer 3: Two-qudit gates
    circuit.cx([0, 1])
    circuit.z(2)
    circuit.cx([2, 3])

    # Layer 4: More gates
    circuit.s(0)
    circuit.h(1)
    circuit.h(2)

    # Display to console
    draw_circuit_to_console(circuit)

    # Export to PDF
    export_circuit_to_pdf(
        circuit,
        "complex_circuit.pdf",
        title="Complex Multi-Qudit Circuit with Various Gate Types",
    )


def demo_entanglement_circuit() -> None:
    """Demo with an entanglement circuit."""
    qreg = QuantumRegister("bell", 3, [2, 2, 2])
    circuit = QuantumCircuit(qreg)

    # Create GHZ-like state
    circuit.h(0)
    circuit.cx([0, 1])
    circuit.cx([1, 2])
    circuit.z(0)
    circuit.s(1)
    circuit.x(2)

    # Display to console
    draw_circuit_to_console(circuit)

    # Export
    export_circuit_to_pdf(
        circuit,
        "entanglement_circuit.pdf",
        title="GHZ-like Entanglement Circuit",
    )


def demo_parameter_display() -> None:
    """Demo showing a variety of gates."""
    qreg = QuantumRegister("mixed", 3, [3, 3, 3])
    circuit = QuantumCircuit(qreg)

    # Various gate types
    circuit.h(0)
    circuit.x(1)
    circuit.z(2)
    circuit.s(0)
    circuit.cx([0, 1])
    circuit.cx([1, 2])
    circuit.cx([0, 2])
    circuit.x(0)
    circuit.h(1)
    circuit.s(2)

    # Display to console
    draw_circuit_to_console(circuit)

    # Export
    export_circuit_to_pdf(
        circuit,
        "mixed_gates.pdf",
        title="Mixed Gate Types",
    )


if __name__ == "__main__":
    try:
        demo_simple_circuit()
        demo_complex_circuit()
        demo_entanglement_circuit()
        demo_parameter_display()

    except ImportError:
        pass
