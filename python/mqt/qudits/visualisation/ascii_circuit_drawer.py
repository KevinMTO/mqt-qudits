# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# Copyright (c) 2025 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""ASCII art circuit drawer for quantum circuits.

This module provides functions to draw quantum circuits using ASCII art,
with support for PDF export.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..quantum_circuit import QuantumCircuit
    from ..quantum_circuit.gate import Gate


def format_parameter(value: float, precision: int = 2) -> str:
    """Format a parameter value for display.

    Args:
        value: The parameter value
        precision: Number of decimal places

    Returns:
        Formatted string representation
    """
    # Try to represent as fractions of π
    if abs(value) < 1e-10:
        return "0"

    pi_fraction = value / np.pi
    if abs(pi_fraction - round(pi_fraction)) < 1e-6:
        n = round(pi_fraction)
        if n == 0:
            return "0"
        if n == 1:
            return "π"
        if n == -1:
            return "-π"
        return f"{n}π"

    # Check for common fractions of π
    for denom in [2, 3, 4, 6, 8]:
        frac = pi_fraction * denom
        if abs(frac - round(frac)) < 1e-6:
            num = round(frac)
            if num == 0:
                return "0"
            if denom == 1:
                return f"{num}π"
            if num == 1:
                return f"π/{denom}"
            if num == -1:
                return f"-π/{denom}"
            return f"{num}π/{denom}"

    # Fall back to decimal
    return f"{value:.{precision}f}"


def get_gate_label(gate: Gate) -> str:
    """Generate a label for a gate based on its type and parameters.

    Args:
        gate: The gate to generate a label for

    Returns:
        String label for the gate
    """
    name = gate.qasm_tag.upper()

    # Handle gates with parameters
    if hasattr(gate, "phi") and hasattr(gate, "theta"):
        # R gate with theta and phi
        theta_str = format_parameter(gate.theta)
        phi_str = format_parameter(gate.phi)
        return f"{name}({theta_str},{phi_str})"

    if hasattr(gate, "phi"):
        # Single parameter rotation
        phi_str = format_parameter(gate.phi)
        return f"{name}({phi_str})"

    if hasattr(gate, "theta"):
        theta_str = format_parameter(gate.theta)
        return f"{name}({theta_str})"

    if hasattr(gate, "lev_a") and hasattr(gate, "lev_b"):
        # Gate with level specification
        return f"{name}{gate.lev_a}{gate.lev_b}"

    if hasattr(gate, "lev_a"):
        return f"{name}{gate.lev_a}"

    return name


def draw_circuit_ascii(circuit: QuantumCircuit, use_unicode: bool = True) -> str:
    """Draw a quantum circuit using ASCII art.

    Args:
        circuit: The quantum circuit to draw
        use_unicode: Use Unicode box-drawing characters (default: True)

    Returns:
        String containing the ASCII art representation

    Examples:
        >>> circuit = QuantumCircuit(qreg)
        >>> circuit.h(0)
        >>> circuit.cx([0, 1])
        >>> print(draw_circuit_ascii(circuit))
    """
    if use_unicode:
        hline = "─"
        vline = "│"
        lbox = "┤"
        rbox = "├"
    else:
        hline = "-"
        vline = "|"
        lbox = "|"
        rbox = "|"

    num_qudits = circuit.num_qudits
    gates = circuit.instructions

    # Calculate column positions for each gate
    columns: list[list[tuple[int, str, list[int]]]] = []  # [[(qudit, label, targets), ...], ...]

    for gate in gates:
        gate_label = get_gate_label(gate)

        if isinstance(gate.target_qudits, list):
            # Multi-qudit gate
            targets = sorted(gate.target_qudits)
            min_qudit = min(targets)
            max_qudit = max(targets)

            # Create column entry
            col_info = []
            for qudit in range(min_qudit, max_qudit + 1):
                if qudit in targets:
                    # Determine if control or target
                    if len(targets) == 2 and qudit == targets[0]:
                        col_info.append((qudit, "●", targets))  # Control
                    else:
                        col_info.append((qudit, gate_label, targets))  # Target
                else:
                    col_info.append((qudit, vline, targets))  # Connection line

            columns.append(col_info)
        else:
            # Single qudit gate
            qudit = gate.target_qudits
            columns.append([(qudit, gate_label, [qudit])])

    # Build the circuit representation
    lines = []

    # Header with qudit labels
    header = "Circuit Diagram:\n"
    header += "=" * (20 + len(columns) * 15) + "\n\n"
    lines.append(header)

    # Draw each qudit line
    for q in range(num_qudits):
        # Initial state
        dim = circuit.dimensions[q]
        line = f"q{q}[d={dim}]: |0⟩{hline * 3}"

        # Add gates
        for col in columns:
            gate_on_qudit = None
            for qudit, label, targets in col:
                if qudit == q:
                    gate_on_qudit = (label, targets)
                    break

            if gate_on_qudit:
                label, targets = gate_on_qudit
                if label == "●":
                    # Control qudit
                    line += f"{hline}{hline}●{hline}{hline}"
                elif label == vline:
                    # Connection line for multi-qudit gate
                    line += f"{hline}{hline}{vline}{hline}{hline}"
                else:
                    # Gate box
                    box_content = f" {label} "
                    padding = max(0, 12 - len(box_content))
                    left_pad = padding // 2
                    right_pad = padding - left_pad
                    line += f"{hline}{' ' * left_pad}{rbox}{label}{lbox}{' ' * right_pad}{hline}"
            else:
                # Empty space (wire continues)
                line += f"{hline * 12}{hline}"

        line += f"{hline * 3}"
        lines.append(line)

    # Footer
    lines.extend((
        "\n" + "=" * (20 + len(columns) * 15),
        f"\nTotal gates: {len(gates)}",
        f"Circuit depth: {len(columns)}",
        f"Qudits: {num_qudits}",
    ))

    return "\n".join(lines)


def draw_circuit_to_console(circuit: QuantumCircuit, use_unicode: bool = True) -> None:
    """Print a quantum circuit to the console using ASCII art.

    Args:
        circuit: The quantum circuit to draw
        use_unicode: Use Unicode box-drawing characters (default: True)
    """
    print(draw_circuit_ascii(circuit, use_unicode))


def export_circuit_to_pdf(
    circuit: QuantumCircuit,
    filename: str,
    title: str | None = None,
    include_stats: bool = True,
) -> None:
    """Export a quantum circuit diagram to PDF.

    Args:
        circuit: The quantum circuit to export
        filename: Output PDF filename
        title: Optional title for the diagram
        include_stats: Include circuit statistics (default: True)

    Raises:
        ImportError: If reportlab is not installed
    """
    try:
        from reportlab.lib.pagesizes import letter  # type: ignore[import-untyped]
        from reportlab.pdfgen import canvas  # type: ignore[import-untyped]
    except ImportError as e:
        msg = "reportlab is required for PDF export. Install it with: pip install reportlab"
        raise ImportError(msg) from e

    # Create PDF
    c = canvas.Canvas(filename, pagesize=letter)
    _width, height = letter

    # Title
    y_position = height - 50
    if title:
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, y_position, title)
        y_position -= 30

    # Circuit ASCII art
    c.setFont("Courier", 9)
    ascii_circuit = draw_circuit_ascii(circuit, use_unicode=False)

    for line in ascii_circuit.split("\n"):
        if y_position < 100:  # Start new page if needed
            c.showPage()
            c.setFont("Courier", 9)
            y_position = height - 50

        c.drawString(50, y_position, line)
        y_position -= 12

    # Statistics
    if include_stats:
        y_position -= 20
        c.setFont("Helvetica", 10)

        stats = [
            f"Number of qudits: {circuit.num_qudits}",
            f"Qudit dimensions: {circuit.dimensions}",
            f"Total gates: {len(circuit.instructions)}",
            f"Circuit depth: {len(circuit.instructions)}",  # Simplified
        ]

        # Gate type breakdown
        gate_types: dict[str, int] = {}
        for gate in circuit.instructions:
            name = gate.qasm_tag
            gate_types[name] = gate_types.get(name, 0) + 1

        stats.append("\nGate breakdown:")
        for gate_name, count in sorted(gate_types.items()):
            stats.append(f"  {gate_name}: {count}")

        for stat in stats:
            if y_position < 50:
                c.showPage()
                c.setFont("Helvetica", 10)
                y_position = height - 50

            c.drawString(50, y_position, stat)
            y_position -= 15

    c.save()
    print(f"✅ Circuit diagram exported to: {filename}")


def export_circuit_to_text(
    circuit: QuantumCircuit,
    filename: str,
    use_unicode: bool = True,
) -> None:
    """Export a quantum circuit diagram to a text file.

    Args:
        circuit: The quantum circuit to export
        filename: Output text filename
        use_unicode: Use Unicode box-drawing characters (default: True)
    """
    ascii_circuit = draw_circuit_ascii(circuit, use_unicode)

    with pathlib.Path(filename).open("w", encoding="utf-8") as f:
        f.write(ascii_circuit)

    print(f"✅ Circuit diagram exported to: {filename}")
