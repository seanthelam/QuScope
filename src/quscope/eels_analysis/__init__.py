"""Quantum EELS (Electron Energy Loss Spectroscopy) analysis module."""

from .preprocessing import preprocess_eels_data
from .quantum_processing import create_eels_circuit, analyze_eels_spectrum

__all__ = [
    'preprocess_eels_data',
    'create_eels_circuit',
    'analyze_eels_spectrum'
]
