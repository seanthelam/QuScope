"""
Quantum Algorithm Microscopy Package

A comprehensive toolkit for quantum algorithm analysis and visualization.
"""

__version__ = "1.0.0"
__author__ = "Roberto Reis"

from .quantum import QuantumMicroscopy
from .classical import ClassicalProcessor
from .visualization import VisualizationEngine
from .config import Config

__all__ = [
    "QuantumMicroscopy",
    "ClassicalProcessor", 
    "VisualizationEngine",
    "Config"
]
