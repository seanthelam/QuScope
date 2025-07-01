"""
Quantum Algorithm Microscopy Package

A comprehensive toolkit for quantum algorithm analysis and visualization.
"""

__version__ = "1.0.0"
__author__ = "Roberto Reis"

from .quantum.microscopy import QuantumMicroscopy
from .config import Config

# Import quscope modules
from .quscope import *

__all__ = [
    "QuantumMicroscopy",
    "Config"
]
