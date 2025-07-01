"""Tests for quantum microscopy implementation."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.quantum.microscopy import QuantumMicroscopy
from src.config import Config, QuantumConfig
from src.exceptions import ValidationError, QuantumCircuitError

class TestQuantumMicroscopy:
    """Test suite for QuantumMicroscopy class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config(quantum=QuantumConfig(shots=1024, max_qubits=10))
    
    @pytest.fixture
    def microscopy(self, config):
        """Create QuantumMicroscopy instance."""
        return QuantumMicroscopy(config)
    
    def test_initialization(self, microscopy):
        """Test proper initialization."""
        assert microscopy.config is not None
        assert microscopy.circuit is None
        assert microscopy.results is None
    
    def test_validate_parameters_success(self, microscopy):
        """Test successful parameter validation."""
        params = {
            'num_qubits': 5,
            'target_function': lambda x: x % 2 == 0
        }
        # Should not raise an exception
        microscopy.validate_parameters(**params)
    
    def test_validate_parameters_missing_required(self, microscopy):
        """Test parameter validation with missing required parameters."""
        with pytest.raises(ValidationError, match="Missing required parameter"):
            microscopy.validate_parameters(num_qubits=5)
    
    def test_validate_parameters_invalid_qubits(self, microscopy):
        """Test parameter validation with invalid qubit count."""
        params = {
            'num_qubits': -1,
            'target_function': lambda x: True
        }
        with pytest.raises(ValidationError, match="must be a positive integer"):
            microscopy.validate_parameters(**params)
    
    def test_validate_parameters_too_many_qubits(self, microscopy):
        """Test parameter validation with too many qubits."""
        params = {
            'num_qubits': 20,  # Exceeds max_qubits=10
            'target_function': lambda x: True
        }
        with pytest.raises(ValidationError, match="exceeds maximum"):
            microscopy.validate_parameters(**params)
    
    def test_build_circuit(self, microscopy):
        """Test circuit building."""
        params = {
            'num_qubits': 3,
            'target_function': lambda x: x % 2 == 0
        }
        circuit = microscopy.build_circuit(**params)
        
        assert circuit.num_qubits == 3
        assert circuit.num_clbits == 3
        assert circuit.depth() > 0
    
    def test_compute_fidelity(self, microscopy):
        """Test fidelity computation."""
        probabilities = {'000': 0.5, '001': 0.3, '010': 0.2}
        fidelity = microscopy._compute_fidelity(probabilities)
        assert fidelity == 0.5
    
    def test_compute_entropy(self, microscopy):
        """Test entropy computation."""
        probabilities = {'000': 0.5, '001': 0.5}
        entropy = microscopy._compute_entropy(probabilities)
        assert np.isclose(entropy, 1.0)  # Maximum entropy for 2 equal states
    
    def test_get_resolution_enhancement(self, microscopy):
        """Test resolution enhancement calculation."""
        enhancement = microscopy.get_resolution_enhancement(num_qubits=3)
        expected = 1.0 / (1.0 / 8)  # 2^3 = 8
        assert enhancement == expected
    
    @patch('src.quantum.microscopy.logger')
    def test_error_handling_in_build_circuit(self, mock_logger, microscopy):
        """Test error handling in circuit building."""
        # Force an error by passing invalid parameters
        with pytest.raises(QuantumCircuitError):
            microscopy.build_circuit()  # Missing required parameters
