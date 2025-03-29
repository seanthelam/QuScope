"""Quantum encoding methods for image processing."""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


def encode_image_to_circuit(img_array):
    """Encode an image into a quantum circuit.
    
    This function takes a preprocessed image array and encodes it into
    a quantum circuit using amplitude encoding.
    
    Args:
        img_array (numpy.ndarray): Preprocessed image array (normalized).
        
    Returns:
        qiskit.QuantumCircuit: Quantum circuit with encoded image.
    """
    # Flatten the image and ensure it's normalized
    img_flat = img_array.flatten()
    img_norm = img_flat / np.linalg.norm(img_flat)
    
    # Calculate number of qubits needed
    n_qubits = int(np.ceil(np.log2(len(img_norm))))
    
    # Pad the vector if needed
    padded_length = 2**n_qubits
    if len(img_norm) < padded_length:
        img_norm = np.pad(img_norm, (0, padded_length - len(img_norm)))
        img_norm = img_norm / np.linalg.norm(img_norm)  # Renormalize
    
    # Create quantum circuit
    qr = QuantumRegister(n_qubits, 'q')
    cr = ClassicalRegister(n_qubits, 'c')
    circuit = QuantumCircuit(qr, cr)
    
    # Initialize with amplitude encoding
    # Note: In a real implementation, you would use qiskit's
    # initialize() method or build a custom circuit to encode
    # the amplitudes properly
    circuit.initialize(img_norm, qr)
    
    return circuit


def encode_binary_image(binary_img):
    """Encode a binary image into a quantum circuit using basis states.
    
    Args:
        binary_img (numpy.ndarray): Binary image array with values 0 and 1.
        
    Returns:
        qiskit.QuantumCircuit: Quantum circuit with encoded binary image.
    """
    # Flatten the image
    img_flat = binary_img.flatten()
    
    # Calculate number of qubits needed (one per pixel)
    n_qubits = len(img_flat)
    
    # Create quantum circuit
    qr = QuantumRegister(n_qubits, 'q')
    cr = ClassicalRegister(n_qubits, 'c')
    circuit = QuantumCircuit(qr, cr)
    
    # Apply X gates where pixel value is 1
    for i, pixel in enumerate(img_flat):
        if pixel == 1:
            circuit.x(qr[i])
    
    return circuit
