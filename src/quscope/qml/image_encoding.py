"""Quantum image encoding methods using PiQture."""

import numpy as np
import torch
from piqture.embeddings.image_embeddings.ineqr import INEQR
from qiskit import QuantumCircuit

def encode_image_ineqr(img_array: np.ndarray) -> QuantumCircuit:
    """Encode an image using the INEQR method from PiQture.
    
    Args:
        img_array (np.ndarray): Input image array (normalized 0-1).
        
    Returns:
        QuantumCircuit: The INEQR encoded quantum circuit.
    """
    # Ensure image is 2D
    if img_array.ndim != 2:
        raise ValueError("Input image array must be 2D.")

    # Convert image size to Python ints (not numpy ints)
    image_size = (int(img_array.shape[0]), int(img_array.shape[1]))
    
    # Convert normalized float array (0-1) to integer array (0-255)
    pixel_vals_np = (img_array * 255).round().astype(np.uint8)
    
    # Convert to Python list and wrap in another list for INEQR grayscale format
    pixel_vals_flat = pixel_vals_np.flatten().tolist()
    pixel_vals_list = [pixel_vals_flat]  # INEQR expects list[list] with single list for grayscale
    
    # Use PiQture's INEQR embedding
    embedding = INEQR(image_size, pixel_vals_list).ineqr()
    
    return embedding
