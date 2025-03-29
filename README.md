# QuScope: Quantum Algorithms for Microscopy

## Overview

This repository contains quantum computing algorithms for microscopy image processing and EELS (Electron Energy Loss Spectroscopy) analysis using Qiskit. The project demonstrates how quantum computing techniques can be applied to scientific image analysis and spectroscopy data.

Key features:
- Quantum image encoding and processing
- Image segmentation using Grover's algorithm
- EELS data analysis with quantum techniques
- Quantum Machine Learning module using PiQture (INEQR encoding)
- Jupyter notebooks with examples and visualizations

## Repository Structure

```
quscope/
├── data/                    # Data directory
│   └── images/              # Sample images for processing
├── notebooks/               # Jupyter notebooks with examples
├── src/                     # Source code
│   └── quscope/             # Main package source
│       ├── image_processing/  # Image processing modules
│       ├── eels_analysis/     # EELS analysis modules
│       └── qml/               # Quantum Machine Learning modules
└── tests/                   # Unit tests (to be added)
```

## Installation

### Prerequisites
- Python 3.7 or higher
- Qiskit
- NumPy
- Matplotlib
- Pillow (PIL)
- SciPy
- Jupyter
- PyTorch
- PiQture

### Setup

1. Clone the repository:
```bash
git clone https://github.com/rmsreis/quantum_algo_microscopy.git # Consider renaming repo later
cd quantum_algo_microscopy
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install the package and dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

### Image Processing

```python
from quscope.image_processing import preprocess_image, encode_image_to_circuit
from quscope.image_processing import apply_grovers_algorithm, interpret_results

# Preprocess an image
img_array = preprocess_image('path/to/image.jpg', size=(8, 8))

# Encode the image into a quantum circuit
circuit = encode_image_to_circuit(img_array)

# Apply Grover's algorithm for segmentation
target_pixels = [0, 1, 4, 5]  # Example target pixels
segmentation_circuit = apply_grovers_algorithm(circuit, target_pixels)

# Execute the circuit and interpret results
# (This would require a Qiskit backend)
```

### EELS Analysis

```python
from quscope.eels_analysis import preprocess_eels_data, create_eels_circuit

# Preprocess EELS data
processed_data = preprocess_eels_data(eels_spectrum)

# Create quantum circuit for EELS analysis
eels_circuit = create_eels_circuit(processed_data)

# Further analysis...
```

### Quantum Machine Learning (QML) - Image Encoding

```python
from quscope.image_processing import preprocess_image
from quscope.qml import encode_image_ineqr

# Preprocess an image (e.g., 8x8)
img_array = preprocess_image('path/to/image.jpg', size=(8, 8))

# Encode using INEQR (from PiQture)
ineqr_circuit = encode_image_ineqr(img_array)

# Use the circuit in a QML model...
```

## Jupyter Notebooks

The repository includes Jupyter notebooks that demonstrate the algorithms:

- `notebooks/image_processing.ipynb`: (Needs update) Quantum image processing examples
- `notebooks/quanum_EELS_Analysis.ipynb`: (Needs update) Quantum EELS analysis examples
- `notebooks/wpo_quantum1D.ipynb`: (Needs update) Additional quantum processing examples
- `notebooks/qml_image_encoding_example.ipynb`: Example of QML image encoding using PiQture/INEQR

To run the notebooks:
```bash
jupyter notebook
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this code in your research, please cite:

```
@software{quscope,
  author = {Roberto dos Reis},
  title = {QuScope: Quantum Algorithms for Electron Microscopy},
  year = {2023},
  url = {https://github.com/rmsreis/quantum_algo_microscopy}
}
