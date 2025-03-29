from setuptools import setup, find_packages

setup(
    name="quscope",
    version="0.1.0",
    description="Quantum algorithms for electron microscopy data processing",
    author="Roberto Reis",
    author_email="robertomsreis@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "qiskit>=0.34.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "pillow>=8.2.0",
        "jupyter>=1.0.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "piqture>=0.1.0",
    ],
    python_requires=">=3.7",
)
