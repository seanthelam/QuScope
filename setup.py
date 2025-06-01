import setuptools
import os

# Function to parse requirements.txt
def parse_requirements(filename="requirements.txt"):
    """Load requirements from a pip requirements file."""
    with open(filename, "r") as f:
        lines = (line.strip() for line in f)
        # Remove comments and empty lines
        requirements = [line for line in lines if line and not line.startswith("#")]
    return requirements

# Get the long description from the README file
def get_long_description(readme_path="README.md"):
    """Reads the README file for the long description."""
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as fh:
            long_description = fh.read()
        return long_description
    return "QuScope: Quantum Algorithms for Advanced Electron Microscopy"

# Package version
VERSION = "0.2.0" # Updated version after refactoring

# Author information
AUTHOR_NAME = "Roberto Reis"
AUTHOR_EMAIL = "roberto.rms.reis@gmail.com" # Placeholder, user can update

# Package name
PACKAGE_NAME = "quscope"
PROJECT_URLS = {
    "Bug Tracker": "https://github.com/rmsreis/quantum_algo_microscopy/issues",
    "Source Code": "https://github.com/rmsreis/quantum_algo_microscopy",
    # "Documentation": "LINK_TO_HOSTED_DOCS_IF_AVAILABLE" # Placeholder
}

# Short description
DESCRIPTION = (
    "A Python package for applying quantum computing algorithms to "
    "electron microscopy image processing and EELS analysis."
)

# Long description
LONG_DESCRIPTION = get_long_description()
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"

# Dependencies
INSTALL_REQUIRES = parse_requirements()

# Python version requirement
PYTHON_REQUIRES = "~=3.8" # Python 3.8 or higher

# Keywords for PyPI
KEYWORDS = [
    "quantum computing",
    "electron microscopy",
    "qiskit",
    "image processing",
    "eels analysis",
    "quantum algorithms",
    "scientific computing",
    "materials science",
    "piqture",
    "quantum machine learning"
]

# Classifiers for PyPI
CLASSIFIERS = [
    "Development Status :: 3 - Alpha", # Indicates active development, potentially unstable
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Physics",
    "Topic :: Scientific/Engineering :: Image Processing",
    "Topic :: Software Development :: Libraries :: Python Modules",
]

setuptools.setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author=AUTHOR_NAME,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    url=PROJECT_URLS["Source Code"],
    project_urls=PROJECT_URLS,
    package_dir={"": "src"},  # Tell setuptools that packages are under src
    packages=setuptools.find_packages(where="src"), # Automatically find packages in src
    classifiers=CLASSIFIERS,
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    keywords=KEYWORDS,
    include_package_data=True, # To include non-code files specified in MANIFEST.in (if any)
    zip_safe=False, # Recommended for better compatibility
    # entry_points={ # No command-line scripts defined for now
    #     'console_scripts': [
    #         'quscope-cli=quscope.cli:main', # Example if a CLI is added
    #     ],
    # },
)
