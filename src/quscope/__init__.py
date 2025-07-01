"""QuScope: Quantum algorithms for microscopy image processing and EELS analysis."""

from importlib.metadata import version as _pkg_version, PackageNotFoundError as _PkgNotFoundError

try:
    __version__ = _pkg_version("quscope")
except _PkgNotFoundError:
    __version__ = "0.0.0+dev"
