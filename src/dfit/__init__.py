from ._version import get_versions
from .dfit import DFit, get_distributions

__version__ = get_versions()["version"]
del get_versions
