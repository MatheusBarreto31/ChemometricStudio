"""chemometrics package for ChemometricsTool

This package is organized into modular subpackages to keep development
clean and focused: `data_input`, `preprocessing`, `processing`, `analysis`,
and `reporting`.
"""

__all__ = [
	"core",
	"data_input",
	"data_processing",
	"processing",
	"analysis",
	"reporting",
]
__version__ = "0.1.0"

from . import data_input
from . import data_processing
from . import processing
from . import reporting

# Convenience imports for quick interactive use
#from .core import *
