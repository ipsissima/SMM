"""
Syncytium Mesh Model (SMM) - A multi-layer neural dynamics model.

This package implements a three-layer model combining:
1. Neural mass dynamics (Wilson-Cowan-like)
2. Kuramoto phase oscillators on a structural connectome
3. A 2D continuous mesh field governed by a damped wave PDE

Contact: Andreu.Ballus@uab.cat
"""

__version__ = "0.1.0"

from . import mesh
from . import neural
from . import coupling
from . import analysis

__all__ = ["mesh", "neural", "coupling", "analysis"]
