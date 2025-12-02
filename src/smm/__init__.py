"""
Syncytium Mesh Model (SMM) - A multi-layer neural dynamics model.

This package implements a tripartite loop model combining:
1. Neural mass dynamics (Wilson-Cowan-like)
2. Kuramoto phase oscillators on a structural connectome
3. A 2D glial (astrocytic) field governed by a telegraph equation
   derived from IP₃/Ca reaction-diffusion dynamics

The tripartite couplings:
- Neurons → Glia: Neural activity drives glial waves
- Glia → Neurons: Gliotransmission modulates neural excitability
- Glia → Connectivity: Ca-gated plasticity reshapes network structure

Contact: Andreu.Ballus@uab.cat
"""

__version__ = "0.1.0"

from . import mesh
from . import glia
from . import neural
from . import coupling
from . import analysis

__all__ = ["mesh", "glia", "neural", "coupling", "analysis"]
