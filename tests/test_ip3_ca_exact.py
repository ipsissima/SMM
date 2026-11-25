"""Independent checks for the exact (non-telegraph) two-field IP3/Ca model."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from smm.ip3_ca_exact import IP3CaLinearParams

@pytest.mark.parametrize('k2', [0, 1e-4, .001, .01, .1])
def test_modal_eigenvalues_agree_with_exact_elimination(k2):
    p=IP3CaLinearParams()
    assert np.allclose(np.sort_complex(p.modal_eigenvalues(k2)),
                       np.sort_complex(np.roots(p.eliminated_characteristic_coefficients(k2))))

@pytest.mark.parametrize('k2',[0,.001,.01,.1])
def test_cooperative_system_has_real_poles(k2):
    p=IP3CaLinearParams()
    assert p.cooperative_discriminant(k2)>0
    assert np.max(np.abs(p.modal_eigenvalues(k2).imag))<1e-12

def test_stability_and_failure_of_wave_claim():
    p=IP3CaLinearParams(alpha=1.,beta=.8,gamma=.9,delta=2.,D_um2_per_s=100.)
    assert p.is_locally_stable()
    assert np.max(p.modal_eigenvalues(0).real)<0
    assert p.characteristic_diffusive_length_um() is None

def test_invalid_wave_numbers():
    p=IP3CaLinearParams()
    for k2 in [-.01,float('nan')]:
        with pytest.raises(ValueError): p.modal_matrix(k2)
