"""Patch the pinned published Sætra--Mori Python driver to add graph-corrected
interunit ionic residual and sparse Jacobian, with original AOT physics unchanged.

Scientific status: engineering verification only; selected 7-site graphs are
synthetic, not independently measured gap-junction and projection networks.
"""
import sys
from pathlib import Path

root=Path(sys.argv[1])
source=root/"src"/"ednm"/"solver.py"
s=source.read_text()
header="from ednm_graph_adapter import correction as graph_correction\n"
if header not in s:s=header+s
needle="        rsd = norm(Res, np.inf)"
insert="""        graph_res_s, _ = graph_correction(model, graph_glial, graph_ecs, dt,
            alpha_s_, c_s, phi_s, layer="soma")
        graph_res_d, _ = graph_correction(model, graph_glial, graph_ecs, dt,
            alpha_d_, c_d, phi_d, layer="dendrite")
        Res += graph_res_s + graph_res_d
"""
assert s.count(needle)==1
s=s.replace(needle,insert+needle)
needle2="        # solve (J(x_) dx = -f(x_)"
insert2="""        _, graph_J_s = graph_correction(model, graph_glial, graph_ecs, dt,
            alpha_s_, c_s, phi_s, layer="soma")
        _, graph_J_d = graph_correction(model, graph_glial, graph_ecs, dt,
            alpha_d_, c_d, phi_d, layer="dendrite")
        A = A + graph_J_s + graph_J_d
"""
assert s.count(needle2)==1
s=s.replace(needle2,insert2+needle2)
# Define additional graph inputs in alpha_c_phi at the call site. It is
# safe to retrieve graphs from model attributes, but jitclass model does not
# have arbitrary attributes: module-level variables are deliberately used.
decl="""\n# Experiment-specific spatial graphs, set before solving.\n# Default source closed path is a zero correction.\nfrom ednm_graph_adapter import closed_path\n_graph_n = None\ngraph_glial = None\ngraph_ecs = None\n"""
s=decl+s
# The runner sets solver.graph_glial and solver.graph_ecs before solve_system().
source.write_text(s)
print("Patched graph-aware original Python driver",source)
