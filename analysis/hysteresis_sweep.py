# analysis/hysteresis_sweep.py
import numpy as np
import matplotlib.pyplot as plt


def sweep_parameter_for_hysteresis(
    param_name="gamma_bg",
    sweep_values_up=None,
    sweep_values_down=None,
    simulator=None,
    e_star=None,
    dt=0.001,
    TperStep=0.5,
):
    if sweep_values_up is None:
        sweep_values_up = np.linspace(simulator.gamma_bg * 0.5, simulator.gamma_bg * 2.0, 40)
    if sweep_values_down is None:
        sweep_values_down = sweep_values_up[::-1]

    e_field = e_star.reshape((simulator.ny, simulator.nx))

    def run_sweep(svals, u_init=None, v_init=None):
        A_trace = []
        u_state, v_state = u_init, v_init
        for val in svals:
            setattr(simulator, param_name, val)
            if param_name == "gamma_bg":
                simulator._build_gamma()
            traces, u_state, v_state = simulator.run(
                T=TperStep,
                dt=dt,
                u0=u_state,
                v0=v_state,
                record_every=1,
            )
            last = traces[-1]
            A = float(np.sum(last * e_field))
            A_trace.append(A)
        return np.array(A_trace), u_state, v_state

    A_up, u_last, v_last = run_sweep(sweep_values_up)
    A_down, _, _ = run_sweep(sweep_values_down, u_init=u_last, v_init=v_last)
    return sweep_values_up, A_up, sweep_values_down, A_down


def plot_hysteresis(alpha_up, A_up, alpha_down, A_down, outname="hysteresis.png"):
    plt.figure(figsize=(6, 4))
    plt.plot(alpha_up, A_up, "-o", label="up-sweep")
    plt.plot(alpha_down, A_down, "-o", label="down-sweep")
    plt.xlabel("control (alpha)")
    plt.ylabel("modal amplitude A")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outname, dpi=150)
    plt.close()
