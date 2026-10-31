import numpy as np
import matplotlib.pyplot as plt
import src.simulator as sim

# ---- same parameters as simulator.py ----
t_c, f_c, m_prop, m_total = sim.load_eng("data/motor_details.eng")
cum = np.concatenate(([0.0], np.cumsum(np.diff(t_c) * (f_c[1:] + f_c[:-1]) / 2)))
p = {"t": t_c, "f": f_c, "cum_impulse": cum,
     "m_dry": 0.856 + (m_total - m_prop), "m_prop": m_prop,
     "area": np.pi / 4 * 0.0762 ** 2, "cd": 0.40}
burn = t_c[-1]

# ---- Test 1: no drag. Coast gain must equal v^2 / (2g) ----
p["cd"] = 0.0
t, h, v = sim.simulate(p)
i = np.argmax(t >= burn)
expected = h[i] + v[i] ** 2 / (2 * sim.G0)
print(f"Test 1 (no drag): sim {h.max():.4f} m | analytic {expected:.4f} m | diff {h.max() - expected:+.4f} m")

# ---- Test 2: constant air density. Coast gain has an exact formula ----
rho = 1.225
original_density = sim.air_density
sim.air_density = lambda h: rho            # temporarily force constant density
p["cd"] = 0.40
t, h, v = sim.simulate(p)
i = np.argmax(t >= burn)
k = 0.5 * rho * p["cd"] * p["area"]
m = p["m_dry"]
expected_gain = (m / (2 * k)) * np.log(1 + k * v[i] ** 2 / (m * sim.G0))
print(f"Test 2 (drag, constant density): sim {h.max() - h[i]:.3f} m | analytic {expected_gain:.3f} m")
sim.air_density = original_density         # put the real atmosphere back

# ---- Test 3: time-step convergence, RK4 vs Euler ----
def apogee_euler(p, dt):
    t, y, top = 0.0, np.array([0.0, 0.0]), 0.0
    while t < 120:
        y = y + dt * sim.derivs(t, y, p)
        t += dt
        top = max(top, y[0])
        if y[1] < 0 and t > 0.5:
            break
    return top

dts = [0.1, 0.05, 0.02, 0.01, 0.005]
ref = sim.simulate(p, dt=0.001)[1].max()   # fine-step reference
err_rk4 = [abs(sim.simulate(p, dt=d)[1].max() - ref) for d in dts]
err_euler = [abs(apogee_euler(p, d) - ref) for d in dts]

plt.loglog(dts, err_euler, "o-", label="Euler")
plt.loglog(dts, err_rk4, "s-", label="RK4")
plt.xlabel("Time step dt (s)"); plt.ylabel("Apogee error (m)")
plt.legend(); plt.grid(True, which="both"); 
plt.tight_layout()
plt.savefig("figures/time_step_convergence.png",dpi=200)
plt.show()