import numpy as np
import matplotlib.pyplot as plt
import simulator as sim

# ---- same parameters as before ----
t_c, f_c, m_prop, m_total = sim.load_eng("data/motor_details.eng")
cum = np.concatenate(([0.0], np.cumsum(np.diff(t_c) * (f_c[1:] + f_c[:-1]) / 2)))
base = {"t": t_c, "f": f_c, "cum_impulse": cum,
        "m_dry": 0.856 + (m_total - m_prop), "m_prop": m_prop,
        "area": np.pi / 4 * 0.0762 ** 2, "cd": 0.415}   # 0.415 = your fitted Cd
OPENROCKET_APOGEE = 1312  # m

def apogee(**changes):
    q = dict(base)
    q.update(changes)
    return sim.simulate(q, dt=0.02)[1].max()

# ---- 1) Sensitivity: change one input at a time ----
nominal = apogee()
print(f"Nominal apogee: {nominal:.1f} m\n")
inputs = [("Drag coeff (Cd)", "cd"), ("Dry mass", "m_dry"),
          ("Frontal area", "area"), ("Motor thrust", "f")]
effect_10 = {}
for label, key in inputs:
    for pct in (-10, -5, 5, 10):
        a = apogee(**{key: base[key] * (1 + pct / 100)})
        print(f"{label:16s} {pct:+3d}% -> apogee {a:7.1f} m ({100 * (a - nominal) / nominal:+.1f}%)")
        if pct == 10:
            effect_10[label] = 100 * (a - nominal) / nominal
    print()

plt.barh(list(effect_10.keys()), list(effect_10.values()))
plt.xlabel("Change in apogee for a +10% change in input (%)")
plt.axvline(0, color="k"); plt.grid(True, axis="x"); plt.tight_layout(); plt.savefig("figures/sensitivity.png", dpi =200); plt.show()

# ---- 2) Monte Carlo: 300 flights with random uncertainty (takes about a minute) ----
rng = np.random.default_rng(42)
N = 300
results = []
for _ in range(N):
    q = dict(base)
    q["cd"] = rng.normal(base["cd"], 0.08 * base["cd"])     # 8% drag uncertainty
    q["m_dry"] = base["m_dry"] * rng.normal(1, 0.02)         # 2% mass uncertainty
    q["f"] = base["f"] * rng.normal(1, 0.03)                 # 3% motor thrust variation
    results.append(sim.simulate(q, dt=0.02)[1].max())
results = np.array(results)

print(f"Monte Carlo ({N} flights): mean {results.mean():.0f} m, std {results.std():.0f} m, "
      f"90% of flights between {np.percentile(results, 5):.0f} and {np.percentile(results, 95):.0f} m")

plt.hist(results, bins=25)
plt.axvline(OPENROCKET_APOGEE, color="k", linestyle="--", label="OpenRocket (1312 m)")
plt.xlabel("Apogee (m)"); plt.ylabel("Number of flights"); plt.legend(); plt.tight_layout(); plt.savefig("figures/monte_carlo.png", dpi=200); plt.show()