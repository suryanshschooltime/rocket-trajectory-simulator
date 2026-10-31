import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

G0 = 9.80665
FT = 0.3048  # feet -> meters

def load_eng(path):
    """Read a RASP .eng motor file from ThrustCurve.org."""
    t, f, header = [], [], None
    with open(path) as file:
        for line in file:
            s = line.strip()
            if not s or s.startswith(";"):
                continue
            parts = s.split()
            if header is None:
                header = parts
            elif len(parts) == 2:
                t.append(float(parts[0])); f.append(float(parts[1]))
    if t[0] > 0:                       # make sure the curve starts at t = 0
        t.insert(0, 0.0); f.insert(0, 0.0)
    return np.array(t), np.array(f), float(header[4]), float(header[5])

def load_openrocket_csv(path):
    header, rows = None, []
    with open(path) as file:
        for line in file:
            s = line.strip()
            if not s:
                continue
            body = s.lstrip("#").strip()
            if body.startswith("Time"):
                header = [c.strip() for c in body.split(",")]
            elif s.startswith("#"):
                continue
            else:
                rows.append([float(x) for x in s.split(",")])
    return pd.DataFrame(rows, columns=header)

def column_si(df, start):
    """Get a column by the start of its name; convert feet to meters."""
    col = next(c for c in df.columns if c.startswith(start))
    vals = df[col].to_numpy()
    unit = col[col.find("(") + 1 : col.find(")")]
    return vals * FT if unit in ("ft", "ft/s") else vals

def air_density(h):
    return 1.225 * (1 - 2.25577e-5 * h) ** 4.25588

def derivs(t, y, p):
    h, v = y
    burned = np.interp(t, p["t"], p["cum_impulse"]) / p["cum_impulse"][-1]
    m = p["m_dry"] + p["m_prop"] * (1 - burned)      # mass falls as impulse is delivered
    T = np.interp(t, p["t"], p["f"], right=0.0)
    drag = 0.5 * air_density(max(h, 0)) * p["cd"] * p["area"] * v * abs(v)
    a = (T - drag) / m - G0
    if h <= 0 and v <= 0 and a < 0:                  # sitting on the pad
        return np.array([0.0, 0.0])
    return np.array([v, a])

def rk4_step(f, t, y, dt, p):
    k1 = f(t, y, p)
    k2 = f(t + dt / 2, y + dt / 2 * k1, p)
    k3 = f(t + dt / 2, y + dt / 2 * k2, p)
    k4 = f(t + dt, y + dt * k3, p)
    return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

def simulate(p, dt=0.01, t_max=120):
    t, y = 0.0, np.array([0.0, 0.0])
    T_, H, V = [t], [y[0]], [y[1]]
    while t < t_max:
        y = rk4_step(derivs, t, y, dt, p)
        t += dt
        T_.append(t); H.append(y[0]); V.append(y[1])
        if y[1] < 0 and t > 0.5:                     # past apogee
            break
    return np.array(T_), np.array(H), np.array(V)

if __name__ == "__main__":
    # ---- inputs: update these from OpenRocket after your fixes ----
    ROCKET_MASS = 0.856        # kg, "Mass with no motors"
    DIAMETER = 0.0762          # m (3 in)

    t_c, f_c, m_prop, m_total = load_eng("motor_details.eng")
    cum = np.concatenate(([0.0], np.cumsum(np.diff(t_c) * (f_c[1:] + f_c[:-1]) / 2)))
    p = {"t": t_c, "f": f_c, "cum_impulse": cum,
         "m_dry": ROCKET_MASS + (m_total - m_prop),  # rocket + empty motor casing
         "m_prop": m_prop,
         "area": np.pi / 4 * DIAMETER ** 2,
         "cd": 0.6}
    print(f"Liftoff mass: {p['m_dry'] + p['m_prop']:.3f} kg (should match OpenRocket's 'Mass with motors')")
    print(f"Total impulse: {cum[-1]:.0f} N*s")

    #climb to apogee only
    df = load_openrocket_csv("Best_open_Rocket.csv")
    t_or, h_or = column_si(df, "Time"), column_si(df, "Altitude")
    i_ap = np.argmax(h_or)
    cd_or = np.nanmean(column_si(df, "Drag coefficient")[: i_ap + 1])
    t_or, h_or = t_or[: i_ap + 1], h_or[: i_ap + 1]

    #Uncalibrated prediction frm OpnRcket avg Cd
    p["cd"] = cd_or
    t0, h0, v0 = simulate(p)
    err0 = 100 * (h0.max() - h_or.max()) / h_or.max()
    print(f"Cd = {cd_or:.2f} (from OpenRocket): apogee {h0.max():.0f} m vs {h_or.max():.0f} m ({err0:+.1f}%)")

    #Calibrated fit of Cd to the OpnRcket altitude curve
    def rmse(cd):
        p["cd"] = cd
        t, h, _ = simulate(p)
        keep = t <= t_or[-1]
        return np.sqrt(np.mean((h[keep] - np.interp(t[keep], t_or, h_or)) ** 2))

    best = minimize_scalar(rmse, bounds=(0.2, 1.2), method="bounded")
    p["cd"] = best.x
    t, h, v = simulate(p)
    print(f"Fitted Cd = {best.x:.3f}, RMSE = {best.fun:.1f} m, max speed = {v.max():.0f} m/s")

    plt.plot(t_or, h_or, label="OpenRocket")
    plt.plot(t, h, "--", label="My simulator (fitted Cd)")
    plt.xlabel("Time (s)"); plt.ylabel("Altitude (m)"); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/altitude_comparison.png",dpi=200)
    plt.show()