"""Measure bump speed vs commanded speed, then set velocity_gain so they match."""

import numpy as np

DIRS = {
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "+z": (0.0, 0.0, 1.0),
    "110": (1.0, 1.0, 0.0),
    "111": (1.0, 1.0, 1.0),
}

CAL_DIRS = ("+x", "111")


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("zero direction vector")
    return v / n


def _peakedness(backend) -> float:
    """max / mean activity. Near 1 means the lattice is gone."""
    s = backend.S.mean(dim=0)
    m = float(s.mean())
    return float(s.max()) / m if m > 1e-12 else float("nan")


def alpha_drive(backend, theta_dot: float) -> float:
    """How hard velocity is pushing relative to b."""
    q = backend.qan
    return float(q.velocity_gain * theta_dot / (q.offset_magnitude * q.b))


def max_bump_speed(backend) -> float:
    """Theoretical top bump speed: offset_magnitude / tau."""
    return float(backend.qan.offset_magnitude) / float(backend.tau)


def measure(backend, theta_0, direction, speed, n_meas=None, burn=None,
            periods=10.0):
    """Drive at constant speed, fit the bump velocity, put the state back."""
    dt = float(backend.qan.dt)
    tau = float(backend.tau)
    u = _unit(direction)
    theta_0 = np.asarray(theta_0, dtype=np.float64)

    period_cells = float(backend.bump_period_cells())
    period_rad = period_cells * 2.0 * np.pi / backend.n
    if burn is None:
        burn = int(np.ceil(50.0 * tau / dt))
    if n_meas is None:
        n_meas = int(np.ceil(float(periods) * period_rad / (speed * dt)))

    S0 = backend.S.clone()
    pk_before = _peakedness(backend)

    T = int(burn) + int(n_meas)
    traj = (theta_0 + u * speed * np.arange(T)[:, None] * dt) % (2.0 * np.pi)
    backend.drive(traj, display_stride=T + 1, snapshot_stride=T + 1)
    pos = backend.pos_com_unwrapped

    seg = pos[burn:]
    t = np.arange(len(seg), dtype=float) * dt
    A = np.vstack([t, np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, seg, rcond=None)
    v_hat = coef[0]
    ripple = float(np.sqrt(((seg - A @ coef) ** 2).sum(axis=1).mean()))

    steps = np.linalg.norm(np.diff(pos, axis=0), axis=1) * (backend.n / (2.0 * np.pi))
    max_step = float(steps.max()) if steps.size else 0.0

    pk_after = _peakedness(backend)
    period_after = float(backend.bump_period_cells())
    backend.S = S0.clone()

    mag = float(np.linalg.norm(v_hat))
    par = float(v_hat @ u)
    angle = (float(np.degrees(np.arccos(np.clip(par / mag, -1.0, 1.0))))
             if mag > 1e-12 else float("nan"))

    lattice_ok = (np.isfinite(pk_after) and pk_after >= 0.8 * pk_before
                  and abs(period_after - period_cells) <= 0.05 * period_cells)
    ok = bool(np.all(np.isfinite(v_hat))
              and max_step <= 0.5 * period_cells
              and lattice_ok)

    return dict(
        speed=float(speed),
        alpha=alpha_drive(backend, speed),
        g_par=par / speed,
        g_mag=mag / speed,
        angle_deg=angle,
        v_hat=v_hat,
        ripple_rad=ripple,
        max_step_cells=max_step,
        pk_before=pk_before, pk_after=pk_after,
        period_cells=period_cells, period_after=period_after,
        ok=ok,
    )


def fit_gain(backend, theta_0, speed, tol=0.02, max_iter=2, dirs=CAL_DIRS,
             **kw):
    """Measure on +x and 111, then scale velocity_gain so g ≈ 1.

    Returns (velocity_gain, g). Restores the old gain if the lattice died.
    """
    vg0 = float(backend.qan.velocity_gain)
    g = float("nan")
    for _ in range(int(max_iter)):
        rows = [measure(backend, theta_0, DIRS[d], speed, **kw) for d in dirs]
        if not all(r["ok"] for r in rows):
            backend.qan.velocity_gain = vg0
            bad = next(r for r in rows if not r["ok"])
            return float("nan"), float(bad["g_par"])
        g = float(np.mean([r["g_par"] for r in rows]))
        if not np.isfinite(g) or g < 0.1:
            backend.qan.velocity_gain = vg0
            return float("nan"), g
        if abs(g - 1.0) < tol:
            return float(backend.qan.velocity_gain), g
        backend.qan.velocity_gain /= g
    return float(backend.qan.velocity_gain), g


def sweep(backend, theta_0, speed, dirs=DIRS, speed_mults=(0.5, 1.0), **kw):
    """measure() for every direction and speed. Winner-only; just read the rows."""
    rows = []
    for mult in speed_mults:
        for name, d in dirs.items():
            r = measure(backend, theta_0, d, float(speed) * float(mult), **kw)
            r["name"] = name
            r["mult"] = float(mult)
            rows.append(r)
    return rows


def probe_edge(backend, theta_0, speed, mults=(2.0, 4.0, 8.0), dirs=CAL_DIRS,
               **kw):
    """Try faster speeds until a run fails. Winner-only; just read the rows."""
    rows = []
    for m in mults:
        for name in dirs:
            r = measure(backend, theta_0, DIRS[name], float(speed) * float(m), **kw)
            r["name"] = name
            r["mult"] = float(m)
            rows.append(r)
        if not all(r["ok"] for r in rows if r["mult"] == float(m)):
            break
    return rows
