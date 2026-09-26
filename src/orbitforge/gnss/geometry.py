from __future__ import annotations
import math

from orbitforge.navigation.least_squares import solve_linear


def _invert(a):
    n = len(a)
    return [solve_linear(a, [1.0 if i == j else 0.0 for j in range(n)]) for i in range(n)]


def _enu_rotation(lat_rad: float, lon_rad: float):
    """Rotation taking an ECEF displacement into local East-North-Up."""
    sl, cl = math.sin(lat_rad), math.cos(lat_rad)
    so, co = math.sin(lon_rad), math.cos(lon_rad)
    return [
        [-so, co, 0.0],
        [-sl * co, -sl * so, cl],
        [cl * co, cl * so, sl],
    ]


def geometry_matrix(views, sat_positions_ecef, receiver_ecef):
    """Rows [-los_x, -los_y, -los_z, 1] for each satellite used in a fix."""
    rows = []
    for view in views:
        sat = sat_positions_ecef[view.satellite_id]
        line_of_sight = (sat - receiver_ecef)
        rng = line_of_sight.norm()
        ux, uy, uz = (line_of_sight.x / rng, line_of_sight.y / rng, line_of_sight.z / rng)
        rows.append([-ux, -uy, -uz, 1.0])
    return rows


def dilution_of_precision(design, lat_rad: float, lon_rad: float, sigma_km=None):
    """GDOP/PDOP/HDOP/VDOP/TDOP plus the 4x4 covariance factor ``(H^T W H)^-1``.

    ``sigma_km`` optionally weights each row (weighted least squares); omitting
    it gives the classic unit-weight geometric DOP.
    """
    if not design:
        raise ValueError('at least one geometry row required')
    m = len(design)
    if sigma_km is not None and len(sigma_km) != m:
        raise ValueError('sigma must match design row count')

    weights = [1.0 / (s * s) for s in sigma_km] if sigma_km is not None else [1.0] * m
    normal = [[0.0 for _ in range(4)] for _ in range(4)]
    for row, w in zip(design, weights):
        for i in range(4):
            for j in range(4):
                normal[i][j] += w * row[i] * row[j]
    q = _invert(normal)  # ECEF x,y,z and time components

    r_enu = _enu_rotation(lat_rad, lon_rad)
    q_xyz = [row[:3] for row in q[:3]]
    # Rotate the position block of the covariance into the local frame.
    q_enu = [[0.0] * 3 for _ in range(3)]
    rr = r_enu
    for i in range(3):
        for j in range(3):
            q_enu[i][j] = sum(rr[i][a] * q_xyz[a][b] * rr[j][b] for a in range(3) for b in range(3))

    var_e = q_enu[0][0]
    var_n = q_enu[1][1]
    var_u = q_enu[2][2]
    var_t = q[3][3]
    pdop2 = var_e + var_n + var_u
    return {
        'Q': q,
        'GDOP': math.sqrt(pdop2 + var_t),
        'PDOP': math.sqrt(pdop2),
        'HDOP': math.sqrt(var_e + var_n),
        'VDOP': math.sqrt(max(var_u, 0.0)),
        'TDOP': math.sqrt(max(var_t, 0.0)),
    }
