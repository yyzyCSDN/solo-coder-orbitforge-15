from __future__ import annotations
from dataclasses import dataclass
import math
from orbitforge.core.vector import Vec3
from orbitforge.core.constants import C_KM_S
from orbitforge.core.errors import NoSolutionError
from orbitforge.frames.geodetic import ecef_to_geodetic
from orbitforge.navigation.least_squares import weighted_least_squares, solve_linear
from .observation import STATUS_OK

MIN_SATELLITES = 4


@dataclass(frozen=True)
class DilutionOfPrecision:
    gdop: float
    pdop: float
    hdop: float
    vdop: float
    tdop: float


@dataclass(frozen=True)
class PositionFix:
    position_ecef_km: Vec3
    clock_bias_s: float
    dop: DilutionOfPrecision
    used_satellites: tuple
    excluded_satellites: tuple
    residuals_km: tuple
    residual_rms_km: float
    iterations: int
    converged: bool


def _invert(matrix):
    n = len(matrix)
    columns = []
    for j in range(n):
        unit = [0.0] * n
        unit[j] = 1.0
        columns.append(solve_linear(matrix, unit))
    return [[columns[c][r] for c in range(n)] for r in range(n)]


def dilution_of_precision(design, position_ecef: Vec3) -> DilutionOfPrecision:
    size = len(design[0])
    normal = [[0.0] * size for _ in range(size)]
    for row in design:
        for i in range(size):
            for j in range(size):
                normal[i][j] += row[i] * row[j]
    q = _invert(normal)
    lat, lon, _ = ecef_to_geodetic(position_ecef)
    sl, cl = math.sin(lat), math.cos(lat)
    so, co = math.sin(lon), math.cos(lon)
    rot = ((-so, co, 0.0), (-sl * co, -sl * so, cl), (cl * co, cl * so, sl))
    q_pos = [row[:3] for row in q[:3]]
    q_enu = [[sum(rot[i][k] * q_pos[k][l] * rot[j][l]
                  for k in range(3) for l in range(3)) for j in range(3)] for i in range(3)]
    return DilutionOfPrecision(
        gdop=math.sqrt(max(q[0][0] + q[1][1] + q[2][2] + q[3][3], 0.0)),
        pdop=math.sqrt(max(q[0][0] + q[1][1] + q[2][2], 0.0)),
        hdop=math.sqrt(max(q_enu[0][0] + q_enu[1][1], 0.0)),
        vdop=math.sqrt(max(q_enu[2][2], 0.0)),
        tdop=math.sqrt(max(q[3][3], 0.0)))


def _solve_once(usable, position: Vec3, clock_km: float, tolerance_km: float, max_iterations: int):
    x = [position.x, position.y, position.z, clock_km]
    design, postfit, rms = [], [], 0.0
    iterations, converged = 0, False
    sigma = [o.sigma_km for o in usable]
    for iterations in range(1, max_iterations + 1):
        design = []
        residual = []
        for o in usable:
            d = Vec3(x[0] - o.sat_ecef_km.x, x[1] - o.sat_ecef_km.y, x[2] - o.sat_ecef_km.z)
            rho = d.norm()
            if rho < 1e-9:
                raise NoSolutionError('receiver collocated with satellite ' + o.prn)
            design.append([d.x / rho, d.y / rho, d.z / rho, 1.0])
            corrected = o.pseudorange_km + o.sat_clock_km
            residual.append(corrected - (rho + x[3]))
        try:
            step, postfit, rms = weighted_least_squares(design, residual, sigma)
        except ValueError as exc:
            raise NoSolutionError('singular satellite geometry') from exc
        x = [xi + dxi for xi, dxi in zip(x, step)]
        if math.sqrt(sum(d * d for d in step)) < tolerance_km:
            converged = True
            break
    return x, design, sigma, postfit, rms, iterations, converged


def solve_position(observations, initial_ecef_km: Vec3 | None = None,
                   initial_clock_km: float = 0.0, tolerance_km: float = 1e-9,
                   max_iterations: int = 12, screen_sigma: float | None = 10.0) -> PositionFix:
    usable = []
    excluded = []
    for o in observations:
        if o.usable:
            usable.append(o)
        elif o.status == STATUS_OK:
            excluded.append((o.prn, 'invalid_pseudorange'))
        else:
            excluded.append((o.prn, o.status))
    if len(usable) < MIN_SATELLITES:
        raise NoSolutionError(
            'gnss fix requires at least %d usable satellites, got %d'
            % (MIN_SATELLITES, len(usable)))
    position = initial_ecef_km if initial_ecef_km is not None else Vec3(0.0, 0.0, 0.0)
    clock_km = initial_clock_km
    while True:
        x, design, sigma, postfit, rms, iterations, converged = _solve_once(
            usable, position, clock_km, tolerance_km, max_iterations)
        position = Vec3(x[0], x[1], x[2])
        clock_km = x[3]
        if screen_sigma is None or len(usable) <= MIN_SATELLITES:
            break
        worst = max(range(len(usable)), key=lambda i: abs(postfit[i]) / sigma[i])
        if abs(postfit[worst]) / sigma[worst] <= screen_sigma:
            break
        excluded.append((usable[worst].prn, 'residual_outlier'))
        usable = usable[:worst] + usable[worst + 1:]
    return PositionFix(
        position_ecef_km=position,
        clock_bias_s=clock_km / C_KM_S,
        dop=dilution_of_precision(design, position),
        used_satellites=tuple(o.prn for o in usable),
        excluded_satellites=tuple(excluded),
        residuals_km=tuple(postfit),
        residual_rms_km=rms,
        iterations=iterations,
        converged=converged)
