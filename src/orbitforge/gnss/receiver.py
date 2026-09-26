from __future__ import annotations
from dataclasses import dataclass
import math

from orbitforge.core.vector import Vec3
from orbitforge.core.constants import C_KM_S
from orbitforge.core.errors import NoSolutionError, ConvergenceError
from orbitforge.frames.geodetic import ecef_to_geodetic
from orbitforge.navigation.least_squares import solve_linear
from orbitforge.gnss.geometry import dilution_of_precision, _enu_rotation
from orbitforge.gnss.models import AtmosphericModel
from orbitforge.gnss.measurement import GnssMeasurementEpoch

# Looser than the floating-point floor at MEO-Earth scales (~1e-11 km roundoff
# in the weighted normal equations) but far below the measurement noise floor:
# 1 cm of position, 1e-9 km of equivalent range (~3 ps of clock).
_POS_TOL_KM = 1.0e-8
_TIME_TOL_KM = 1.0e-9
_MAX_ITERATIONS = 25


@dataclass(frozen=True)
class UsedSatellite:
    satellite_id: str
    constellation: str
    azimuth_rad: float
    elevation_rad: float
    sigma_km: float
    postfit_residual_km: float


@dataclass(frozen=True)
class PositionFix:
    epoch_tai_s: float
    position_ecef_km: Vec3
    lat_rad: float
    lon_rad: float
    alt_km: float
    clock_bias_s: float
    used_satellites: tuple[UsedSatellite, ...]
    rejected_satellite_ids: tuple[str, ...]
    excluded: tuple[tuple[str, str], ...]
    dop: dict
    sigma_east_km: float
    sigma_north_km: float
    sigma_up_km: float
    sigma_clock_s: float
    rms_residual_km: float
    iterations: int

    @property
    def used_satellite_ids(self) -> tuple[str, ...]:
        return tuple(s.satellite_id for s in self.used_satellites)

    def describe_usage(self) -> str:
        """Human-readable summary of which satellites the fix used and why."""
        lines = [f'used {len(self.used_satellites)} satellites: '
                 + ', '.join(f'{s.constellation}:{s.satellite_id}' for s in self.used_satellites)]
        if self.rejected_satellite_ids:
            lines.append('rejected as blunders: ' + ', '.join(self.rejected_satellite_ids))
        for sat_id, reason in self.excluded:
            lines.append(f'excluded {sat_id}: {reason}')
        d = self.dop
        lines.append(f'GDOP={d["GDOP"]:.2f} PDOP={d["PDOP"]:.2f} '
                     f'HDOP={d["HDOP"]:.2f} VDOP={d["VDOP"]:.2f} TDOP={d["TDOP"]:.2f}')
        return '\n'.join(lines)


def _corrected_ranges(epoch: GnssMeasurementEpoch, satellites, atmosphere):
    sat_by_id = {sat.satellite_id: sat for sat in satellites}
    corrected = {}
    for obs in epoch.measured():
        sat = sat_by_id[obs.satellite_id]
        sat_clock_km = sat.clock_bias_s * C_KM_S
        delay = atmosphere.slant_delay_km(obs.elevation_rad) if atmosphere is not None else 0.0
        corrected[obs.satellite_id] = obs.pseudorange_km + sat_clock_km - delay
    return corrected


def _iterate(active, obs_by_id, positions, corrected, x, b_km):
    """One Gauss-Newton weighted least-squares run over ``active`` ids."""
    for iteration in range(1, _MAX_ITERATIONS + 1):
        normal = [[0.0] * 4 for _ in range(4)]
        rhs = [0.0] * 4
        residuals = {}
        design = []
        for sat_id in active:
            obs = obs_by_id[sat_id]
            sat_pos = positions[sat_id]
            delta = sat_pos - x
            rng = delta.norm()
            ux, uy, uz = delta.x / rng, delta.y / rng, delta.z / rng
            row = [-ux, -uy, -uz, 1.0]
            design.append(row)
            residual = corrected[sat_id] - (rng + b_km)
            residuals[sat_id] = residual
            w = 1.0 / (obs.sigma_km * obs.sigma_km)
            for i in range(4):
                rhs[i] += w * row[i] * residual
                for j in range(4):
                    normal[i][j] += w * row[i] * row[j]
        step = solve_linear(normal, rhs)
        x = Vec3(x.x + step[0], x.y + step[1], x.z + step[2])
        b_km += step[3]
        if math.hypot(step[0], step[1], step[2]) < _POS_TOL_KM and abs(step[3]) < _TIME_TOL_KM:
            break
    else:
        raise ConvergenceError('pseudorange least squares did not converge')

    # Final pass at the converged state for residuals/design rows.
    final_residuals = {}
    final_design = []
    for sat_id in active:
        sat_pos = positions[sat_id]
        delta = sat_pos - x
        rng = delta.norm()
        ux, uy, uz = delta.x / rng, delta.y / rng, delta.z / rng
        final_design.append([-ux, -uy, -uz, 1.0])
        final_residuals[sat_id] = corrected[sat_id] - (rng + b_km)
    return x, b_km, final_residuals, final_design, iteration


def solve_position(epoch: GnssMeasurementEpoch, satellites, sat_positions_ecef: dict,
                   atmosphere: AtmosphericModel | None = AtmosphericModel(),
                   blunder_sigma_limit: float = 4.0,
                   min_satellites: int = 4) -> PositionFix:
    """Solve receiver position and clock bias from noisy pseudoranges.

    Only observations tagged ``measured`` enter the solution; blocked and
    unhealthy satellites (whose pseudorange is ``None``) are reported as
    excluded rather than forced to zero. Tracked satellites whose postfit
    residual exceeds ``blunder_sigma_limit`` sigmas are iteratively dropped as
    blunders while at least ``min_satellites`` remain.

    Note that postfit blunder detection needs redundancy: with barely four
    satellites the residuals are identically zero, and even six or seven can
    absorb a large gross error into the solution. Multi-constellation geometry
    (10+ satellites) is where rejection becomes reliable.
    """
    measured = epoch.measured()
    if len(measured) < min_satellites:
        raise NoSolutionError(
            f'{len(measured)} measured satellites, {min_satellites} required')

    obs_by_id = {obs.satellite_id: obs for obs in measured}
    corrected = _corrected_ranges(epoch, satellites, atmosphere)
    active = [obs.satellite_id for obs in measured]

    # Start from Earth's centre with zero clock offset; the geometry is benign.
    x = Vec3(0.0, 0.0, 0.0)
    b_km = 0.0
    rejected: list[str] = []

    x, b_km, residuals, design, iterations = _iterate(
        active, obs_by_id, sat_positions_ecef, corrected, x, b_km)

    # Sequential blunder detection on the weighted postfit residuals.
    while len(active) > min_satellites:
        worst = max(active, key=lambda sid: abs(residuals[sid]) / obs_by_id[sid].sigma_km)
        if abs(residuals[worst]) / obs_by_id[worst].sigma_km <= blunder_sigma_limit:
            break
        active.remove(worst)
        rejected.append(worst)
        del corrected[worst]
        x, b_km, residuals, design, iters = _iterate(
            active, obs_by_id, sat_positions_ecef, corrected, x, b_km)
        iterations += iters

    if len(active) < min_satellites:
        raise NoSolutionError('blunder rejection removed too many satellites')

    lat, lon, alt = ecef_to_geodetic(x)
    dop = dilution_of_precision(design, lat, lon)

    # Formal covariance from the weighted normal equations, a posteriori scale.
    normal = [[0.0] * 4 for _ in range(4)]
    for row, sid in zip(design, active):
        w = 1.0 / (obs_by_id[sid].sigma_km ** 2)
        for i in range(4):
            for j in range(4):
                normal[i][j] += w * row[i] * row[j]
    dof = len(active) - 4
    chi2 = sum((residuals[sid] / obs_by_id[sid].sigma_km) ** 2 for sid in active)
    scale = chi2 / dof if dof > 0 else 1.0
    weighted_q = _invert4(normal)
    cov = [[scale * weighted_q[i][j] for j in range(4)] for i in range(4)]

    r_enu = _enu_rotation(lat, lon)
    # Rotate the position covariance block into East-North-Up.
    q_enu_block = [[sum(r_enu[i][a] * cov[a][b] * r_enu[j][b]
                        for a in range(3) for b in range(3))
                    for j in range(3)] for i in range(3)]
    sigma_east = math.sqrt(max(q_enu_block[0][0], 0.0))
    sigma_north = math.sqrt(max(q_enu_block[1][1], 0.0))
    sigma_up = math.sqrt(max(q_enu_block[2][2], 0.0))
    sigma_clock_s = math.sqrt(max(cov[3][3], 0.0)) / C_KM_S

    rms = math.sqrt(sum(residuals[sid] ** 2 for sid in active) / len(active))

    used = tuple(
        UsedSatellite(
            sat_id, obs_by_id[sat_id].constellation, obs_by_id[sat_id].azimuth_rad,
            obs_by_id[sat_id].elevation_rad, obs_by_id[sat_id].sigma_km,
            residuals[sat_id])
        for sat_id in active)
    excluded = tuple((obs.satellite_id, obs.reason) for obs in epoch.excluded())

    return PositionFix(
        epoch_tai_s=epoch.epoch_tai_s,
        position_ecef_km=x,
        lat_rad=lat,
        lon_rad=lon,
        alt_km=alt,
        clock_bias_s=b_km / C_KM_S,
        used_satellites=used,
        rejected_satellite_ids=tuple(rejected),
        excluded=excluded,
        dop=dop,
        sigma_east_km=sigma_east,
        sigma_north_km=sigma_north,
        sigma_up_km=sigma_up,
        sigma_clock_s=sigma_clock_s,
        rms_residual_km=rms,
        iterations=iterations,
    )


def _invert4(a):
    return [solve_linear(a, [1.0 if i == j else 0.0 for j in range(4)]) for i in range(4)]
