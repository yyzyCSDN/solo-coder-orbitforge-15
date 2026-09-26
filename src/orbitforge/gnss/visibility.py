from __future__ import annotations
from dataclasses import dataclass
import math

from orbitforge.core.vector import Vec3
from orbitforge.frames.topocentric import ecef_to_enu, az_el_range


def local_terrain_elevation(horizon_profile, azimuth_rad: float) -> float:
    """Terrain mask elevation at an azimuth.

    The profile is treated as a set of finite obstruction segments: between
    samples the elevation is linearly interpolated, and outside the covered
    azimuth span the terrain drops away (``-pi/2``). This differs from
    ``ground.horizon_profile.interpolate_profile``, which clamps to the nearest
    endpoint and would turn a narrow wall into an all-sky mask.
    """
    if not horizon_profile:
        return -math.pi / 2.0

    def _norm(a):
        m = a % (2.0 * math.pi)
        # Keep a sample placed at the 0/2pi seam on the 2pi side so an explicit
        # 0..2pi profile still reads as full-circle coverage.
        return 2.0 * math.pi if m == 0.0 and a not in (0.0, -0.0) else m

    normalized = sorted(((_norm(a), e) for a, e in horizon_profile),
                        key=lambda item: item[0])
    az = azimuth_rad % (2.0 * math.pi)
    first_a = normalized[0][0]
    last_a = normalized[-1][0]
    if first_a == 0.0 and abs(last_a - 2.0 * math.pi) < 1e-12:
        # Profile spans the full circle: interpolate directly across it
        # (including the seam, where both endpoints describe azimuth 0).
        span = last_a - first_a
        u = (az - first_a) / span if span > 0 else 0.0
        return normalized[0][1] * (1.0 - u) + normalized[-1][1] * u
    if az < first_a or az > last_a:
        return -math.pi / 2.0
    previous = normalized[0]
    for current in normalized[1:]:
        if previous[0] <= az <= current[0]:
            span = current[0] - previous[0]
            u = (az - previous[0]) / span if span > 0 else 0.0
            return previous[1] * (1.0 - u) + current[1] * u
        previous = current
    return normalized[-1][1]


@dataclass(frozen=True)
class SatelliteView:
    satellite_id: str
    constellation: str
    azimuth_rad: float
    elevation_rad: float
    range_km: float
    healthy: bool
    above_mask: bool


def view_geometry(sat_ecef_km: Vec3, receiver_lat_rad: float, receiver_lon_rad: float,
                  receiver_alt_km: float = 0.0):
    """Azimuth, elevation and slant range from a receiver to an ECEF satellite."""
    enu = ecef_to_enu(sat_ecef_km, receiver_lat_rad, receiver_lon_rad, receiver_alt_km)
    az, el, rng = az_el_range(enu)
    return az, el, rng


def compute_views(satellites, sat_positions_ecef, receiver,
                  elevation_mask_rad: float = math.radians(5.0),
                  horizon_profile=()) -> list[SatelliteView]:
    """Classify every satellite against the receiver sky mask.

    A satellite is above the mask only when its elevation clears both the
    constant cut-off angle and the local terrain/horizon profile at its azimuth.
    Blocked or unhealthy satellites remain in the list with ``above_mask``
    ``False`` so callers never mistake an unused satellite for a zero range.
    """
    views: list[SatelliteView] = []
    for sat in satellites:
        pos = sat_positions_ecef[sat.satellite_id]
        az, el, rng = view_geometry(pos, receiver.lat_rad, receiver.lon_rad, receiver.alt_km)
        terrain = local_terrain_elevation(horizon_profile, az)
        mask = max(elevation_mask_rad, terrain)
        views.append(SatelliteView(
            satellite_id=sat.satellite_id,
            constellation=sat.constellation,
            azimuth_rad=az,
            elevation_rad=el,
            range_km=rng,
            healthy=sat.healthy,
            above_mask=el > mask,
        ))
    views.sort(key=lambda v: v.satellite_id)
    return views


def visible_satellites(views: list[SatelliteView], require_healthy: bool = True):
    return [v for v in views if v.above_mask and (v.healthy or not require_healthy)]
