from __future__ import annotations
from dataclasses import dataclass
import math
import random
from orbitforge.core.vector import Vec3
from orbitforge.core.constants import R_EARTH_EQUATOR_KM
from orbitforge.core.state import CartesianState
from orbitforge.navigation.clock import ClockModel
from orbitforge.orbits.elements import KeplerianElements, elements_to_state
from orbitforge.orbits.two_body import propagate_two_body
from orbitforge.frames.eci_ecef import eci_to_ecef

GPS_ALTITUDE_KM = 20180.0
GPS_INCLINATION_RAD = math.radians(55.0)


@dataclass(frozen=True)
class GnssSatellite:
    prn: str
    elements: KeplerianElements
    elements_epoch_tai_s: float
    clock: ClockModel
    healthy: bool = True

    def clock_offset_s(self, epoch_tai_s: float) -> float:
        return self.clock.offset(epoch_tai_s - self.elements_epoch_tai_s)


def walker_delta(prefix: str, satellite_count: int, plane_count: int, altitude_km: float,
                 inclination_rad: float, phasing: int = 1, raan0_rad: float = 0.0,
                 epoch_tai_s: float = 0.0, clock_sigma_s: float = 3e-7, seed: int = 2024):
    if satellite_count % plane_count:
        raise ValueError('satellite count must be a multiple of plane count')
    rng = random.Random(seed)
    a_km = R_EARTH_EQUATOR_KM + altitude_km
    per_plane = satellite_count // plane_count
    satellites = []
    for plane in range(plane_count):
        raan = raan0_rad + plane * 2.0 * math.pi / plane_count
        for slot in range(per_plane):
            mean_anomaly = (slot * 2.0 * math.pi / per_plane
                            + plane * phasing * 2.0 * math.pi / satellite_count)
            elements = KeplerianElements(a_km, 0.0, inclination_rad, raan, 0.0,
                                         mean_anomaly % (2.0 * math.pi))
            clock = ClockModel(bias_s=rng.gauss(0.0, clock_sigma_s),
                               drift_s_per_s=rng.gauss(0.0, 1e-11))
            prn = '%s%02d' % (prefix, plane * per_plane + slot + 1)
            satellites.append(GnssSatellite(prn, elements, epoch_tai_s, clock))
    return satellites


def gps_constellation(epoch_tai_s: float = 0.0, seed: int = 2024):
    return walker_delta('G', 24, 6, GPS_ALTITUDE_KM, GPS_INCLINATION_RAD,
                        phasing=1, epoch_tai_s=epoch_tai_s, seed=seed)


def satellite_ecef(satellite: GnssSatellite, epoch_tai_s: float) -> Vec3:
    r0, v0 = elements_to_state(satellite.elements)
    state = propagate_two_body(
        CartesianState(satellite.elements_epoch_tai_s, r0, v0),
        epoch_tai_s - satellite.elements_epoch_tai_s)
    return eci_to_ecef(state.position_km, epoch_tai_s)
