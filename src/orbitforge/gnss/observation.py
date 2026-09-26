from __future__ import annotations
from dataclasses import dataclass
import math
import random
from orbitforge.core.vector import Vec3
from orbitforge.core.constants import C_KM_S, OMEGA_EARTH_RAD_S
from orbitforge.frames.geodetic import ecef_to_geodetic
from orbitforge.frames.topocentric import ecef_to_enu, az_el_range
from orbitforge.frames.rotations import rz, mv
from orbitforge.navigation.ionosphere import group_delay_m, slant_tec
from orbitforge.navigation.troposphere import range_delay_km
from orbitforge.visibility.horizon import HorizonMask
from .constellation import GnssSatellite, satellite_ecef

STATUS_OK = 'ok'
STATUS_BELOW_MASK = 'below_mask'
STATUS_OCCLUDED = 'occluded'
STATUS_UNHEALTHY = 'unhealthy'

L1_FREQUENCY_HZ = 1575.42e6


@dataclass(frozen=True)
class GnssObservation:
    prn: str
    status: str
    elevation_rad: float
    azimuth_rad: float
    geometric_range_km: float
    sat_ecef_km: Vec3
    pseudorange_km: float | None = None
    sigma_km: float | None = None
    sat_clock_km: float = 0.0
    iono_km: float = 0.0
    tropo_km: float = 0.0
    noise_km: float = 0.0

    @property
    def usable(self):
        return (self.status == STATUS_OK and self.pseudorange_km is not None
                and math.isfinite(self.pseudorange_km) and self.pseudorange_km > 0.0
                and self.sigma_km is not None and self.sigma_km > 0.0)


def transmit_geometry(satellite: GnssSatellite, receive_tai_s: float, receiver_ecef: Vec3,
                      tolerance_s: float = 1e-12):
    tau = 0.07
    rotated = satellite_ecef(satellite, receive_tai_s - tau)
    range_km = 0.0
    for _ in range(8):
        rotated = mv(rz(OMEGA_EARTH_RAD_S * tau), satellite_ecef(satellite, receive_tai_s - tau))
        range_km = (rotated - receiver_ecef).norm()
        updated = range_km / C_KM_S
        if abs(updated - tau) < tolerance_s:
            tau = updated
            break
        tau = updated
    return rotated, (rotated - receiver_ecef).norm(), tau


def simulate_observations(receiver_ecef: Vec3, receiver_clock_bias_s: float,
                          satellites, epoch_tai_s: float,
                          elevation_mask_rad: float = math.radians(5.0),
                          horizon=None, sigma_ura_km: float = 0.001,
                          vertical_tec: float = 8.0, frequency_hz: float = L1_FREQUENCY_HZ,
                          clock_broadcast_sigma_s: float = 2e-9,
                          unhealthy_prns=frozenset(), seed: int = 1):
    rng = random.Random(seed)
    lat, lon, alt = ecef_to_geodetic(receiver_ecef)
    mask = HorizonMask(list(horizon)) if horizon else None
    observations = []
    for sat in satellites:
        sat_ecef, range_km, _ = transmit_geometry(sat, epoch_tai_s, receiver_ecef)
        enu = ecef_to_enu(sat_ecef, lat, lon, alt)
        az, el, _ = az_el_range(enu)
        if el < elevation_mask_rad:
            status = STATUS_BELOW_MASK
        elif mask is not None and not mask.clear(az, el):
            status = STATUS_OCCLUDED
        elif not sat.healthy or sat.prn in unhealthy_prns:
            status = STATUS_UNHEALTHY
        else:
            status = STATUS_OK
        if status != STATUS_OK:
            observations.append(GnssObservation(sat.prn, status, el, az, range_km, sat_ecef))
            continue
        sat_clock_true_s = sat.clock_offset_s(epoch_tai_s)
        broadcast_s = sat_clock_true_s + rng.gauss(0.0, clock_broadcast_sigma_s)
        iono_km = group_delay_m(slant_tec(vertical_tec, el), frequency_hz) / 1000.0
        tropo_km = range_delay_km(el)
        sigma_km = sigma_ura_km / max(math.sin(el), math.sin(elevation_mask_rad))
        noise_km = rng.gauss(0.0, sigma_km)
        pseudorange_km = (range_km + C_KM_S * (receiver_clock_bias_s - sat_clock_true_s)
                          + iono_km + tropo_km + noise_km)
        observations.append(GnssObservation(
            sat.prn, status, el, az, range_km, sat_ecef,
            pseudorange_km=pseudorange_km, sigma_km=sigma_km,
            sat_clock_km=C_KM_S * broadcast_s,
            iono_km=iono_km, tropo_km=tropo_km, noise_km=noise_km))
    return observations
