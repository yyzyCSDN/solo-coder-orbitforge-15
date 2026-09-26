from __future__ import annotations
from dataclasses import dataclass
import math
import random

from orbitforge.core.vector import Vec3
from orbitforge.core.constants import MU_EARTH_KM3_S2
from orbitforge.orbits.elements import KeplerianElements, elements_to_state
from orbitforge.frames.eci_ecef import eci_to_ecef

# Standard medium-Earth-orbit constellation presets (period follows from mu/a^3).
GPS_A_KM = 26559.8      # ~12 h sidereal repeating ground track
GALILEO_A_KM = 29599.8  # ~14 h repeating ground track
GPS_I_RAD = math.radians(55.0)
GALILEO_I_RAD = math.radians(56.0)


@dataclass(frozen=True)
class GnssSatellite:
    """A navigation satellite on a circular two-body orbit defined at t=0 (TAI s)."""

    satellite_id: str
    constellation: str
    elements: KeplerianElements
    clock_bias_s: float = 0.0
    healthy: bool = True

    def position_ecef_km(self, epoch_tai_s: float, mu: float = MU_EARTH_KM3_S2) -> Vec3:
        """Propagate the two-body orbit to the epoch and rotate into ECEF."""
        r_eci, _ = self._state_eci(epoch_tai_s, mu)
        return eci_to_ecef(r_eci, epoch_tai_s)

    def _state_eci(self, epoch_tai_s: float, mu: float):
        el0 = self.elements
        n = math.sqrt(mu / el0.a_km ** 3)
        # Circular orbit: mean and true anomaly coincide and advance linearly.
        nu = (el0.nu_rad + n * epoch_tai_s) % (2.0 * math.pi)
        el = KeplerianElements(el0.a_km, el0.e, el0.i_rad, el0.raan_rad, el0.argp_rad, nu)
        return elements_to_state(el, mu)


def _walker_elements(a_km: float, i_rad: float, planes: int, sats_per_plane: int,
                     walker_phase: float, plane: int, slot: int) -> KeplerianElements:
    total = planes * sats_per_plane
    raan = 2.0 * math.pi * plane / planes
    # Walker phasing: adjacent planes are offset by F slots around the orbit.
    nu = 2.0 * math.pi * slot / sats_per_plane + 2.0 * math.pi * walker_phase * plane / total
    return KeplerianElements(a_km, 0.0, i_rad, raan, 0.0, nu % (2.0 * math.pi))


def build_walker_constellation(name: str, a_km: float, i_rad: float, planes: int,
                               sats_per_plane: int, walker_phase: float = 1.0,
                               id_prefix: str | None = None,
                               clock_bias_spread_s: float = 1.0e-5,
                               unhealthy_ids=(), seed: int | None = None) -> list[GnssSatellite]:
    """Build a Walker delta constellation with small randomized satellite clock biases."""
    rng = random.Random(seed)
    prefix = id_prefix if id_prefix is not None else name[0].upper()
    sats: list[GnssSatellite] = []
    index = 1
    unhealthy = set(unhealthy_ids)
    for plane in range(planes):
        for slot in range(sats_per_plane):
            sat_id = f'{prefix}{index:02d}'
            elements = _walker_elements(a_km, i_rad, planes, sats_per_plane,
                                        walker_phase, plane, slot)
            bias = rng.uniform(-clock_bias_spread_s, clock_bias_spread_s)
            sats.append(GnssSatellite(sat_id, name, elements, bias, sat_id not in unhealthy))
            index += 1
    return sats


def gps_constellation(seed: int = 20260926) -> list[GnssSatellite]:
    # Nominal GPS: 24 slots, 6 planes x 4.
    return build_walker_constellation('GPS', GPS_A_KM, GPS_I_RAD, 6, 4,
                                      walker_phase=1.0, seed=seed)


def galileo_constellation(seed: int = 20260927) -> list[GnssSatellite]:
    # Nominal Galileo Walker 24/3/1; E-prefix keeps IDs distinct from GPS.
    return build_walker_constellation('GAL', GALILEO_A_KM, GALILEO_I_RAD, 3, 8,
                                      walker_phase=1.0, id_prefix='E', seed=seed)


def combine_constellations(*constellations) -> list[GnssSatellite]:
    satellites: list[GnssSatellite] = []
    seen = set()
    for constellation in constellations:
        for sat in constellation:
            key = (sat.constellation, sat.satellite_id)
            if key in seen:
                raise ValueError(f'duplicate satellite {sat.constellation}:{sat.satellite_id}')
            seen.add(key)
            satellites.append(sat)
    return satellites


def satellite_positions_ecef(satellites, epoch_tai_s: float) -> dict[str, Vec3]:
    return {sat.satellite_id: sat.position_ecef_km(epoch_tai_s) for sat in satellites}
