from __future__ import annotations
from dataclasses import dataclass
import math

from orbitforge.core.vector import Vec3
from orbitforge.core.constants import C_KM_S
from orbitforge.frames.geodetic import geodetic_to_ecef


@dataclass(frozen=True)
class Receiver:
    """Ground receiver/antenna position in WGS-84 geodetic coordinates."""

    lat_rad: float
    lon_rad: float
    alt_km: float = 0.0

    @property
    def position_ecef_km(self) -> Vec3:
        return geodetic_to_ecef(self.lat_rad, self.lon_rad, self.alt_km)


@dataclass(frozen=True)
class AtmosphericModel:
    """Nominal zenith delays applied with a 1/sin(elevation) mapping function."""

    iono_zenith_km: float = 0.007   # ~7 m vertical at L-band (order of magnitude)
    tropo_zenith_km: float = 0.0024  # ~2.4 m dry+wet zenith

    def slant_delay_km(self, elevation_rad: float) -> float:
        sin_el = max(math.sin(elevation_rad), 0.05)
        return (self.iono_zenith_km + self.tropo_zenith_km) / sin_el


def nominal_uere_sigma_km(elevation_rad: float,
                          sigma_nominal_km: float = 0.002,
                          sigma_horizon_km: float = 0.012) -> float:
    """Elevation-dependent user-equivalent range error (larger near the horizon)."""
    sin_el = max(math.sin(elevation_rad), 0.02)
    return sigma_nominal_km + (sigma_horizon_km - sigma_nominal_km) / (1.0 + 10.0 * sin_el)


def clock_equivalent_km(bias_s: float) -> float:
    return bias_s * C_KM_S
