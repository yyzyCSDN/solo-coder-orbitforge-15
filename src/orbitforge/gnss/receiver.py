from __future__ import annotations
from dataclasses import dataclass
import math
from orbitforge.core.vector import Vec3
from orbitforge.frames.geodetic import geodetic_to_ecef, ecef_to_geodetic
from .observation import simulate_observations, STATUS_OK, STATUS_UNHEALTHY
from .positioning import PositionFix, solve_position


@dataclass(frozen=True)
class ReceiverSite:
    lat_rad: float
    lon_rad: float
    alt_km: float = 0.0
    clock_bias_s: float = 0.0
    elevation_mask_rad: float = math.radians(5.0)
    horizon: tuple = ()

    @property
    def ecef_km(self) -> Vec3:
        return geodetic_to_ecef(self.lat_rad, self.lon_rad, self.alt_km)


@dataclass(frozen=True)
class GnssEpochResult:
    epoch_tai_s: float
    site: ReceiverSite
    observations: tuple
    visible_satellites: tuple
    fix: PositionFix
    position_error_km: float
    clock_error_s: float

    def summary(self):
        lat, lon, alt = ecef_to_geodetic(self.fix.position_ecef_km)
        return {
            'epoch_tai_s': self.epoch_tai_s,
            'used_satellites': list(self.fix.used_satellites),
            'excluded_satellites': [{'prn': p, 'reason': r}
                                    for p, r in self.fix.excluded_satellites],
            'visible_satellites': list(self.visible_satellites),
            'position_ecef_km': list(self.fix.position_ecef_km.as_tuple()),
            'position_geodetic': {'lat_rad': lat, 'lon_rad': lon, 'alt_km': alt},
            'clock_bias_s': self.fix.clock_bias_s,
            'dop': {'gdop': self.fix.dop.gdop, 'pdop': self.fix.dop.pdop,
                    'hdop': self.fix.dop.hdop, 'vdop': self.fix.dop.vdop,
                    'tdop': self.fix.dop.tdop},
            'residual_rms_km': self.fix.residual_rms_km,
            'converged': self.fix.converged,
            'position_error_km': self.position_error_km,
            'clock_error_s': self.clock_error_s,
        }


def measure_and_fix(site: ReceiverSite, satellites, epoch_tai_s: float, seed: int = 1,
                    sigma_ura_km: float = 0.001, vertical_tec: float = 8.0,
                    unhealthy_prns=frozenset(), screen_sigma: float | None = 10.0,
                    initial_ecef_km: Vec3 | None = None) -> GnssEpochResult:
    observations = simulate_observations(
        site.ecef_km, site.clock_bias_s, satellites, epoch_tai_s,
        elevation_mask_rad=site.elevation_mask_rad, horizon=site.horizon,
        sigma_ura_km=sigma_ura_km, vertical_tec=vertical_tec,
        unhealthy_prns=unhealthy_prns, seed=seed)
    fix = solve_position(observations, initial_ecef_km=initial_ecef_km,
                         screen_sigma=screen_sigma)
    visible = tuple(o.prn for o in observations
                    if o.status in (STATUS_OK, STATUS_UNHEALTHY))
    return GnssEpochResult(
        epoch_tai_s=epoch_tai_s,
        site=site,
        observations=tuple(observations),
        visible_satellites=visible,
        fix=fix,
        position_error_km=(fix.position_ecef_km - site.ecef_km).norm(),
        clock_error_s=fix.clock_bias_s - site.clock_bias_s)
