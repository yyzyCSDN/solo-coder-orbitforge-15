from __future__ import annotations
from dataclasses import dataclass
import math
import random

from orbitforge.gnss.visibility import compute_views, visible_satellites
from orbitforge.gnss.models import Receiver, AtmosphericModel, nominal_uere_sigma_km, clock_equivalent_km

# Observation status values:
STATUS_MEASURED = 'measured'   # tracked, healthy, above mask -> pseudorange present
STATUS_BLOCKED = 'blocked'     # below elevation mask / behind terrain
STATUS_UNHEALTHY = 'unhealthy'  # satellite flagged bad, no trusted pseudorange


@dataclass(frozen=True)
class PseudorangeObservation:
    """Pseudorange to one satellite.

    ``pseudorange_km`` is ``None`` for blocked or unhealthy satellites so an
    absent measurement can never be silently turned into a zero range inside
    the solver.
    """

    satellite_id: str
    constellation: str
    azimuth_rad: float
    elevation_rad: float
    status: str
    pseudorange_km: float | None = None
    true_range_km: float = 0.0
    sigma_km: float = 0.0
    reason: str = ''


@dataclass(frozen=True)
class GnssMeasurementEpoch:
    epoch_tai_s: float
    receiver: Receiver
    observations: tuple[PseudorangeObservation, ...]
    receiver_clock_bias_s: float

    def measured(self) -> list[PseudorangeObservation]:
        return [o for o in self.observations if o.status == STATUS_MEASURED]

    def excluded(self) -> list[PseudorangeObservation]:
        return [o for o in self.observations if o.status != STATUS_MEASURED]

    def by_status(self) -> dict[str, list[PseudorangeObservation]]:
        groups: dict[str, list[PseudorangeObservation]] = {}
        for obs in self.observations:
            groups.setdefault(obs.status, []).append(obs)
        return groups


def simulate_pseudoranges(satellites, epoch_tai_s: float, receiver: Receiver,
                          sat_positions_ecef: dict,
                          receiver_clock_bias_s: float = 1.0e-6,
                          elevation_mask_rad: float = math.radians(5.0),
                          horizon_profile=(),
                          atmosphere: AtmosphericModel | None = AtmosphericModel(),
                          noise_sigma_km: float | None = None,
                          blunder_ids=(), blunder_km: float = 50.0,
                          rng: random.Random | None = None) -> GnssMeasurementEpoch:
    """Generate one epoch of GNSS pseudorange observations.

    Pseudorange model (km):
        rho = geometric range + c*(receiver clock bias - satellite clock bias)
              + iono/tropo slant delay + Gaussian noise (+ optional blunder)

    Blocked and unhealthy satellites are emitted with status tags and a
    ``None`` pseudorange. ``blunder_ids`` marks tracked satellites that carry a
    large unmodelled range error (a "bad" satellite that is still transmitting);
    the receiver must detect and reject those from the fix.
    """
    rng = rng if rng is not None else random.Random()
    blunders = set(blunder_ids)
    receiver_ecef = receiver.position_ecef_km
    views = compute_views(satellites, sat_positions_ecef, receiver,
                          elevation_mask_rad, horizon_profile)
    tracked = {v.satellite_id: v for v in visible_satellites(views)}

    observations: list[PseudorangeObservation] = []
    for view in views:
        sat = next(s for s in satellites if s.satellite_id == view.satellite_id)
        if not view.above_mask:
            status, reason = STATUS_BLOCKED, 'below sky mask'
        elif not sat.healthy:
            status, reason = STATUS_UNHEALTHY, 'satellite flagged unhealthy'
        else:
            status, reason = STATUS_MEASURED, ''

        if status != STATUS_MEASURED:
            observations.append(PseudorangeObservation(
                view.satellite_id, view.constellation, view.azimuth_rad,
                view.elevation_rad, status, None, view.range_km, 0.0, reason))
            continue

        sigma = noise_sigma_km if noise_sigma_km is not None else nominal_uere_sigma_km(view.elevation_rad)
        delay = atmosphere.slant_delay_km(view.elevation_rad) if atmosphere is not None else 0.0
        sat_clock_km = clock_equivalent_km(sat.clock_bias_s)
        rx_clock_km = clock_equivalent_km(receiver_clock_bias_s)
        noise_km = rng.gauss(0.0, sigma)
        blunder_delta = blunder_km if view.satellite_id in blunders else 0.0
        pseudorange = view.range_km + rx_clock_km - sat_clock_km + delay + noise_km + blunder_delta
        observations.append(PseudorangeObservation(
            view.satellite_id, view.constellation, view.azimuth_rad,
            view.elevation_rad, status, pseudorange, view.range_km, sigma))

    return GnssMeasurementEpoch(epoch_tai_s, receiver, tuple(observations), receiver_clock_bias_s)
