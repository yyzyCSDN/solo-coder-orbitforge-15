from __future__ import annotations
import math
import random

from orbitforge.gnss.measurement import simulate_pseudoranges
from orbitforge.gnss.receiver import solve_position
from orbitforge.gnss.models import Receiver, AtmosphericModel
from orbitforge.gnss.constellation import satellite_positions_ecef


def run_gnss_epoch(satellites, epoch_tai_s: float, receiver: Receiver,
                   receiver_clock_bias_s: float = 1.0e-6,
                   elevation_mask_rad: float = math.radians(5.0),
                   horizon_profile=(),
                   atmosphere: AtmosphericModel | None = AtmosphericModel(),
                   noise_sigma_km: float | None = None,
                   blunder_ids=(), blunder_km: float = 50.0,
                   blunder_sigma_limit: float = 4.0,
                   seed: int | None = None):
    """Simulate an epoch and immediately solve the receiver PVT.

    Returns ``(measurement_epoch, fix)`` so the raw observables (including
    blocked/unhealthy entries and their status) stay auditable alongside the
    position solution.
    """
    rng = random.Random(seed)
    positions = satellite_positions_ecef(satellites, epoch_tai_s)
    epoch = simulate_pseudoranges(
        satellites, epoch_tai_s, receiver, positions,
        receiver_clock_bias_s=receiver_clock_bias_s,
        elevation_mask_rad=elevation_mask_rad,
        horizon_profile=horizon_profile,
        atmosphere=atmosphere,
        noise_sigma_km=noise_sigma_km,
        blunder_ids=blunder_ids,
        blunder_km=blunder_km,
        rng=rng)
    fix = solve_position(epoch, satellites, positions,
                         atmosphere=atmosphere,
                         blunder_sigma_limit=blunder_sigma_limit)
    return epoch, fix
