import math
import random

import pytest

from orbitforge.gnss import (
    Receiver, AtmosphericModel, gps_constellation, galileo_constellation,
    combine_constellations, satellite_positions_ecef, compute_views,
    visible_satellites, geometry_matrix, dilution_of_precision,
    simulate_pseudoranges, solve_position, run_gnss_epoch,
    STATUS_MEASURED, STATUS_BLOCKED, STATUS_UNHEALTHY,
)
from orbitforge.gnss.visibility import local_terrain_elevation
from orbitforge.core.errors import NoSolutionError, ConvergenceError

EPOCH = 1_700_000_000.0
LAT, LON = math.radians(39.9), math.radians(116.4)


def _receiver():
    return Receiver(LAT, LON, 0.05)


def _setup():
    sats = combine_constellations(gps_constellation(), galileo_constellation())
    positions = satellite_positions_ecef(sats, EPOCH)
    return sats, positions


def test_walker_constellation_count_and_orbit():
    gps = gps_constellation()
    gal = galileo_constellation()
    assert len(gps) == 24 and len(gal) == 24
    # No ID collisions when constellations are combined (Galileo uses E-prefix).
    combined = combine_constellations(gps, gal)
    ids = {s.satellite_id for s in combined}
    assert len(ids) == 48
    assert all(s.constellation in ('GPS', 'GAL') for s in combined)
    for sat in combined:
        pos = sat.position_ecef_km(EPOCH)
        # MEO orbital radius within 1 km of the prescribed semimajor axis.
        assert abs(pos.norm() - sat.elements.a_km) < 1.0


def test_visibility_counts_and_ordering():
    sats, positions = _setup()
    receiver = _receiver()
    views = compute_views(sats, positions, receiver)
    assert len(views) == 48
    visible = visible_satellites(views)
    assert 8 <= len(visible) <= 30
    assert all(v.above_mask and v.elevation_rad > math.radians(5.0) for v in visible)
    ids = [v.satellite_id for v in views]
    assert ids == sorted(ids)


def test_elevation_mask_reduces_visibility():
    sats, positions = _setup()
    low = compute_views(sats, positions, _receiver(), math.radians(5.0))
    high = compute_views(sats, positions, _receiver(), math.radians(40.0))
    assert len(visible_satellites(high)) < len(visible_satellites(low))


def test_terrain_profile_segment_semantics():
    az = math.radians(120.0)
    wall = [(az - 0.02, math.radians(30.0)), (az + 0.02, math.radians(30.0))]
    assert local_terrain_elevation(wall, az) == pytest.approx(math.radians(30.0))
    # A narrow wall must not clamp the rest of the sky to its elevation.
    assert local_terrain_elevation(wall, az + math.pi) < 0.0
    # An explicit 0..2pi wall covers the full circle.
    full = [(0.0, math.radians(80.0)), (2.0 * math.pi, math.radians(80.0))]
    assert local_terrain_elevation(full, math.pi) == pytest.approx(math.radians(80.0))
    assert local_terrain_elevation((), 1.2) == -math.pi / 2.0


def test_pseudorange_model_recovers_clock_and_range():
    sats, positions = _setup()
    receiver = _receiver()
    no_noise = AtmosphericModel(0.0, 0.0)
    epoch = simulate_pseudoranges(
        sats, EPOCH, receiver, positions, receiver_clock_bias_s=1.5e-6,
        atmosphere=no_noise, noise_sigma_km=0.0, rng=random.Random(0))
    sat_by_id = {s.satellite_id: s for s in sats}
    for obs in epoch.measured():
        sat = sat_by_id[obs.satellite_id]
        expected = obs.true_range_km + 299792.458 * (1.5e-6 - sat.clock_bias_s)
        assert obs.pseudorange_km == pytest.approx(expected, abs=1e-9)


def test_blocked_and_unhealthy_are_excluded_not_zero():
    sats, positions = _setup()
    receiver = _receiver()
    views = compute_views(sats, positions, receiver)
    target = visible_satellites(views)[0]
    for sat in sats:
        if sat.satellite_id == target.satellite_id:
            object.__setattr__(sat, 'healthy', False)
    epoch = simulate_pseudoranges(sats, EPOCH, receiver, positions,
                                  elevation_mask_rad=math.radians(89.9),
                                  rng=random.Random(1))
    # Near-zenith mask leaves no measured satellites.
    assert len(epoch.measured()) == 0
    assert epoch.excluded()
    for obs in epoch.observations:
        if obs.elevation_rad > 0:
            assert obs.status in (STATUS_BLOCKED, STATUS_UNHEALTHY)
            assert obs.pseudorange_km is None
    # Explicitly unhealthy tagging survives even with an open sky is checked
    # end-to-end in test_fix_reports_usage.


def test_solve_position_accuracy_and_clock():
    sats = gps_constellation()
    positions = satellite_positions_ecef(sats, EPOCH)
    receiver = _receiver()
    truth = receiver.position_ecef_km
    errors = []
    clock_errors = []
    for seed in range(8):
        _, fix = run_gnss_epoch(sats, EPOCH, receiver,
                                receiver_clock_bias_s=-1.2e-6, seed=seed)
        errors.append((fix.position_ecef_km - truth).norm())
        clock_errors.append(abs(fix.clock_bias_s - (-1.2e-6)))
    # A few metres of position error and tens of nanoseconds of clock error.
    assert sum(errors) / len(errors) < 0.02
    assert max(clock_errors) < 1.0e-7
    assert abs(fix.lat_rad - LAT) < 1e-5
    assert abs(fix.lon_rad - LON) < 1e-5
    assert abs(fix.alt_km - 0.05) < 0.05


def test_dilution_of_precision_ordering_and_positivity():
    sats = gps_constellation()
    positions = satellite_positions_ecef(sats, EPOCH)
    receiver = _receiver()
    views = visible_satellites(compute_views(sats, positions, receiver))
    design = geometry_matrix(views, positions, receiver.position_ecef_km)
    dop = dilution_of_precision(design, LAT, LON)
    assert dop['GDOP'] >= dop['PDOP'] >= dop['HDOP'] > 0
    assert dop['PDOP'] >= dop['VDOP']
    assert dop['GDOP'] >= dop['TDOP'] > 0


def test_fix_reports_which_satellites_were_used():
    sats, positions = _setup()
    receiver = _receiver()
    views = compute_views(sats, positions, receiver)
    visible = visible_satellites(views)
    unhealthy_id = visible[0].satellite_id
    blunder_id = visible[1].satellite_id
    for sat in sats:
        if sat.satellite_id == unhealthy_id:
            object.__setattr__(sat, 'healthy', False)
    epoch = simulate_pseudoranges(
        sats, EPOCH, receiver, positions, receiver_clock_bias_s=2.0e-6,
        blunder_ids=[blunder_id], blunder_km=40.0, rng=random.Random(11))
    fix = solve_position(epoch, sats, positions, blunder_sigma_limit=4.0)

    assert len(fix.used_satellites) >= 4
    assert blunder_id in fix.rejected_satellite_ids
    assert blunder_id not in fix.used_satellite_ids
    excluded_ids = [sid for sid, _ in fix.excluded]
    assert unhealthy_id in excluded_ids
    # No blocked/unhealthy satellite carries a zero range into the solution.
    assert all(sid not in fix.used_satellite_ids for sid, _ in fix.excluded)
    summary = fix.describe_usage()
    assert 'used' in summary and blunder_id in summary and unhealthy_id in summary


def test_blunder_without_rejection_biases_but_is_dropped_with_redundancy():
    # Blunder detection needs redundancy: use the combined GPS+Galileo sky.
    sats, positions = _setup()
    receiver = _receiver()
    truth = receiver.position_ecef_km
    views = visible_satellites(compute_views(sats, positions, receiver))
    blunder_id = views[0].satellite_id
    epoch = simulate_pseudoranges(
        sats, EPOCH, receiver, positions, blunder_ids=[blunder_id], blunder_km=60.0,
        rng=random.Random(5))
    # A huge limit effectively disables rejection: the blunder enters the fix.
    permissive = solve_position(epoch, sats, positions, blunder_sigma_limit=1.0e9)
    strict = solve_position(epoch, sats, positions, blunder_sigma_limit=4.0)
    assert blunder_id in permissive.used_satellite_ids
    assert blunder_id in strict.rejected_satellite_ids
    assert (strict.position_ecef_km - truth).norm() < (
        permissive.position_ecef_km - truth).norm()
    # The rejected satellite is named, never silently dropped.
    assert all(s.satellite_id != blunder_id for s in strict.used_satellites)


def test_fewer_than_four_satellites_raises():
    sats = gps_constellation()
    positions = satellite_positions_ecef(sats, EPOCH)
    epoch = simulate_pseudoranges(
        sats, EPOCH, _receiver(), positions,
        elevation_mask_rad=math.radians(89.9), rng=random.Random(2))
    assert len(epoch.measured()) < 4
    with pytest.raises(NoSolutionError):
        solve_position(epoch, sats, positions)


def test_deterministic_seeded_simulation():
    sats, positions = _setup()
    receiver = _receiver()
    e1 = simulate_pseudoranges(sats, EPOCH, receiver, positions, rng=random.Random(42))
    e2 = simulate_pseudoranges(sats, EPOCH, receiver, positions, rng=random.Random(42))
    r1 = [(o.satellite_id, o.pseudorange_km) for o in e1.measured()]
    r2 = [(o.satellite_id, o.pseudorange_km) for o in e2.measured()]
    assert r1 == r2
