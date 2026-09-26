import math
from dataclasses import replace
from orbitforge.core.vector import Vec3
from orbitforge.core.constants import C_KM_S
from orbitforge.core.errors import NoSolutionError
from orbitforge.gnss.constellation import gps_constellation, walker_delta, satellite_ecef
from orbitforge.gnss.observation import simulate_observations, GnssObservation
from orbitforge.gnss.positioning import solve_position
from orbitforge.gnss.receiver import ReceiverSite, measure_and_fix

SITE = ReceiverSite(math.radians(35.0), math.radians(139.0), 0.05, clock_bias_s=1.5e-4)


def synthetic_fix(extra=(), clock_km=10.0):
    rx = Vec3(6378.137, 0.0, 0.0)
    sat_pos = [Vec3(20200, 0, 0), Vec3(0, 20200, 0), Vec3(0, 0, 20200),
               Vec3(-12000, 0, 16000), Vec3(0, -20200, 0)]
    obs = []
    for i, sp in enumerate(sat_pos):
        rng = (sp - rx).norm()
        obs.append(GnssObservation('S%02d' % i, 'ok', 0.5, 0.0, rng, sp,
                                   pseudorange_km=rng + clock_km, sigma_km=0.002))
    return rx, clock_km, solve_position(obs + list(extra))


def test_constellation_size_and_clocks():
    sats = gps_constellation()
    assert len(sats) == 24 and len({s.prn for s in sats}) == 24
    assert all(s.healthy for s in sats)
    assert len({s.clock.bias_s for s in sats}) > 1
    w = walker_delta('B', 6, 3, 21500.0, math.radians(55.0))
    assert len(w) == 6 and w[0].prn == 'B01'


def test_satellite_ecef_altitude():
    sat = gps_constellation()[0]
    r = satellite_ecef(sat, 0.0).norm()
    assert abs(r - (6378.137 + 20180.0)) < 1.0


def test_observations_carry_pseudorange_and_components():
    obs = simulate_observations(SITE.ecef_km, SITE.clock_bias_s, gps_constellation(), 0.0, seed=7)
    ok = [o for o in obs if o.status == 'ok']
    below = [o for o in obs if o.status == 'below_mask']
    assert len(ok) >= 4 and len(below) > 0
    for o in ok:
        assert 18000.0 < o.pseudorange_km < 28000.0
        assert o.iono_km > 0.0 and o.tropo_km > 0.0 and o.sigma_km > 0.0
        assert abs(o.pseudorange_km - o.geometric_range_km
                   - C_KM_S * SITE.clock_bias_s + o.sat_clock_km) < 1.0
    for o in below:
        assert o.pseudorange_km is None and o.elevation_rad < SITE.elevation_mask_rad


def test_fix_recovers_position_and_clock():
    result = measure_and_fix(SITE, gps_constellation(), 0.0, seed=7)
    assert result.fix.converged
    assert result.position_error_km < 0.05
    assert abs(result.clock_error_s) < 5e-7
    assert len(result.fix.used_satellites) >= 4
    assert set(result.fix.used_satellites) <= set(result.visible_satellites)


def test_dop_ordering_and_scale():
    result = measure_and_fix(SITE, gps_constellation(), 0.0, seed=7)
    d = result.fix.dop
    assert 0.0 < d.hdop <= d.pdop <= d.gdop
    assert 0.0 < d.vdop <= d.pdop and 0.0 < d.tdop <= d.gdop
    assert d.gdop < 50.0


def test_unhealthy_satellites_excluded():
    result = measure_and_fix(SITE, gps_constellation(), 0.0, seed=7,
                             unhealthy_prns=frozenset({'G01', 'G05'}))
    assert 'G01' not in result.fix.used_satellites
    assert 'G05' not in result.fix.used_satellites
    reasons = dict(result.fix.excluded_satellites)
    assert reasons['G01'] == 'unhealthy' and reasons['G05'] == 'unhealthy'
    assert result.position_error_km < 0.05


def test_occluded_satellite_excluded():
    obs = simulate_observations(SITE.ecef_km, SITE.clock_bias_s, gps_constellation(), 0.0, seed=7)
    target = next(o for o in obs if o.status == 'ok')
    bump = ((target.azimuth_rad - 0.3, 0.0),
            (target.azimuth_rad, target.elevation_rad + 0.05),
            (target.azimuth_rad + 0.3, 0.0))
    site = ReceiverSite(SITE.lat_rad, SITE.lon_rad, SITE.alt_km, SITE.clock_bias_s, horizon=bump)
    result = measure_and_fix(site, gps_constellation(), 0.0, seed=7)
    assert target.prn not in result.fix.used_satellites
    assert dict(result.fix.excluded_satellites)[target.prn] == 'occluded'


def test_zero_pseudorange_not_solved():
    rx = Vec3(6378.137, 0.0, 0.0)
    dead = GnssObservation('S99', 'ok', 0.5, 0.0, 20200.0, Vec3(0, 0, -20200),
                           pseudorange_km=0.0, sigma_km=0.002)
    rx0, clock_km, fix = synthetic_fix(extra=[dead])
    assert 'S99' not in fix.used_satellites
    assert dict(fix.excluded_satellites)['S99'] == 'invalid_pseudorange'
    assert (fix.position_ecef_km - rx0).norm() < 1e-6


def test_biased_satellite_screened():
    obs = simulate_observations(SITE.ecef_km, SITE.clock_bias_s, gps_constellation(), 0.0, seed=7)
    obs = [replace(o, pseudorange_km=o.pseudorange_km + 5.0) if o.prn == 'G08' else o
           for o in obs]
    fix = solve_position(obs)
    assert 'G08' not in fix.used_satellites
    assert dict(fix.excluded_satellites)['G08'] == 'residual_outlier'
    assert (fix.position_ecef_km - SITE.ecef_km).norm() < 0.05


def test_exact_synthetic_geometry():
    rx, clock_km, fix = synthetic_fix()
    assert (fix.position_ecef_km - rx).norm() < 1e-6
    assert abs(fix.clock_bias_s - clock_km / C_KM_S) < 1e-11
    assert fix.excluded_satellites == ()


def test_insufficient_satellites_raises():
    obs = simulate_observations(SITE.ecef_km, SITE.clock_bias_s, gps_constellation(), 0.0, seed=7)
    try:
        solve_position([o for o in obs if o.status == 'ok'][:3])
        assert False
    except NoSolutionError:
        pass


def test_api_gnss_fix():
    from fastapi.testclient import TestClient
    from orbitforge.api.app import app
    c = TestClient(app)
    r = c.post('/v1/gnss/fix', json={'lat_deg': 35.0, 'lon_deg': 139.0, 'alt_km': 0.05,
                                     'clock_bias_s': 1.5e-4, 'seed': 7,
                                     'unhealthy': ['G01']})
    assert r.status_code == 200
    body = r.json()
    assert 'G01' not in body['used_satellites']
    assert body['position_error_km'] < 0.05
    assert body['dop']['gdop'] > 0.0
