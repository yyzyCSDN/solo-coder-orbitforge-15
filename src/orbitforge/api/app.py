from __future__ import annotations
import math
from fastapi import FastAPI
from pydantic import BaseModel
from orbitforge.core.vector import Vec3
from orbitforge.orbits.kepler import solve_kepler_elliptic
from orbitforge.maneuvers.hohmann import hohmann
from orbitforge.link.budget import free_space_loss_db
from orbitforge.environment.eclipse import eclipse_state
from orbitforge.attitude.quaternion import Quaternion
from orbitforge.gnss.constellation import gps_constellation
from orbitforge.gnss.receiver import ReceiverSite, measure_and_fix
from orbitforge.storage.sqlite import Store
app = FastAPI(title='OrbitForge Mission Lab', version='1.0.0')
store = Store(':memory:')

class KeplerReq(BaseModel):
    mean_anomaly: float
    eccentricity: float

class HohmannReq(BaseModel):
    r1_km: float
    r2_km: float

class LinkReq(BaseModel):
    range_km: float
    freq_hz: float

class EclipseReq(BaseModel):
    sat: list[float]
    sun: list[float]

class RotateReq(BaseModel):
    q: list[float]
    v: list[float]

class GnssFixReq(BaseModel):
    lat_deg: float
    lon_deg: float
    alt_km: float = 0.0
    epoch_tai_s: float = 0.0
    clock_bias_s: float = 0.0
    elevation_mask_deg: float = 5.0
    seed: int = 1
    unhealthy: list[str] = []

@app.get('/live')
def live():
    return {'status': 'live'}

@app.get('/ready')
def ready():
    return {'status': 'ready'}

@app.post('/v1/orbit/kepler')
def kepler(r: KeplerReq):
    return {'eccentric_anomaly': solve_kepler_elliptic(r.mean_anomaly, r.eccentricity)}

@app.post('/v1/maneuver/hohmann')
def h(r: HohmannReq):
    return hohmann(r.r1_km, r.r2_km)

@app.post('/v1/link/fspl')
def l(r: LinkReq):
    return {'loss_db': free_space_loss_db(r.range_km, r.freq_hz)}

@app.post('/v1/environment/eclipse')
def e(r: EclipseReq):
    return {'state': eclipse_state(Vec3(*r.sat), Vec3(*r.sun))}

@app.post('/v1/attitude/rotate')
def rotate(r: RotateReq):
    q = Quaternion(*r.q)
    v = q.rotate(Vec3(*r.v))
    return {'v': v.as_tuple()}

@app.post('/v1/gnss/fix')
def gnss_fix(r: GnssFixReq):
    site = ReceiverSite(math.radians(r.lat_deg), math.radians(r.lon_deg), r.alt_km,
                        r.clock_bias_s, math.radians(r.elevation_mask_deg))
    result = measure_and_fix(site, gps_constellation(), r.epoch_tai_s,
                             seed=r.seed, unhealthy_prns=frozenset(r.unhealthy))
    return result.summary()

@app.get('/v1/system/audit')
def audit():
    return store.audit_chain()

def main():
    import uvicorn
    uvicorn.run('orbitforge.api.app:app', host='127.0.0.1', port=8080)
