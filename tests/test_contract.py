import math
import pytest
from multimodal_decision.contracts import contract, VALUES
from multimodal_decision.observation import normalize_observation
from multimodal_decision.safety import guard

def test_latest_contract_full_cartesian_and_simultaneous_turn():
    c=contract()
    assert c['cartesian_action_count']==2401 and not c['legacy_26_options']
    assert all(min(v)<0<max(v) for v in VALUES)
    assert [v[4] for v in VALUES]==[.45,.45,.45,.15]

def test_privileged_and_nonfinite_rejected():
    for state in [{'obstacle_map': []},{'height_m':math.nan}]:
        with pytest.raises(ValueError): normalize_observation({'state':state})

def test_bad_sensor_vectors_rejected():
    for state in [{'velocity_vehicle_mps':[1,2]}, {'goal_error_vehicle_m':['x',0,0]},
                  {'attitude_quaternion_xyzw':[True,0,0,1]}]:
        with pytest.raises(ValueError): normalize_observation({'state':state})

def test_chronological_and_modality_contract():
    with pytest.raises(ValueError):
        normalize_observation({'images':[dict(timestamp_ms=2),dict(timestamp_ms=1)]})
    with pytest.raises(ValueError): normalize_observation({'images':[dict(timestamp_ms=1,modality='depth')]})

def fixture():
    proposal={'proposed_physical':[.45,.45,0.,.15]}
    telemetry=dict(timestamp_ms=1000,localization_valid=True,emergency=False,
        velocity_vehicle_mps=[.1,0.,0.],brake_accel_mps2=1.5,reaction_s=.2,safety_margin_m=.2,
        joint_swept_path=dict(timestamp_ms=1000,fully_observed=True,sensor_derived=True,
            action_physical=[.45,.45,0.,.15],clearance_m=3.))
    return proposal,telemetry

def test_fresh_joint_path_accepts_combined_motion_and_expiry():
    p,t=fixture(); r=guard(p,t,1000,now_ms=1100)
    assert r['executable'] and r['velocity4_physical']==p['proposed_physical']
    assert r['valid_until_ms']==1400

@pytest.mark.parametrize('mutation,reason',[
    ('stale','stale_or_future_observation'),('unobserved','unobserved_joint_corridor'),
    ('different','corridor_for_different_action'),('clearance','insufficient_joint_braking_clearance'),
    ('nan','insufficient_joint_braking_clearance'),('emergency','emergency_or_unknown_status')])
def test_rejects_unsafe_joint_or_delayed_commands(mutation,reason):
    p,t=fixture(); observation_ms=1000
    if mutation=='stale': observation_ms=0
    if mutation=='unobserved': t['joint_swept_path']['fully_observed']=False
    if mutation=='different': t['joint_swept_path']['action_physical']=[.45,0.,0.,.15]
    if mutation=='clearance': t['joint_swept_path']['clearance_m']=.1
    if mutation=='nan': t['joint_swept_path']['clearance_m']=math.nan
    if mutation=='emergency': t['emergency']=True
    r=guard(p,t,observation_ms,now_ms=1100)
    assert not r['executable'] and r['reason']==reason
    assert r['velocity4_physical']==[0.,0.,0.,0.]
