"""Separate measured-sensor guard. Four axis marginals cannot prove joint safety."""
import math
import time
from .contracts import SCALES

def zero(reason, proposal=None):
    return dict(executable=False, velocity4_physical=[0.,0.,0.,0.],
                velocity4_normalized=[0.,0.,0.,0.], reason=reason, proposal=proposal,
                requires_flight_controller_failsafe=True)

def guard(proposal, telemetry, observation_ms, now_ms=None, max_age_ms=500., ttl_ms=300.):
    now = time.time()*1000 if now_ms is None else now_ms
    try:
        timestamp = float(telemetry['timestamp_ms'])
        if not all(math.isfinite(v) for v in [now,timestamp,observation_ms]): return zero('nonfinite_timestamp')
        if not 0 <= now-timestamp <= max_age_ms: return zero('stale_or_future_telemetry')
        if not 0 <= now-observation_ms <= max_age_ms: return zero('stale_or_future_observation',proposal)
        if telemetry.get('localization_valid') is not True: return zero('invalid_localization',proposal)
        if telemetry.get('emergency') is not False: return zero('emergency_or_unknown_status',proposal)
        action = [float(v) for v in proposal['proposed_physical']]
        velocity = [float(v) for v in telemetry['velocity_vehicle_mps']]
        if len(action)!=4 or len(velocity)!=3 or not all(math.isfinite(x) for x in action+velocity):
            return zero('invalid_action_or_velocity',proposal)
        if any(abs(v)>s for v,s in zip(action,SCALES)): return zero('out_of_contract_action',proposal)
        speed = max(math.sqrt(sum(v*v for v in action[:3])), math.sqrt(sum(v*v for v in velocity)))
        a = float(telemetry['brake_accel_mps2']); reaction = float(telemetry['reaction_s'])
        margin = float(telemetry['safety_margin_m'])
        if not all(math.isfinite(x) for x in [a,reaction,margin]) or a<=0 or reaction<0 or margin<0:
            return zero('invalid_braking_model',proposal)
        required = speed*(reaction+ttl_ms/1000) + speed*speed/(2*a)+margin
        # Caller must derive this from live depth/LiDAR, using current inertia,
        # proposed simultaneous xyz+yaw and vehicle footprint. Goal cannot set this mask.
        check = telemetry['joint_swept_path']
        if check.get('sensor_derived') is not True or check.get('fully_observed') is not True:
            return zero('unobserved_joint_corridor',proposal)
        if check.get('action_physical') != action: return zero('corridor_for_different_action',proposal)
        corridor_ms = float(check['timestamp_ms'])
        clearance = float(check['clearance_m'])
        if not math.isfinite(corridor_ms) or not 0<=now-corridor_ms<=max_age_ms: return zero('stale_corridor',proposal)
        if not math.isfinite(clearance) or clearance<=required: return zero('insufficient_joint_braking_clearance',proposal)
        return dict(executable=True,velocity4_physical=action,
                    velocity4_normalized=[v/s for v,s in zip(action,SCALES)],
                    reason='measured_joint_corridor_pass',required_clearance_m=required,
                    valid_until_ms=now+ttl_ms,proposal=proposal,
                    controller_frame='body_FLU_yaw_aligned')
    except (KeyError,ValueError,TypeError,OverflowError): return zero('missing_or_invalid_safety_telemetry',proposal)
