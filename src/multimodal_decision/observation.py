"""Measured state plus ordered RGB frames; no teacher/map leakage."""
import base64
import hashlib
import io
import json
import math
from pathlib import Path
from PIL import Image

MAX_IMAGES = 8
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_SOURCE_PIXELS = 16_000_000
ALLOWED_STATE = {'goal_error_vehicle_m', 'velocity_vehicle_mps', 'height_m',
                 'attitude_quaternion_xyzw', 'angular_velocity_body_radps', 'battery_fraction',
                 'localization_valid', 'localization_covariance', 'sensor_coverage',
                 'depth_sectors_m', 'mission', 'previous_velocity4_command',
                 'measured_history', 'observation_kind', 'state_timestamp_ms'}

def finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Nonfinite observation')
    if isinstance(value, dict):
        for v in value.values(): finite_json(v)
    if isinstance(value, (list, tuple)):
        for v in value: finite_json(v)

def normalize_observation(request):
    state = request.get('state', {})
    if isinstance(state, str):
        state = json.loads(state)
    if not isinstance(state, dict): raise ValueError('State must be an object')
    unexpected = set(state) - ALLOWED_STATE
    if unexpected: raise ValueError('Unsupported or privileged state fields: ' + ','.join(sorted(unexpected)))
    finite_json(state)
    shapes = {'goal_error_vehicle_m': 3, 'velocity_vehicle_mps': 3,
              'attitude_quaternion_xyzw': 4, 'angular_velocity_body_radps': 3,
              'previous_velocity4_command': 4}
    for key, size in shapes.items():
        if key in state and state[key] is not None:
            value = state[key]
            if not isinstance(value, (list, tuple)) or len(value) != size or any(
                    isinstance(v, bool) or not isinstance(v, (int, float)) for v in value):
                raise ValueError('Invalid numeric vector: ' + key)
    images = request.get('images', [])
    if not isinstance(images, list) or len(images) > MAX_IMAGES:
        raise ValueError('At most eight explicitly ordered image frames')
    out = []
    last = {}
    for item in images:
        if not isinstance(item, dict): raise ValueError('Image needs timestamp and camera metadata')
        timestamp = float(item['timestamp_ms'])
        if not math.isfinite(timestamp): raise ValueError('Nonfinite image timestamp')
        camera = str(item.get('camera_id', 'front'))
        if timestamp < last.get(camera, -math.inf): raise ValueError('Frames must be chronological per camera')
        last[camera] = timestamp
        if item.get('modality', 'rgb') != 'rgb':
            raise ValueError('Raw depth/thermal needs a separately validated sensor adapter; RGB only here')
        calibration = item.get('calibration', {})
        finite_json(calibration)
        if not isinstance(calibration, dict): raise ValueError('Calibration must be an object')
        out.append(dict(item, timestamp_ms=timestamp, camera_id=camera, modality='rgb', calibration=calibration))
    return dict(state=state, images=out)

def load_image(item, roots):
    if 'path' in item:
        path = Path(item['path']).resolve(strict=True)
        if not any(path.is_relative_to(Path(root).resolve()) for root in roots):
            raise ValueError('Image path outside configured read-only roots')
        if path.stat().st_size > MAX_IMAGE_BYTES: raise ValueError('Image file too large')
        raw = path.read_bytes()
    elif 'base64' in item:
        if len(item['base64']) > MAX_IMAGE_BYTES * 4 // 3 + 4: raise ValueError('Image payload too large')
        raw = base64.b64decode(item['base64'], validate=True)
    else:
        raise ValueError('Local path or base64 image required; network fetch disabled')
    if len(raw) > MAX_IMAGE_BYTES: raise ValueError('Image payload too large')
    with Image.open(io.BytesIO(raw)) as src:
        if src.width * src.height > MAX_SOURCE_PIXELS: raise ValueError('Image pixel count too large')
        original_size = list(src.size)
        image = src.convert('RGB')
        image.thumbnail((1024, 1024))
    return image, dict(sha256=hashlib.sha256(raw).hexdigest(), original_size=original_size,
                       timestamp_ms=item['timestamp_ms'], camera_id=item['camera_id'], modality='rgb',
                       calibration=item.get('calibration', {}))
