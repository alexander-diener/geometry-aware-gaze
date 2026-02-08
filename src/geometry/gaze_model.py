import numpy as np
from .camera import yaw_pitch_to_unit

def geometry_baseline(head_yaw: np.ndarray, head_pitch: np.ndarray,
                      eye_yaw: np.ndarray, eye_pitch: np.ndarray) -> np.ndarray:
    """
    Simple geometry-inspired baseline: gaze = head orientation + relative eye orientation (in angle space).
    This is intentionally simplistic to demonstrate inductive bias.
    """
    yaw = head_yaw + eye_yaw
    pitch = head_pitch + eye_pitch
    return yaw_pitch_to_unit(yaw, pitch)