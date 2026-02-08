import numpy as np

def normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + eps)

def yaw_pitch_to_unit(yaw: np.ndarray, pitch: np.ndarray) -> np.ndarray:
    """
    Converts yaw/pitch (rad) to 3D unit vector in camera coordinates.
    Convention:
      x: right, y: down, z: forward
    """
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    # z forward
    x = sy * cp
    y = sp
    z = cy * cp
    v = np.stack([x, y, z], axis=-1)
    return normalize(v)

def angular_error_deg(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    a = normalize(a, eps)
    b = normalize(b, eps)
    dot = np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))