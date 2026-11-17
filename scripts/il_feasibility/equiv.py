"""State-equivalent action rows (used for scoring imitation and, later, for a set-valued imitation loss).

On the ground (and not jumping) only throttle, steer, boost and handbrake matter; in the air (not jumping) only
pitch, yaw, roll and boost. Any jump row is compared on all of jump, pitch, yaw+roll (dodge direction), roll, boost.
"""
import numpy as np


def equivalence_mask(T: np.ndarray, label: int, on_ground: bool) -> np.ndarray:
    t = T[label]
    if t[5] > 0.5:
        return np.all(T[:, [2, 4, 5, 6]] == t[[2, 4, 5, 6]], axis=1) & (T[:, 3] + T[:, 4] == t[3] + t[4])
    if on_ground:
        return (T[:, 5] == 0) & np.all(T[:, [0, 1, 6, 7]] == t[[0, 1, 6, 7]], axis=1)
    return (T[:, 5] == 0) & np.all(T[:, [2, 3, 4, 6]] == t[[2, 3, 4, 6]], axis=1)
