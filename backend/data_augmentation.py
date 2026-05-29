import numpy as np


def mirror_landmarks(v: np.ndarray) -> np.ndarray:
    """Mirrors hands (swaps left/right, negates X coords).
    Holistic layout: left[0:63], right[63:126], face[126:225].
    """
    out = v.copy()
    left_part  = v[0:63].copy()
    right_part = v[63:126].copy()
    # Negate X coordinate (index 0 of each triplet x,y,z)
    left_part[0::3]  *= -1
    right_part[0::3] *= -1
    # Swap: old right → new left slot, old left → new right slot
    out[0:63]   = right_part
    out[63:126] = left_part
    # Face: negate X
    out[126:225:3] *= -1
    return out


def add_noise(v: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    return v + np.random.normal(0, sigma, v.shape).astype(v.dtype)


def scale_landmarks(v: np.ndarray, factor: float | None = None) -> np.ndarray:
    if factor is None:
        factor = float(np.random.uniform(0.9, 1.1))
    return v * factor


def augment_sample(v: np.ndarray) -> list[np.ndarray]:
    """Returns 4 variants of a landmark vector (225,): original, mirrored, noisy, scaled."""
    return [
        v,
        mirror_landmarks(v),
        add_noise(v),
        scale_landmarks(v),
    ]


def augment_sequence(seq: np.ndarray) -> list[np.ndarray]:
    """Returns 4 variants of a sequence (T, 225): original, mirrored, noisy, scaled."""
    factor = float(np.random.uniform(0.9, 1.1))
    return [
        seq,
        np.array([mirror_landmarks(f) for f in seq]),
        np.array([add_noise(f) for f in seq]),
        np.array([scale_landmarks(f, factor) for f in seq]),
    ]
