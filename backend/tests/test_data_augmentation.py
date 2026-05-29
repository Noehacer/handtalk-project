import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np


def test_augment_sample_returns_4_variants():
    from data_augmentation import augment_sample
    v = np.random.rand(225)
    variants = augment_sample(v)
    assert len(variants) == 4
    assert all(vv.shape == (225,) for vv in variants)


def test_mirror_swaps_hands():
    from data_augmentation import mirror_landmarks
    v = np.zeros(225)
    v[0:63] = 1.0    # left hand
    v[63:126] = 2.0  # right hand
    mirrored = mirror_landmarks(v)
    # After mirror the left slot should have right hand values
    assert not np.allclose(mirrored[0:63], v[0:63])


def test_add_noise_changes_values():
    from data_augmentation import add_noise
    v = np.ones(225)
    noisy = add_noise(v, sigma=0.1)
    assert not np.allclose(v, noisy)
    assert noisy.shape == (225,)


def test_scale_landmarks_multiplies():
    from data_augmentation import scale_landmarks
    v = np.ones(225)
    scaled = scale_landmarks(v, factor=2.0)
    assert np.allclose(scaled, 2.0)


def test_augment_sequence_returns_4_variants():
    from data_augmentation import augment_sequence
    from labels import SEQUENCE_LENGTH
    seq = np.random.rand(SEQUENCE_LENGTH, 225)
    variants = augment_sequence(seq)
    assert len(variants) == 4
    assert all(vv.shape == (SEQUENCE_LENGTH, 225) for vv in variants)


def test_mirror_preserves_shape():
    from data_augmentation import mirror_landmarks
    v = np.random.rand(225)
    result = mirror_landmarks(v)
    assert result.shape == (225,)


def test_augment_sample_first_variant_is_original():
    from data_augmentation import augment_sample
    v = np.random.rand(225)
    variants = augment_sample(v)
    assert np.allclose(variants[0], v)
