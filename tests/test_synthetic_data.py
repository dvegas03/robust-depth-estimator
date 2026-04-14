"""Extensive tests for SyntheticDataGenerator."""

import os

import numpy as np
import pytest

from engine_grounder.utils.synthetic_data import SyntheticDataGenerator


# ===================================================================
# 1. OUTPUT SHAPE & DTYPE
# ===================================================================
class TestOutputProperties:
    def test_default_shape(self):
        gen = SyntheticDataGenerator(size=(100, 100))
        crop = gen.generate_noisy_crop()
        assert crop.shape == (100, 100)

    def test_rectangular_shape(self):
        gen = SyntheticDataGenerator(size=(30, 80))
        crop = gen.generate_noisy_crop()
        assert crop.shape == (30, 80)

    def test_single_pixel(self):
        gen = SyntheticDataGenerator(size=(1, 1))
        crop = gen.generate_noisy_crop(void_ratio=0.0, seed=0)
        assert crop.shape == (1, 1)
        assert crop[0, 0] > 0

    def test_output_is_float(self):
        gen = SyntheticDataGenerator(size=(10, 10))
        crop = gen.generate_noisy_crop()
        assert np.issubdtype(crop.dtype, np.floating)


# ===================================================================
# 2. VOID RATIO
# ===================================================================
class TestVoidRatio:
    def test_zero_void_has_no_zeros(self):
        gen = SyntheticDataGenerator(size=(50, 50), true_depth=1.0)
        crop = gen.generate_noisy_crop(void_ratio=0.0, noise_std=0.01, seed=0)
        assert np.all(crop > 0), "void_ratio=0 must produce no zeros"

    def test_full_void(self):
        gen = SyntheticDataGenerator(size=(50, 50))
        crop = gen.generate_noisy_crop(void_ratio=1.0, seed=0)
        assert np.all(crop == 0.0), "void_ratio=1 must produce all zeros"

    def test_void_ratio_approximately_correct(self):
        gen = SyntheticDataGenerator(size=(100, 100))
        crop = gen.generate_noisy_crop(void_ratio=0.30, seed=7)
        actual_ratio = np.sum(crop == 0.0) / crop.size
        assert actual_ratio == pytest.approx(0.30, abs=0.02)

    def test_void_ratio_negative_raises(self):
        gen = SyntheticDataGenerator()
        with pytest.raises(ValueError, match="void_ratio"):
            gen.generate_noisy_crop(void_ratio=-0.1)

    def test_void_ratio_above_one_raises(self):
        gen = SyntheticDataGenerator()
        with pytest.raises(ValueError, match="void_ratio"):
            gen.generate_noisy_crop(void_ratio=1.5)


# ===================================================================
# 3. PHYSICAL PLAUSIBILITY
# ===================================================================
class TestPhysicalPlausibility:
    def test_no_negative_depths(self):
        """Gaussian noise must never produce negative depth values."""
        gen = SyntheticDataGenerator(size=(200, 200), true_depth=0.1)
        crop = gen.generate_noisy_crop(void_ratio=0.0, noise_std=0.5, seed=99)
        assert np.all(crop >= 0.0)

    def test_mean_near_true_depth(self):
        gen = SyntheticDataGenerator(size=(200, 200), true_depth=2.0)
        crop = gen.generate_noisy_crop(void_ratio=0.0, noise_std=0.05, seed=0)
        assert np.mean(crop) == pytest.approx(2.0, abs=0.02)


# ===================================================================
# 4. REPRODUCIBILITY
# ===================================================================
class TestReproducibility:
    def test_same_seed_same_output(self):
        gen = SyntheticDataGenerator(size=(50, 50))
        a = gen.generate_noisy_crop(void_ratio=0.3, seed=42)
        b = gen.generate_noisy_crop(void_ratio=0.3, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_different_output(self):
        gen = SyntheticDataGenerator(size=(50, 50))
        a = gen.generate_noisy_crop(void_ratio=0.3, seed=1)
        b = gen.generate_noisy_crop(void_ratio=0.3, seed=2)
        assert not np.array_equal(a, b)


# ===================================================================
# 5. VALIDATION
# ===================================================================
class TestValidation:
    def test_negative_true_depth_raises(self):
        with pytest.raises(ValueError, match="true_depth"):
            SyntheticDataGenerator(true_depth=-1.0)

    def test_zero_true_depth_raises(self):
        with pytest.raises(ValueError, match="true_depth"):
            SyntheticDataGenerator(true_depth=0.0)
