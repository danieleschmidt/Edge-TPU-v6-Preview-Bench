"""Tests for QuantizationRecipe."""
from __future__ import annotations

import pytest
import numpy as np

from edge_tpu_v6_bench import QuantizationRecipe


class TestQuantizationRecipeConstructors:
    def test_int8_precision(self):
        r = QuantizationRecipe.int8()
        assert r.precision == "INT8"

    def test_int8_weight_scale(self):
        r = QuantizationRecipe.int8()
        assert r.weight_scale == pytest.approx(1.0 / 128.0)

    def test_int8_activation_scale(self):
        r = QuantizationRecipe.int8()
        assert r.activation_scale == pytest.approx(1.0 / 128.0)

    def test_int4_precision(self):
        r = QuantizationRecipe.int4()
        assert r.precision == "INT4"

    def test_int4_weight_scale(self):
        r = QuantizationRecipe.int4()
        assert r.weight_scale == pytest.approx(1.0 / 8.0)

    def test_int4_activation_scale(self):
        r = QuantizationRecipe.int4()
        assert r.activation_scale == pytest.approx(1.0 / 8.0)


class TestQuantizationApply:
    def test_int8_output_within_range(self):
        r = QuantizationRecipe.int8()
        data = np.linspace(-2.0, 2.0, 100, dtype=np.float32)
        out = r.apply(data)
        assert out.min() >= -128
        assert out.max() <= 127

    def test_int4_output_within_range(self):
        r = QuantizationRecipe.int4()
        data = np.linspace(-2.0, 2.0, 100, dtype=np.float32)
        out = r.apply(data)
        assert out.min() >= -8
        assert out.max() <= 7

    def test_output_dtype_is_int32(self):
        r = QuantizationRecipe.int8()
        out = r.apply(np.array([0.5, -0.5], dtype=np.float32))
        assert out.dtype == np.int32

    def test_zero_input_stays_zero(self):
        for recipe in [QuantizationRecipe.int8(), QuantizationRecipe.int4()]:
            out = recipe.apply(np.array([0.0], dtype=np.float32))
            assert out[0] == 0

    def test_apply_clips_large_values(self):
        r = QuantizationRecipe.int8()
        out = r.apply(np.array([1e9, -1e9], dtype=np.float32))
        assert out[0] == 127
        assert out[1] == -128

    def test_invalid_precision_raises(self):
        r = QuantizationRecipe(precision="INVALID", weight_scale=1.0, activation_scale=1.0)
        with pytest.raises(ValueError):
            r.apply(np.array([1.0]))
