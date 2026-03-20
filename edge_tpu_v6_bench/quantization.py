"""Quantization recipes for INT8 and INT4 precision."""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np


@dataclass
class QuantizationRecipe:
    """Describes how to quantize weights and activations.

    Attributes
    ----------
    precision: Target numeric precision — 'INT8' or 'INT4'.
    weight_scale: Scale factor applied to weights during quantization.
    activation_scale: Scale factor applied to activations during quantization.
    """

    precision: str
    weight_scale: float
    activation_scale: float

    # Precision → integer bit-range
    _RANGES: ClassVar[dict[str, tuple[int, int]]] = {
        "INT8": (-128, 127),
        "INT4": (-8, 7),
    }

    # ------------------------------------------------------------------ #
    # Class-method constructors                                            #
    # ------------------------------------------------------------------ #

    @classmethod
    def int8(cls) -> "QuantizationRecipe":
        """Return the standard INT8 quantization recipe."""
        return cls(precision="INT8", weight_scale=1.0 / 128.0, activation_scale=1.0 / 128.0)

    @classmethod
    def int4(cls) -> "QuantizationRecipe":
        """Return the standard INT4 quantization recipe."""
        return cls(precision="INT4", weight_scale=1.0 / 8.0, activation_scale=1.0 / 8.0)

    # ------------------------------------------------------------------ #
    # Core operation                                                       #
    # ------------------------------------------------------------------ #

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Quantize *data* to the integer range defined by this recipe.

        The array is:
        1. Divided by *weight_scale* (simulates quantization step).
        2. Rounded to the nearest integer.
        3. Clipped to the valid range for the chosen precision.

        Parameters
        ----------
        data: Floating-point array to quantize.

        Returns
        -------
        Quantized array with dtype int32.
        """
        if self.precision not in self._RANGES:
            raise ValueError(
                f"Unsupported precision '{self.precision}'. "
                f"Expected one of {list(self._RANGES)}"
            )

        lo, hi = self._RANGES[self.precision]
        scale = self.weight_scale if self.weight_scale != 0 else 1.0
        quantized = np.round(np.asarray(data, dtype=np.float64) / scale)
        clipped = np.clip(quantized, lo, hi).astype(np.int32)
        return clipped
