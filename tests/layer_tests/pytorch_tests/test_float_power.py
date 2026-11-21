# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestFloatPower(PytorchLayerTest):
    def _prepare_input(self):
        return (np.array(self.lhs, dtype=self.lhs_dtype), np.array(self.rhs, dtype=self.rhs_dtype))

    def create_model(self):
        class aten_float_power(torch.nn.Module):
            def forward(self, input_data, exponent):
                return torch.float_power(input_data, exponent)

        return aten_float_power(), "aten::float_power"

    @pytest.mark.parametrize(("lhs", "rhs", "lhs_dtype", "rhs_dtype"), [
        # same-dtype tensor-tensor
        ([[1.0, 2.0], [3.0, 4.0]], [[1.0, 1.0], [2.0, 2.0]], np.float32, np.float32),
        ([[1.0, 2.0], [3.0, 4.0]], [[1.0, 1.0], [2.0, 2.0]], np.float64, np.float64),
        ([[1, 2], [3, 4]], [[1, 1], [2, 2]], np.int32, np.int32),
        ([[1, 2], [3, 4]], [[1, 1], [2, 2]], np.int64, np.int64),
        # mixed dtypes
        ([[1.0, 2.0], [3.0, 4.0]], [[1, 1], [2, 2]], np.float32, np.int64),
        ([[1, 2], [3, 4]], [[1.0, 1.0], [2.0, 2.0]], np.int32, np.float32),
        ([[1.0, 2.0], [3.0, 4.0]], [[1.0, 1.0], [2.0, 2.0]], np.float32, np.float64),
        ([[1, 2], [3, 4]], [[1, 1], [2, 2]], np.int64, np.int32),
        # broadcasting
        ([[1.0, 2.0], [3.0, 4.0]], [2.0, 3.0], np.float32, np.float32),
        ([5.0, 6.0], [[1.0, 2.0], [3.0, 4.0]], np.float32, np.float32),
        ([[1.0, 2.0, 3.0]], [[2.0], [3.0]], np.float64, np.float32),
        # negative bases: integer exponents stay finite; fractional exponents are NaN
        ([-2.0, -4.0, 9.0, 2.0], [3.0, 2.0, 0.5, -1.0], np.float32, np.float32),
        ([-2.0, -4.0], [0.5, 1.5], np.float64, np.float64),
        ([-2, -3, 4], [3, 2, 2], np.int32, np.int32),
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_float_power(self, lhs, rhs, lhs_dtype, rhs_dtype, ie_device, precision, ir_version):
        self.lhs = lhs
        self.rhs = rhs
        self.lhs_dtype = lhs_dtype
        self.rhs_dtype = rhs_dtype
        self._test(*self.create_model(), ie_device, precision, ir_version)


class TestFloatPowerZeroToNegative(PytorchLayerTest):
    # torch.float_power(0, -n) is +inf. The CPU plugin evaluates f64 Power as f32
    # and saturates at FLT_MAX (~3.4e38) instead of inf, so framework values are
    # not compared here; conversion and inference are still required to succeed.
    skip_framework = True

    def _prepare_input(self):
        return (np.array([0.0, 0.0], dtype=np.float32),
                np.array([-1.0, -2.0], dtype=np.float32))

    def create_model(self):
        class aten_float_power(torch.nn.Module):
            def forward(self, input_data, exponent):
                return torch.float_power(input_data, exponent)

        return aten_float_power(), "aten::float_power"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_float_power_zero_to_negative(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version)


class TestFloatPowerScalarExponent(PytorchLayerTest):
    def _prepare_input(self):
        return (np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),)

    def create_model(self):
        class aten_float_power(torch.nn.Module):
            def forward(self, x):
                return torch.float_power(x, 2)

        return aten_float_power(), "aten::float_power"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_float_power_scalar_exponent(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version)


class TestFloatPowerScalarBase(PytorchLayerTest):
    def _prepare_input(self):
        return (np.array([1.0, 2.0, 3.0], dtype=np.float32),)

    def create_model(self):
        class aten_float_power(torch.nn.Module):
            def forward(self, x):
                return torch.float_power(2, x)

        return aten_float_power(), "aten::float_power"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_float_power_scalar_base(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version)


class TestFloatPowerInplace(PytorchLayerTest):
    def _prepare_input(self):
        return (np.array([2.0, 3.0, 4.0], dtype=np.float64),
                np.array([2.0, 1.0, 0.5], dtype=np.float64))

    def create_model(self):
        class aten_float_power_(torch.nn.Module):
            def forward(self, x, exponent):
                return x.float_power_(exponent)

        return aten_float_power_(), "aten::float_power_"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_float_power_inplace(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version)


class TestFloatPowerOut(PytorchLayerTest):
    def _prepare_input(self):
        x = np.array([2.0, 3.0, 4.0], dtype=np.float32)
        exponent = np.array([2.0, 1.0, 0.5], dtype=np.float32)
        return (x, exponent, np.zeros_like(x, dtype=np.float64))

    def create_model(self):
        class aten_float_power_out(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.out_op = PytorchLayerTest.out_variant(torch.float_power,
                                                           torch.ops.aten.float_power.Tensor_Tensor_out)

            def forward(self, x, exponent, out):
                return self.out_op(x, exponent, out=out), out

        return aten_float_power_out(), "aten::float_power"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_float_power_out(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version)
