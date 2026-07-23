"""Unit tests for the low-bit storage and numerical invariants."""

import io
import unittest

import torch
from torch import nn

from core import (
    fake_quantize_ste,
    pack_int4,
    pack_uint4,
    quantize_tensor,
    unpack_int4,
    unpack_uint4,
)
from linear import (
    QATLinear,
    QuantizedLinear,
    quantize_model_linears,
)
from nf4 import quantize_nf4
from q4_k import (
    pack_q4_k_codes,
    pack_scale_min_6bit,
    quantize_q4_k,
    unpack_q4_k_codes,
    unpack_scale_min_6bit,
)
from q8_0 import Q8_0Linear, quantize_q8_0


class PackingTests(unittest.TestCase):
    def test_signed_int4_round_trip(self):
        values = torch.arange(-8, 8, dtype=torch.int8).reshape(2, 8)
        packed = pack_int4(values)
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(packed.numel() * 2, values.numel())
        self.assertTrue(torch.equal(unpack_int4(packed), values))

    def test_unsigned_int4_round_trip(self):
        values = torch.arange(16, dtype=torch.uint8).reshape(2, 8)
        packed = pack_uint4(values)
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertTrue(torch.equal(unpack_uint4(packed), values))


class AffineQuantizationTests(unittest.TestCase):
    def test_shapes_padding_and_finite_values(self):
        torch.manual_seed(1)
        for symmetric in (True, False):
            for bits in (4, 8):
                for shape in ((5,), (2, 5), (3, 7, 9)):
                    with self.subTest(
                        symmetric=symmetric, bits=bits, shape=shape
                    ):
                        x = torch.randn(*shape)
                        qt = quantize_tensor(
                            x,
                            bits=bits,
                            group_size=4,
                            symmetric=symmetric,
                        )
                        restored = qt.dequantize()
                        self.assertEqual(tuple(restored.shape), shape)
                        self.assertTrue(torch.isfinite(restored).all())

    def test_zero_tensor(self):
        for symmetric in (True, False):
            qt = quantize_tensor(
                torch.zeros(3, 11),
                bits=4,
                group_size=8,
                symmetric=symmetric,
            )
            self.assertTrue(torch.equal(qt.dequantize(), torch.zeros(3, 11)))

    def test_asymmetric_zero_is_representable(self):
        x = torch.tensor([[2.0, 3.0, 4.0, 5.0]])
        qt = quantize_tensor(
            x,
            bits=4,
            group_size=4,
            symmetric=False,
            pack=False,
        )
        # Positive-only range is expanded to start at real zero.
        self.assertEqual(int(qt.zero_points.item()), 0)
        self.assertLess((qt.dequantize() - x).abs().max().item(), 0.2)

    def test_group32_int4_fp16_scale_is_4_5_bpw(self):
        x = torch.randn(16, 64)
        qt = quantize_tensor(
            x,
            bits=4,
            group_size=32,
            symmetric=True,
            scale_dtype=torch.float16,
        )
        self.assertEqual(qt.payload_nbytes, 512)
        self.assertEqual(qt.metadata_nbytes, 64)
        self.assertEqual(qt.effective_bits_per_weight, 4.5)

    def test_ste_identity_gradient(self):
        x = torch.tensor([0.12, -0.37, 0.88], requires_grad=True)
        y = fake_quantize_ste(x, bits=4, group_size=3)
        y.sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.ones_like(x)))


class LinearTests(unittest.TestCase):
    def test_quantized_linear_has_no_float_weight(self):
        torch.manual_seed(2)
        base = nn.Linear(7, 5)
        layer = QuantizedLinear.from_float(
            base, bits=4, group_size=4
        )
        self.assertFalse(hasattr(layer, "weight"))
        self.assertEqual(layer.qweight.dtype, torch.uint8)
        self.assertNotIn("weight", layer.state_dict())
        self.assertIn("qweight", layer.state_dict())
        self.assertEqual(tuple(layer(torch.randn(3, 7)).shape), (3, 5))

    def test_quantized_linear_state_round_trip(self):
        torch.manual_seed(3)
        base = nn.Linear(8, 4)
        original = QuantizedLinear.from_float(base, bits=4, group_size=4)
        rebuilt = QuantizedLinear.from_float(base, bits=4, group_size=4)
        buffer = io.BytesIO()
        torch.save(original.state_dict(), buffer)
        buffer.seek(0)
        rebuilt.load_state_dict(torch.load(buffer, weights_only=True))
        x = torch.randn(2, 8)
        self.assertTrue(torch.equal(original(x), rebuilt(x)))

    def test_qat_gradient_and_conversion(self):
        torch.manual_seed(4)
        qat = QATLinear(8, 4, bits=4, group_size=4)
        loss = qat(torch.randn(3, 8)).square().mean()
        loss.backward()
        self.assertIsNotNone(qat.weight.grad)
        packed = qat.to_quantized()
        self.assertIsInstance(packed, QuantizedLinear)
        self.assertEqual(packed.qweight.dtype, torch.uint8)

    def test_model_replacement_storage_order(self):
        torch.manual_seed(5)
        float_model = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)
        )
        int8_model = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)
        )
        int4_model = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)
        )
        int8_report = quantize_model_linears(
            int8_model, bits=8, group_size=8, exclude=()
        )
        int4_report = quantize_model_linears(
            int4_model, bits=4, group_size=8, exclude=()
        )
        self.assertLess(
            int4_report.quantized_weight_bytes,
            int8_report.quantized_weight_bytes,
        )
        self.assertLess(
            int8_report.quantized_weight_bytes,
            int8_report.fp32_weight_bytes,
        )


class FormatTests(unittest.TestCase):
    def test_q4_k_scale_min_pack_round_trip(self):
        torch.manual_seed(6)
        scales = torch.randint(0, 64, (5, 8), dtype=torch.uint8)
        mins = torch.randint(0, 64, (5, 8), dtype=torch.uint8)
        packed = pack_scale_min_6bit(scales, mins)
        scales2, mins2 = unpack_scale_min_6bit(packed)
        self.assertEqual(tuple(packed.shape), (5, 12))
        self.assertTrue(torch.equal(scales, scales2))
        self.assertTrue(torch.equal(mins, mins2))

    def test_q4_k_code_pack_round_trip(self):
        torch.manual_seed(7)
        codes = torch.randint(0, 16, (3, 8, 32), dtype=torch.uint8)
        packed = pack_q4_k_codes(codes)
        self.assertEqual(tuple(packed.shape), (3, 128))
        self.assertTrue(torch.equal(codes, unpack_q4_k_codes(packed)))

    def test_q4_k_exact_block_size(self):
        qt = quantize_q4_k(torch.randn(256))
        self.assertEqual(qt.storage_nbytes, 144)
        self.assertEqual(qt.effective_bits_per_weight, 4.5)
        self.assertTrue(torch.isfinite(qt.dequantize()).all())

    def test_nf4_normal_sample(self):
        torch.manual_seed(0)
        values = torch.randn(4096)
        nf4 = quantize_nf4(values, block_size=64)
        uniform = quantize_tensor(
            values,
            bits=4,
            group_size=64,
            symmetric=True,
            scale_dtype=torch.float16,
        )
        nf4_mse = (nf4.dequantize() - values).square().mean().item()
        uniform_mse = (
            uniform.dequantize() - values
        ).square().mean().item()
        self.assertLess(nf4_mse, uniform_mse)
        self.assertEqual(nf4.packed_indices.dtype, torch.uint8)


class Q8_0Tests(unittest.TestCase):
    def test_exact_block_layout_and_codes(self):
        values = torch.tensor(
            [-12.7, -6.35, -3.2, -1.05, 0.0, 0.14, 1.25, 4.4, 8.88, 12.7]
            + [0.0] * 22
        )
        qt = quantize_q8_0(values)
        expected = torch.tensor(
            [-127, -64, -32, -11, 0, 1, 13, 44, 89, 127],
            dtype=torch.int8,
        )
        self.assertTrue(torch.equal(qt.qs[0, :10], expected))
        self.assertEqual(qt.qs.dtype, torch.int8)
        self.assertEqual(qt.d.dtype, torch.float16)
        self.assertEqual(qt.payload_nbytes, 32)
        self.assertEqual(qt.metadata_nbytes, 2)
        self.assertEqual(qt.storage_nbytes, 34)
        self.assertEqual(qt.effective_bits_per_weight, 8.5)

    def test_zero_block_and_row_validation(self):
        qt = quantize_q8_0(torch.zeros(3, 32))
        self.assertTrue(torch.equal(qt.qs, torch.zeros_like(qt.qs)))
        self.assertTrue(torch.equal(qt.dequantize(), torch.zeros(3, 32)))
        with self.assertRaises(ValueError):
            quantize_q8_0(torch.zeros(2, 31))

    def test_block_scale_protects_small_values(self):
        small = torch.linspace(-0.25, 0.25, 32)
        large = torch.linspace(-20.0, 20.0, 32)
        values = torch.cat((small, large))
        block_hat = quantize_q8_0(values).dequantize()
        global_hat = quantize_tensor(
            values,
            bits=8,
            group_size=64,
            symmetric=True,
            scale_dtype=torch.float16,
        ).dequantize()
        block_mse = (block_hat[:32] - small).square().mean()
        global_mse = (global_hat[:32] - small).square().mean()
        self.assertLess(block_mse.item(), global_mse.item())

    def test_q8_linear_has_no_float_weight_and_reloads(self):
        torch.manual_seed(8)
        base = nn.Linear(64, 16)
        original = Q8_0Linear.from_float(base)
        rebuilt = Q8_0Linear.from_float(base)
        self.assertFalse(hasattr(original, "weight"))
        self.assertEqual(original.qweight.dtype, torch.int8)
        self.assertEqual(original.scales.dtype, torch.float16)
        self.assertNotIn("weight", original.state_dict())

        buffer = io.BytesIO()
        torch.save(original.state_dict(), buffer)
        buffer.seek(0)
        rebuilt.load_state_dict(torch.load(buffer, weights_only=True))
        x = torch.randn(2, 64)
        self.assertTrue(torch.equal(original(x), rebuilt(x)))


if __name__ == "__main__":
    unittest.main()
