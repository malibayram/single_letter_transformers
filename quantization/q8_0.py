"""q8_0.py — llama.cpp Q8_0 block format, byte by byte.

Q8_0 is a block-wise, symmetric 8-bit tensor encoding. Every block stores:

    32 bytes  32 signed int8 codes
     2 bytes  one FP16 scale (`d`, or delta)
    ------------------------------------------
    34 bytes  = 272 bits / 32 weights = 8.5 bits per weight

The current llama.cpp reference quantizer uses:

    amax = max(abs(x))
    d    = amax / 127
    q    = round(x / d), clamped to [-127, 127]
    x_hat = q * FP16(d)

The integer codes are calculated from the FP32 `d`; the stored scale is FP16.
That tiny distinction is preserved here. The implementation also preserves
llama.cpp's 32-value block requirement instead of silently padding rows.

This is a readable storage/numerical reference. It does not write GGUF and its
forward path dequantizes before a floating-point matrix multiplication, so it
does not claim production inference speed.

Run:
    PYTHONPATH=quantization .venv/bin/python quantization/q8_0.py
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from core import _round_half_away_from_zero, quantize_tensor, tensor_nbytes
from linear import (
    ModelQuantizationReport,
    _replace_module,
    _selected_linear_names,
)


QK8_0 = 32
Q8_0_QMIN = -127
Q8_0_QMAX = 127


@dataclass
class Q8_0Tensor:
    """A structurally faithful collection of Q8_0 blocks.

    `qs` has shape `[n_blocks, 32]` and dtype `torch.int8`.
    `d` has shape `[n_blocks]` and dtype `torch.float16`.
    No original floating-point tensor is retained.
    """

    qs: torch.Tensor
    d: torch.Tensor
    original_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.qs.dtype != torch.int8:
            raise TypeError(f"Q8_0 codes must be int8, got {self.qs.dtype}")
        if self.d.dtype != torch.float16:
            raise TypeError(f"Q8_0 scales must be float16, got {self.d.dtype}")
        if self.qs.ndim != 2 or self.qs.shape[1] != QK8_0:
            raise ValueError(
                f"Q8_0 codes must have shape [blocks, {QK8_0}], "
                f"got {tuple(self.qs.shape)}"
            )
        if tuple(self.d.shape) != (self.qs.shape[0],):
            raise ValueError("Q8_0 needs exactly one FP16 scale per block")
        if math.prod(self.original_shape) != self.qs.numel():
            raise ValueError("Q8_0 logical shape and block payload disagree")

    @property
    def numel(self) -> int:
        return math.prod(self.original_shape)

    @property
    def n_blocks(self) -> int:
        return self.qs.shape[0]

    @property
    def payload_nbytes(self) -> int:
        return tensor_nbytes(self.qs)

    @property
    def metadata_nbytes(self) -> int:
        return tensor_nbytes(self.d)

    @property
    def storage_nbytes(self) -> int:
        return self.payload_nbytes + self.metadata_nbytes

    @property
    def effective_bits_per_weight(self) -> float:
        return 8.0 * self.storage_nbytes / self.numel

    def dequantize(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Decode each value with `x_hat = int8_code * FP16_scale`."""
        restored = self.qs.to(torch.float32) * self.d.to(torch.float32).unsqueeze(1)
        return restored.reshape(self.original_shape).to(dtype)


def quantize_q8_0(tensor: torch.Tensor) -> Q8_0Tensor:
    """Quantize a tensor using llama.cpp Q8_0's reference equations.

    GGML quantized rows must contain complete blocks. For a matrix, the last
    dimension is the row length, so it must be divisible by 32. This explicit
    check prevents an educational convenience padding rule from being mistaken
    for the real on-disk encoding.
    """
    if not tensor.is_floating_point() or tensor.numel() == 0:
        raise ValueError("expected a non-empty floating-point tensor")
    if tensor.shape[-1] % QK8_0 != 0:
        raise ValueError(
            f"Q8_0 row length must be divisible by {QK8_0}, "
            f"got last dimension {tensor.shape[-1]}"
        )

    original_shape = tuple(tensor.shape)
    blocks = tensor.detach().to(torch.float32).reshape(-1, QK8_0)

    # This follows quantize_row_q8_0_ref in ggml-quants.c:
    # codes use the FP32 scale, while dequantization later uses stored FP16 d.
    amax = blocks.abs().amax(dim=1)
    d_float = amax / Q8_0_QMAX
    inverse_scale = torch.where(
        d_float > 0,
        d_float.reciprocal(),
        torch.zeros_like(d_float),
    )
    qs = _round_half_away_from_zero(
        blocks * inverse_scale.unsqueeze(1)
    ).clamp(Q8_0_QMIN, Q8_0_QMAX)

    return Q8_0Tensor(
        qs=qs.to(torch.int8).contiguous(),
        d=d_float.to(torch.float16).contiguous(),
        original_shape=original_shape,
    )


def q8_0_layout_report(qt: Q8_0Tensor) -> str:
    """Return exact payload/metadata accounting for display and tests."""
    return (
        f"Q8_0: {qt.n_blocks} block(s), "
        f"int8 codes={qt.payload_nbytes} B, "
        f"FP16 scales={qt.metadata_nbytes} B, "
        f"total={qt.storage_nbytes} B, "
        f"effective={qt.effective_bits_per_weight:.3f} bpw"
    )


class Q8_0Linear(nn.Module):
    """Inference-only Linear whose weight is stored as real Q8_0 blocks."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        bias: torch.Tensor | None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.original_shape = (out_features, in_features)

        # Buffers move with the module and are serialized, but are not trainable.
        # There is intentionally no full-precision `weight` Parameter.
        self.register_buffer("qweight", qweight)
        self.register_buffer("scales", scales)
        self.register_buffer("bias", bias)

    @classmethod
    def from_float(cls, layer: nn.Linear) -> "Q8_0Linear":
        qt = quantize_q8_0(layer.weight)
        bias = None if layer.bias is None else layer.bias.detach().clone()
        return cls(
            layer.in_features,
            layer.out_features,
            qweight=qt.qs,
            scales=qt.d,
            bias=bias,
        )

    def as_quantized_tensor(self) -> Q8_0Tensor:
        return Q8_0Tensor(
            qs=self.qweight,
            d=self.scales,
            original_shape=self.original_shape,
        )

    def dequantize_weight(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return self.as_quantized_tensor().dequantize(dtype=dtype)

    @property
    def weight_storage_nbytes(self) -> int:
        return self.as_quantized_tensor().storage_nbytes

    @property
    def total_storage_nbytes(self) -> int:
        return self.weight_storage_nbytes + tensor_nbytes(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.dequantize_weight(dtype=x.dtype)
        bias = None if self.bias is None else self.bias.to(dtype=x.dtype)
        return F.linear(x, weight, bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"block_size={QK8_0}, weight_storage={self.weight_storage_nbytes} B"
        )


def quantize_model_q8_0(
    model: nn.Module,
    *,
    include: tuple[str, ...] | str = "all",
    exclude: tuple[str, ...] = ("lm_head",),
) -> ModelQuantizationReport:
    """Replace selected `nn.Linear` modules with exact-layout Q8_0Linear."""
    selected = _selected_linear_names(model, include=include, exclude=exclude)
    if not selected:
        raise ValueError("no nn.Linear layer matched include/exclude settings")

    names: list[str] = []
    weight_count = 0
    fp32_bytes = 0
    q8_bytes = 0
    for name, layer in selected:
        quantized = Q8_0Linear.from_float(layer)
        _replace_module(model, name, quantized)
        names.append(name)
        weight_count += layer.weight.numel()
        fp32_bytes += layer.weight.numel() * 4
        q8_bytes += quantized.weight_storage_nbytes

    return ModelQuantizationReport(
        layer_names=names,
        weight_count=weight_count,
        fp32_weight_bytes=fp32_bytes,
        quantized_weight_bytes=q8_bytes,
    )


def _mse(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    return (candidate.float() - reference.float()).square().mean().item()


def main() -> None:
    print("Q8_0 — 32-value block, exact byte accounting\n")

    # 12.7 makes the unrounded FP32 scale exactly 0.1 in decimal notation.
    # The stored FP16 scale is close to, but not exactly, 0.1.
    visible = [-12.7, -6.35, -3.2, -1.05, 0.0, 0.14, 1.25, 4.4, 8.88, 12.7]
    x = torch.tensor(visible + [0.0] * (QK8_0 - len(visible)))
    qx = quantize_q8_0(x)
    x_hat = qx.dequantize()

    print("1) One block calculated step by step")
    print(f"   amax                    = {x.abs().max().item():.6f}")
    print(f"   FP32 scale amax/127     = {x.abs().max().item()/127:.9f}")
    print(f"   stored FP16 scale       = {qx.d[0].item():.9f}")
    print("   formula                 = q=round(x/d), x_hat=q*FP16(d)")
    print("\n   index       x      q       x_hat       error")
    for index in range(len(visible)):
        error = x_hat[index].item() - x[index].item()
        print(
            f"   {index:>5} {x[index].item():>8.3f} "
            f"{int(qx.qs[0, index]):>6} {x_hat[index].item():>12.6f} "
            f"{error:>11.6f}"
        )
    print(f"\n   {q8_0_layout_report(qx)}")
    print(f"   block MSE = {_mse(x, x_hat):.9f}")

    print("\n2) Why block-wise scale matters")
    small = torch.linspace(-0.25, 0.25, QK8_0)
    large = torch.linspace(-20.0, 20.0, QK8_0)
    two_blocks = torch.cat((small, large))
    block_q8 = quantize_q8_0(two_blocks)
    global_i8 = quantize_tensor(
        two_blocks,
        bits=8,
        group_size=64,
        symmetric=True,
        scale_dtype=torch.float16,
    )
    block_hat = block_q8.dequantize()
    global_hat = global_i8.dequantize()
    print(
        "   Q8_0 scales             = "
        f"{[round(v, 7) for v in block_q8.d.float().tolist()]}"
    )
    print(f"   one global INT8 scale   = {global_i8.scales.item():.7f}")
    print(
        f"   small-block MSE, Q8_0   = {_mse(small, block_hat[:QK8_0]):.9f}"
    )
    print(
        f"   small-block MSE, global = {_mse(small, global_hat[:QK8_0]):.9f}"
    )
    print(f"   total MSE, Q8_0         = {_mse(two_blocks, block_hat):.9f}")
    print(f"   total MSE, global INT8  = {_mse(two_blocks, global_hat):.9f}")
    print(
        "   trade-off: Q8_0 uses one extra FP16 scale here "
        "(68 B instead of 66 B),\n"
        "   but the small block no longer shares the 20.0 outlier's step size."
    )

    print("\n3) A real Linear with no floating-point weight copy")
    torch.manual_seed(8)
    float_layer = nn.Linear(64, 16, bias=True)
    q8_layer = Q8_0Linear.from_float(float_layer)
    sample = torch.randn(4, 64)
    float_output = float_layer(sample)
    q8_output = q8_layer(sample)
    print(f"   has `weight` attribute  = {hasattr(q8_layer, 'weight')}")
    print(f"   qweight dtype/shape     = {q8_layer.qweight.dtype}, {tuple(q8_layer.qweight.shape)}")
    print(f"   scale dtype/shape       = {q8_layer.scales.dtype}, {tuple(q8_layer.scales.shape)}")
    print(f"   FP32 weight bytes       = {float_layer.weight.numel() * 4}")
    print(f"   Q8_0 weight bytes       = {q8_layer.weight_storage_nbytes}")
    print(f"   output MSE              = {_mse(float_output, q8_output):.9f}")

    print(
        "\nInterpretation: Q8_0 is an 8-bit payload plus block metadata. "
        "It is usually\n"
        "a high-quality, larger baseline—not a promise of losslessness, "
        "and not the\n"
        "same thing as W8A8 activation quantization."
    )

    # Executable lesson invariants.
    assert qx.storage_nbytes == 34
    assert qx.effective_bits_per_weight == 8.5
    assert qx.qs.dtype == torch.int8 and qx.d.dtype == torch.float16
    assert _mse(small, block_hat[:QK8_0]) < _mse(small, global_hat[:QK8_0])
    assert not hasattr(q8_layer, "weight")
    assert torch.isfinite(q8_output).all()


if __name__ == "__main__":
    main()
