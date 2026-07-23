"""core.py — quantization primitives small enough to read top to bottom.

This file implements the numerical core used by every example in this folder:

  * symmetric and asymmetric affine quantization;
  * per-row, group-wise scales;
  * real INT4 nibble packing (two values in one uint8 byte);
  * INT8 storage;
  * dequantization back to floating point;
  * a straight-through-estimator (STE) fake-quantizer for QAT.

The code is intentionally plain PyTorch. It performs REAL low-bit storage, but
it does not provide a fused integer matrix-multiplication kernel. `linear.py`
therefore unpacks/dequantizes before calling PyTorch's regular floating-point
linear operation. That separation is pedagogically important:

    storage format != compute kernel != model-level quantization recipe

Run `python quantization/by_hand.py` for pencil-friendly numbers and
`python quantization/demo.py` for the TinyQwen model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Small validation and accounting helpers
# ---------------------------------------------------------------------------


def _check_bits(bits: int) -> None:
    """This teaching implementation supports the two most useful cases."""
    if bits not in (4, 8):
        raise ValueError(f"bits must be 4 or 8, got {bits}")


def tensor_nbytes(tensor: torch.Tensor | None) -> int:
    """Exact bytes occupied by one dense PyTorch tensor (zero for None)."""
    if tensor is None:
        return 0
    return tensor.numel() * tensor.element_size()


def _round_half_away_from_zero(x: torch.Tensor) -> torch.Tensor:
    """Round halves away from zero instead of PyTorch's ties-to-even rule.

    Quantization papers and kernels commonly describe `round()` without making
    tie behavior the lesson. This explicit rule makes examples predictable:
    +2.5 -> +3 and -2.5 -> -3.
    """
    return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)


# ---------------------------------------------------------------------------
# 4-bit packing
# ---------------------------------------------------------------------------


def pack_int4(values: torch.Tensor) -> torch.Tensor:
    """Pack signed INT4 values into uint8 bytes.

    The last dimension must be even. Element 0 goes in the low nibble and
    element 1 in the high nibble:

        byte = (value_1 & 0x0F) << 4 | (value_0 & 0x0F)

    Signed values use two's-complement nibble representation, so -1 is 0xF,
    -7 is 0x9, and +7 is 0x7. Two 4-bit values now occupy one real byte.
    """
    if values.shape[-1] % 2:
        raise ValueError("INT4 packing needs an even last dimension")
    if values.numel():
        lo, hi = int(values.min().item()), int(values.max().item())
        if lo < -8 or hi > 7:
            raise ValueError(f"signed INT4 range is [-8, 7], got [{lo}, {hi}]")

    # int16 prevents shifting/bitwise surprises; the final byte is uint8.
    nibbles = torch.bitwise_and(values.to(torch.int16), 0x0F)
    low = nibbles[..., 0::2]
    high = torch.bitwise_left_shift(nibbles[..., 1::2], 4)
    return torch.bitwise_or(low, high).to(torch.uint8).contiguous()


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Reverse `pack_int4`, returning signed int8 values."""
    if packed.dtype != torch.uint8:
        raise TypeError(f"packed INT4 data must be uint8, got {packed.dtype}")

    low = torch.bitwise_and(packed, 0x0F).to(torch.int16)
    high = torch.bitwise_right_shift(packed, 4).to(torch.int16)
    nibbles = torch.stack((low, high), dim=-1).flatten(-2)

    # Convert the unsigned nibble 8..15 to signed -8..-1.
    signed = torch.where(nibbles >= 8, nibbles - 16, nibbles)
    return signed.to(torch.int8)


def pack_uint4(values: torch.Tensor) -> torch.Tensor:
    """Pack unsigned 0..15 values, two per uint8 byte."""
    if values.shape[-1] % 2:
        raise ValueError("UINT4 packing needs an even last dimension")
    if values.numel():
        lo, hi = int(values.min().item()), int(values.max().item())
        if lo < 0 or hi > 15:
            raise ValueError(f"unsigned INT4 range is [0, 15], got [{lo}, {hi}]")

    values = values.to(torch.uint8)
    low = values[..., 0::2]
    high = torch.bitwise_left_shift(values[..., 1::2], 4)
    return torch.bitwise_or(low, high).contiguous()


def unpack_uint4(packed: torch.Tensor) -> torch.Tensor:
    """Reverse `pack_uint4`, returning uint8 values in 0..15."""
    if packed.dtype != torch.uint8:
        raise TypeError(f"packed UINT4 data must be uint8, got {packed.dtype}")
    low = torch.bitwise_and(packed, 0x0F)
    high = torch.bitwise_right_shift(packed, 4)
    return torch.stack((low, high), dim=-1).flatten(-2)


# ---------------------------------------------------------------------------
# QuantizedTensor: low-bit payload + the metadata needed to decode it
# ---------------------------------------------------------------------------


@dataclass
class QuantizedTensor:
    """A group-wise affine-quantized tensor with no floating-point weight copy.

    `data` is packed uint8 for 4-bit, int8/uint8 for 8-bit.
    `scales` has one value per row and group.
    `zero_points` is omitted for symmetric quantization because it is always 0.
    """

    data: torch.Tensor
    scales: torch.Tensor
    zero_points: torch.Tensor | None
    original_shape: tuple[int, ...]
    padded_last_dim: int
    bits: int
    group_size: int
    symmetric: bool
    packed: bool

    @property
    def numel(self) -> int:
        """Number of logical, unpadded values represented."""
        return math.prod(self.original_shape)

    @property
    def payload_nbytes(self) -> int:
        """Bytes containing integer codes only."""
        return tensor_nbytes(self.data)

    @property
    def metadata_nbytes(self) -> int:
        """Bytes containing scales and optional zero-points."""
        return tensor_nbytes(self.scales) + tensor_nbytes(self.zero_points)

    @property
    def storage_nbytes(self) -> int:
        """Total tensor storage: integer payload + decoding metadata."""
        return self.payload_nbytes + self.metadata_nbytes

    @property
    def effective_bits_per_weight(self) -> float:
        """Actual stored bits divided by logical (unpadded) values."""
        return 8.0 * self.storage_nbytes / self.numel

    def integer_values(self) -> torch.Tensor:
        """Return unpacked integer codes, including right-padding."""
        if self.bits == 4 and self.packed:
            if self.symmetric:
                return unpack_int4(self.data)
            return unpack_uint4(self.data)
        return self.data

    def dequantize(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Reconstruct the approximate floating-point tensor."""
        q = self.integer_values().to(torch.float32)
        rows = math.prod(self.original_shape[:-1]) if len(self.original_shape) > 1 else 1
        n_groups = self.padded_last_dim // self.group_size

        q = q.reshape(rows, n_groups, self.group_size)
        scales = self.scales.to(torch.float32).reshape(rows, n_groups, 1)
        if self.zero_points is None:
            zero_points = 0.0
        else:
            zero_points = self.zero_points.to(torch.float32).reshape(rows, n_groups, 1)

        restored = (q - zero_points) * scales
        restored = restored.reshape(rows, self.padded_last_dim)
        restored = restored[:, : self.original_shape[-1]]
        return restored.reshape(self.original_shape).to(dtype)


# ---------------------------------------------------------------------------
# Affine quantization
# ---------------------------------------------------------------------------


def quantize_tensor(
    tensor: torch.Tensor,
    *,
    bits: int = 4,
    group_size: int = 32,
    symmetric: bool = True,
    pack: bool = True,
    scale_dtype: torch.dtype = torch.float32,
) -> QuantizedTensor:
    """Group-wise affine-quantize `tensor` along its last dimension.

    Every leading-dimension row is split into independent groups. For a weight
    matrix `[out_features, in_features]`, this means each output row receives
    `ceil(in_features / group_size)` scales.

    Symmetric:
        qmax  = 2^(bits-1)-1
        scale = max(abs(group)) / qmax
        q     = clamp(round(x/scale), -qmax, qmax)

    Asymmetric:
        qmin, qmax = 0, 2^bits-1
        scale      = (max(group)-min(group)) / (qmax-qmin)
        zero_point = round(qmin-min(group)/scale)
        q          = clamp(round(x/scale)+zero_point, qmin, qmax)

    Right-padding is used only to complete the final group. Padded codes decode
    to zero and are removed by `dequantize()`.
    """
    _check_bits(bits)
    if not tensor.is_floating_point():
        raise TypeError(f"expected a floating-point tensor, got {tensor.dtype}")
    if tensor.ndim < 1 or tensor.numel() == 0:
        raise ValueError("tensor must have at least one non-empty dimension")
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")

    original_shape = tuple(tensor.shape)
    last_dim = original_shape[-1]
    rows = math.prod(original_shape[:-1]) if len(original_shape) > 1 else 1
    n_groups = math.ceil(last_dim / group_size)
    padded_last_dim = n_groups * group_size

    # Quantization statistics are normally accumulated in FP32, even if the
    # source weight is FP16/BF16.
    flat = tensor.detach().to(torch.float32).reshape(rows, last_dim)
    if padded_last_dim != last_dim:
        flat = F.pad(flat, (0, padded_last_dim - last_dim))
    groups = flat.reshape(rows, n_groups, group_size)

    # Mark padding so asymmetric min/max statistics ignore artificial zeros.
    valid_flat = torch.arange(padded_last_dim, device=tensor.device) < last_dim
    valid = valid_flat.reshape(1, n_groups, group_size)

    if symmetric:
        qmax = 2 ** (bits - 1) - 1
        qmin = -qmax
        max_abs = groups.abs().masked_fill(~valid, 0.0).amax(dim=-1)
        scales = max_abs / qmax
        scales = torch.where(scales == 0, torch.ones_like(scales), scales)
        zero_points = None
        q = _round_half_away_from_zero(groups / scales.unsqueeze(-1))
        q = q.clamp(qmin, qmax)
        # Zero is the correct padded code in symmetric quantization.
        q = q.masked_fill(~valid, 0)
    else:
        qmin, qmax = 0, 2**bits - 1
        mins = groups.masked_fill(~valid, float("inf")).amin(dim=-1)
        maxs = groups.masked_fill(~valid, float("-inf")).amax(dim=-1)
        # Make real zero exactly representable. Without this adjustment an
        # all-positive range could require a negative zero-point, or an
        # all-negative range a zero-point above qmax.
        mins = torch.minimum(mins, torch.zeros_like(mins))
        maxs = torch.maximum(maxs, torch.zeros_like(maxs))
        scales = (maxs - mins) / (qmax - qmin)
        scales = torch.where(scales == 0, torch.ones_like(scales), scales)
        zero_points = _round_half_away_from_zero(qmin - mins / scales)
        zero_points = zero_points.clamp(qmin, qmax)
        q = _round_half_away_from_zero(
            groups / scales.unsqueeze(-1) + zero_points.unsqueeze(-1)
        )
        q = q.clamp(qmin, qmax)
        # A padded real zero should decode back to zero (within zp rounding).
        padding_codes = zero_points.unsqueeze(-1).expand_as(q)
        q = torch.where(valid, q, padding_codes)

    q = q.reshape(rows, padded_last_dim)
    scales = scales.to(scale_dtype).contiguous()
    if zero_points is not None:
        # int16 is enough for UINT4/UINT8 zero-points and easy to inspect.
        zero_points = zero_points.to(torch.int16).contiguous()

    if bits == 4 and pack:
        data = pack_int4(q) if symmetric else pack_uint4(q)
    elif bits == 8:
        data = q.to(torch.int8 if symmetric else torch.uint8).contiguous()
    else:
        # Unpacked INT4 is useful for showing values and fake quantization.
        data = q.to(torch.int8 if symmetric else torch.uint8).contiguous()

    return QuantizedTensor(
        data=data,
        scales=scales,
        zero_points=zero_points,
        original_shape=original_shape,
        padded_last_dim=padded_last_dim,
        bits=bits,
        group_size=group_size,
        symmetric=symmetric,
        packed=(bits == 4 and pack),
    )


def quantization_error(
    original: torch.Tensor, quantized: QuantizedTensor
) -> dict[str, float]:
    """Return simple reconstruction-error metrics."""
    restored = quantized.dequantize(dtype=torch.float32)
    reference = original.detach().to(torch.float32)
    error = restored - reference
    return {
        "mse": error.square().mean().item(),
        "mae": error.abs().mean().item(),
        "max_abs": error.abs().max().item(),
        "signal_rms": reference.square().mean().sqrt().item(),
        "error_rms": error.square().mean().sqrt().item(),
    }


def fake_quantize(
    tensor: torch.Tensor,
    *,
    bits: int = 4,
    group_size: int = 32,
    symmetric: bool = True,
) -> torch.Tensor:
    """Quantize then dequantize, returning floating-point approximations.

    This is appropriate for evaluation, but its round/clamp operations do not
    provide useful gradients. Use `fake_quantize_ste` during QAT.
    """
    qt = quantize_tensor(
        tensor,
        bits=bits,
        group_size=group_size,
        symmetric=symmetric,
        pack=False,
    )
    return qt.dequantize(dtype=tensor.dtype)


def fake_quantize_ste(
    tensor: torch.Tensor,
    *,
    bits: int = 4,
    group_size: int = 32,
    symmetric: bool = True,
) -> torch.Tensor:
    """Fake-quantize in forward, use identity gradient in backward.

    `quantized` is detached numerical noise. The algebra

        tensor + (quantized - tensor).detach()

    returns `quantized` in the forward pass, while d(output)/d(tensor) = 1.
    This is the straight-through estimator (STE) used to make rounding
    trainable enough for a small QAT demonstration.
    """
    quantized = fake_quantize(
        tensor.detach(),
        bits=bits,
        group_size=group_size,
        symmetric=symmetric,
    )
    return tensor + (quantized - tensor).detach()


def format_quantized_tensor(name: str, qt: QuantizedTensor) -> str:
    """One-line storage report used by the examples."""
    fp32_bytes = qt.numel * 4
    ratio = fp32_bytes / qt.storage_nbytes
    return (
        f"{name}: shape={qt.original_shape}, {qt.bits}-bit, "
        f"group={qt.group_size}, payload={qt.payload_nbytes:,} B, "
        f"metadata={qt.metadata_nbytes:,} B, total={qt.storage_nbytes:,} B, "
        f"effective={qt.effective_bits_per_weight:.3f} bpw, "
        f"FP32/quant={ratio:.2f}x"
    )
