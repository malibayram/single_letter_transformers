"""q4_k.py — inspect the 256-value structure behind GGUF Q4_K.

Q4_K is more than "put every number in four bits". One super-block represents
256 weights:

    128 bytes  256 x 4-bit integer codes
     12 bytes  eight 6-bit scales + eight 6-bit minimum codes
      2 bytes  one FP16 super-block scale
      2 bytes  one FP16 super-block minimum scale
    ---------------------------------------------------------
    144 bytes  = 1152 bits / 256 weights = 4.5 bits per weight

This file implements that storage SHAPE and its scale/min packing so students
can unpack every byte. Its simple min/max parameter fitting is educational; it
does not reproduce llama.cpp's optimized reference quantizer byte-for-byte and
does not write GGUF files. llama.cpp also applies optional importance weighting
and model-level mixed recipes such as Q4_K_M.

In short:

  * representation/layout: structurally faithful and genuinely packed;
  * parameter search: intentionally simple;
  * GGUF compatibility / optimized kernels: not claimed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from core import _round_half_away_from_zero, tensor_nbytes


QK_K = 256
SUBBLOCK = 32
N_SUBBLOCKS = 8
K_SCALE_SIZE = 12


def pack_scale_min_6bit(
    scales: torch.Tensor, mins: torch.Tensor
) -> torch.Tensor:
    """Pack 8 scale codes + 8 minimum codes (all 0..63) into 12 bytes.

    This follows Q4_K's compact layout: codes 0..3 use six low bits directly;
    codes 4..7 are split between the high two bits of bytes 0..7 and nibbles
    in bytes 8..11.
    """
    if scales.shape != mins.shape or scales.shape[-1] != 8:
        raise ValueError("scales and mins must have matching [..., 8] shapes")
    if scales.numel():
        for name, values in (("scales", scales), ("mins", mins)):
            lo, hi = int(values.min().item()), int(values.max().item())
            if lo < 0 or hi > 63:
                raise ValueError(f"{name} must be 6-bit values, got [{lo}, {hi}]")

    s = scales.to(torch.uint8)
    m = mins.to(torch.uint8)
    out = torch.zeros(*s.shape[:-1], K_SCALE_SIZE, dtype=torch.uint8, device=s.device)

    # A-D and a-d: low six bits hold scale/min codes 0..3.
    out[..., 0:4] = s[..., 0:4]
    out[..., 4:8] = m[..., 0:4]

    # E-H: upper two bits live in bytes 0..3; lower four in bytes 8..11.
    out[..., 0:4] |= torch.bitwise_left_shift(s[..., 4:8] >> 4, 6)
    # e-h: upper two bits live in bytes 4..7; lower four share 8..11.
    out[..., 4:8] |= torch.bitwise_left_shift(m[..., 4:8] >> 4, 6)
    out[..., 8:12] = torch.bitwise_or(
        torch.bitwise_and(s[..., 4:8], 0x0F),
        torch.bitwise_left_shift(torch.bitwise_and(m[..., 4:8], 0x0F), 4),
    )
    return out.contiguous()


def unpack_scale_min_6bit(
    packed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reverse `pack_scale_min_6bit`."""
    if packed.dtype != torch.uint8 or packed.shape[-1] != K_SCALE_SIZE:
        raise ValueError("packed scale/min metadata must be uint8 [..., 12]")

    scales = torch.empty(*packed.shape[:-1], 8, dtype=torch.uint8, device=packed.device)
    mins = torch.empty_like(scales)
    scales[..., 0:4] = torch.bitwise_and(packed[..., 0:4], 0x3F)
    mins[..., 0:4] = torch.bitwise_and(packed[..., 4:8], 0x3F)
    scales[..., 4:8] = torch.bitwise_or(
        torch.bitwise_and(packed[..., 8:12], 0x0F),
        torch.bitwise_left_shift(packed[..., 0:4] >> 6, 4),
    )
    mins[..., 4:8] = torch.bitwise_or(
        packed[..., 8:12] >> 4,
        torch.bitwise_left_shift(packed[..., 4:8] >> 6, 4),
    )
    return scales, mins


def pack_q4_k_codes(codes: torch.Tensor) -> torch.Tensor:
    """Pack `[blocks, 8, 32]` UINT4 codes into `[blocks, 128]` bytes.

    Within every pair of 32-value sub-blocks, the first sub-block occupies low
    nibbles and the second occupies high nibbles. This mirrors how Q4_K's
    dequantizer reads groups of 64 values.
    """
    if codes.ndim != 3 or codes.shape[1:] != (N_SUBBLOCKS, SUBBLOCK):
        raise ValueError("Q4_K codes must have shape [blocks, 8, 32]")
    if codes.numel():
        lo, hi = int(codes.min().item()), int(codes.max().item())
        if lo < 0 or hi > 15:
            raise ValueError(f"Q4 codes must be in [0, 15], got [{lo}, {hi}]")

    codes = codes.to(torch.uint8)
    chunks = []
    for pair in range(4):
        low = codes[:, 2 * pair, :]
        high = torch.bitwise_left_shift(codes[:, 2 * pair + 1, :], 4)
        chunks.append(torch.bitwise_or(low, high))
    return torch.cat(chunks, dim=1).contiguous()


def unpack_q4_k_codes(packed: torch.Tensor) -> torch.Tensor:
    """Reverse `pack_q4_k_codes`, returning `[blocks, 8, 32]`."""
    if packed.dtype != torch.uint8 or packed.ndim != 2 or packed.shape[1] != 128:
        raise ValueError("packed Q4_K codes must be uint8 [blocks, 128]")
    subblocks = []
    for pair in range(4):
        chunk = packed[:, pair * 32 : (pair + 1) * 32]
        subblocks.append(torch.bitwise_and(chunk, 0x0F))
        subblocks.append(chunk >> 4)
    return torch.stack(subblocks, dim=1)


@dataclass
class Q4KTensor:
    """Packed educational Q4_K tensor."""

    qs: torch.Tensor
    scales_and_mins: torch.Tensor
    d: torch.Tensor
    dmin: torch.Tensor
    original_shape: tuple[int, ...]
    padded_numel: int

    @property
    def numel(self) -> int:
        return math.prod(self.original_shape)

    @property
    def n_superblocks(self) -> int:
        return self.padded_numel // QK_K

    @property
    def storage_nbytes(self) -> int:
        return (
            tensor_nbytes(self.qs)
            + tensor_nbytes(self.scales_and_mins)
            + tensor_nbytes(self.d)
            + tensor_nbytes(self.dmin)
        )

    @property
    def effective_bits_per_weight(self) -> float:
        return 8.0 * self.storage_nbytes / self.numel

    def dequantize(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Decode `x_hat = d*scale_code*q - dmin*min_code`."""
        q = unpack_q4_k_codes(self.qs).to(torch.float32)
        scale_codes, min_codes = unpack_scale_min_6bit(self.scales_and_mins)
        local_scales = self.d.float().unsqueeze(1) * scale_codes.float()
        local_mins = self.dmin.float().unsqueeze(1) * min_codes.float()
        restored = q * local_scales.unsqueeze(-1) - local_mins.unsqueeze(-1)
        restored = restored.reshape(-1)[: self.numel]
        return restored.reshape(self.original_shape).to(dtype)


def quantize_q4_k(tensor: torch.Tensor) -> Q4KTensor:
    """Quantize to the educational Q4_K representation.

    Local fitting uses:

        minimum_term = max(0, -min(x))
        local_scale  = (max(x) + minimum_term) / 15

    Then eight local scales/minimums are themselves quantized to six bits using
    one FP16 `d` and `dmin` per 256-value super-block.
    """
    if not tensor.is_floating_point() or tensor.numel() == 0:
        raise ValueError("expected a non-empty floating-point tensor")

    original_shape = tuple(tensor.shape)
    flat = tensor.detach().float().flatten()
    n_superblocks = math.ceil(flat.numel() / QK_K)
    padded_numel = n_superblocks * QK_K
    if padded_numel != flat.numel():
        flat = F.pad(flat, (0, padded_numel - flat.numel()))
    blocks = flat.reshape(n_superblocks, N_SUBBLOCKS, SUBBLOCK)

    local_min = blocks.amin(dim=-1)
    local_max = blocks.amax(dim=-1)
    minimum_term = (-local_min).clamp_min(0.0)
    local_scale = (local_max + minimum_term).clamp_min(0.0) / 15.0

    # Quantize the eight local parameters to six-bit codes. Store the global
    # multipliers as FP16 exactly as the Q4_K size calculation assumes.
    d_float = local_scale.amax(dim=-1) / 63.0
    dmin_float = minimum_term.amax(dim=-1) / 63.0
    d = d_float.to(torch.float16)
    dmin = dmin_float.to(torch.float16)

    # Use the STORED FP16 values to choose codes, so dequantization matches.
    d_used = d.float().unsqueeze(1)
    dmin_used = dmin.float().unsqueeze(1)
    scale_codes = torch.where(
        d_used > 0,
        _round_half_away_from_zero(local_scale / d_used),
        torch.zeros_like(local_scale),
    ).clamp(0, 63)
    min_codes = torch.where(
        dmin_used > 0,
        _round_half_away_from_zero(minimum_term / dmin_used),
        torch.zeros_like(minimum_term),
    ).clamp(0, 63)
    scales_and_mins = pack_scale_min_6bit(scale_codes, min_codes)

    # Reconstruct quantized local params before choosing 4-bit weight codes.
    represented_scale = d_used * scale_codes
    represented_min = dmin_used * min_codes
    q = torch.where(
        represented_scale.unsqueeze(-1) > 0,
        _round_half_away_from_zero(
            (blocks + represented_min.unsqueeze(-1))
            / represented_scale.unsqueeze(-1)
        ),
        torch.zeros_like(blocks),
    ).clamp(0, 15)
    qs = pack_q4_k_codes(q)

    return Q4KTensor(
        qs=qs,
        scales_and_mins=scales_and_mins,
        d=d,
        dmin=dmin,
        original_shape=original_shape,
        padded_numel=padded_numel,
    )


def q4_k_layout_report(qt: Q4KTensor) -> str:
    """Human-readable byte accounting."""
    return (
        f"Q4_K-like: {qt.n_superblocks} super-block(s), "
        f"codes={tensor_nbytes(qt.qs)} B, "
        f"6-bit scale/min metadata={tensor_nbytes(qt.scales_and_mins)} B, "
        f"d+dmin={tensor_nbytes(qt.d)+tensor_nbytes(qt.dmin)} B, "
        f"total={qt.storage_nbytes} B, "
        f"effective={qt.effective_bits_per_weight:.3f} bpw"
    )
