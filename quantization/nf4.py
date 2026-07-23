"""nf4.py — a readable NormalFloat4 (NF4) block quantizer.

Uniform INT4 gives equally spaced levels. NF4 instead places 16 levels according
to normal-distribution quantiles, spending more codes near zero where normally
distributed neural-network weights are dense. QLoRA uses NF4 for the frozen
base model and additionally "double quantizes" scale constants.

This file implements:
  * the standard 16-value NF4 codebook;
  * one absmax scale per block;
  * nearest-codebook assignment;
  * real 4-bit packing of codebook indices;
  * dequantization.

It intentionally does not implement QLoRA's double-quantized scale hierarchy or
CUDA kernels. With one FP16 scale per 64 weights, its storage is:

    4 payload bits + 16/64 scale bits = 4.25 bits per weight.

Run directly for a normal-vs-uniform distribution comparison:
    PYTHONPATH=quantization .venv/bin/python quantization/nf4.py
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from core import pack_uint4, tensor_nbytes, unpack_uint4


# Values used by the common NF4 implementation. They are asymmetric around
# zero because the construction reserves an exact representation for zero.
NF4_CODEBOOK = torch.tensor(
    [
        -1.0000000,
        -0.6961928,
        -0.5250731,
        -0.3949175,
        -0.2844414,
        -0.1847734,
        -0.0910500,
         0.0000000,
         0.0795803,
         0.1609302,
         0.2461123,
         0.3379152,
         0.4407098,
         0.5626170,
         0.7229568,
         1.0000000,
    ],
    dtype=torch.float32,
)
NF4_ZERO_INDEX = 7


@dataclass
class NF4Tensor:
    packed_indices: torch.Tensor
    absmax: torch.Tensor
    original_shape: tuple[int, ...]
    block_size: int
    padded_numel: int

    @property
    def numel(self) -> int:
        return math.prod(self.original_shape)

    @property
    def storage_nbytes(self) -> int:
        return tensor_nbytes(self.packed_indices) + tensor_nbytes(self.absmax)

    @property
    def effective_bits_per_weight(self) -> float:
        return 8.0 * self.storage_nbytes / self.numel

    def dequantize(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        indices = unpack_uint4(self.packed_indices).reshape(-1)[: self.padded_numel]
        codebook = NF4_CODEBOOK.to(indices.device)
        normalized = codebook[indices.long()].reshape(-1, self.block_size)
        restored = normalized * self.absmax.float().unsqueeze(1)
        restored = restored.flatten()[: self.numel]
        return restored.reshape(self.original_shape).to(dtype)


def quantize_nf4(
    tensor: torch.Tensor,
    *,
    block_size: int = 64,
    scale_dtype: torch.dtype = torch.float16,
) -> NF4Tensor:
    """Block-wise NF4 quantization with nearest-codebook lookup."""
    if not tensor.is_floating_point() or tensor.numel() == 0:
        raise ValueError("expected a non-empty floating-point tensor")
    if block_size <= 0 or block_size % 2:
        raise ValueError("block_size must be a positive even number")

    original_shape = tuple(tensor.shape)
    flat = tensor.detach().float().flatten()
    n_blocks = math.ceil(flat.numel() / block_size)
    padded_numel = n_blocks * block_size
    if padded_numel != flat.numel():
        flat = F.pad(flat, (0, padded_numel - flat.numel()))
    blocks = flat.reshape(n_blocks, block_size)

    absmax = blocks.abs().amax(dim=1)
    safe_absmax = torch.where(absmax == 0, torch.ones_like(absmax), absmax)
    normalized = blocks / safe_absmax.unsqueeze(1)

    codebook = NF4_CODEBOOK.to(tensor.device)
    distances = (normalized.unsqueeze(-1) - codebook).abs()
    indices = distances.argmin(dim=-1).to(torch.uint8)

    # Padding represents real zero and should use NF4's exact zero code.
    if padded_numel != tensor.numel():
        indices.flatten()[tensor.numel() :] = NF4_ZERO_INDEX

    return NF4Tensor(
        packed_indices=pack_uint4(indices.reshape(1, -1)),
        absmax=absmax.to(scale_dtype).contiguous(),
        original_shape=original_shape,
        block_size=block_size,
        padded_numel=padded_numel,
    )


def main() -> None:
    """Show why a distribution-matched codebook matters."""
    from core import quantization_error, quantize_tensor

    torch.manual_seed(0)
    distributions = {
        "normal N(0,1)": torch.randn(4096),
        "uniform [-1,1]": torch.rand(4096) * 2 - 1,
    }
    print("Uniform INT4 vs NF4 (same block_size=64)")
    print(f"  {'distribution':<18} {'uniform MSE':>13} {'NF4 MSE':>13} {'winner':>10}")
    for name, values in distributions.items():
        uniform = quantize_tensor(
            values,
            bits=4,
            group_size=64,
            symmetric=True,
            scale_dtype=torch.float16,
        )
        nf4 = quantize_nf4(values, block_size=64)
        uniform_mse = quantization_error(values, uniform)["mse"]
        nf4_mse = (nf4.dequantize() - values).square().mean().item()
        winner = "NF4" if nf4_mse < uniform_mse else "uniform"
        print(f"  {name:<18} {uniform_mse:>13.8f} {nf4_mse:>13.8f} {winner:>10}")

    nf4_normal = quantize_nf4(distributions["normal N(0,1)"])
    print(
        f"\nNF4 storage: {nf4_normal.storage_nbytes} bytes for "
        f"{nf4_normal.numel} weights = "
        f"{nf4_normal.effective_bits_per_weight:.3f} bpw"
    )
    print(
        "NF4 wins on the normal sample, but uniform levels win on uniform data.\n"
        "The codebook is a distribution assumption, not universal magic."
    )

    assert nf4_normal.packed_indices.dtype == torch.uint8
    assert nf4_normal.packed_indices.numel() * 2 == nf4_normal.padded_numel


if __name__ == "__main__":
    main()
