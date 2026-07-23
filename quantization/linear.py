"""linear.py — use the low-bit tensors from `core.py` inside neural networks.

There are two deliberately different layers:

  QuantizedLinear
      Stores no floating-point weight. Its weight is genuinely INT8 or packed
      INT4 plus scales. For portability and readability it dequantizes in
      `forward()` and then calls PyTorch's floating-point `F.linear`.
      Result: real storage compression, correct outputs, NO speedup claim.

  QATLinear
      Keeps a trainable FP32 weight but fake-quantizes it during every forward
      pass with a straight-through estimator. This is the training form.
      `to_quantized()` converts the trained layer to QuantizedLinear.

The helpers at the bottom walk any PyTorch model and replace its `nn.Linear`
modules, much like `lora/inject.py` replaces Linear layers with adapters.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from core import (
    QuantizedTensor,
    fake_quantize_ste,
    format_quantized_tensor,
    quantize_tensor,
    tensor_nbytes,
)


class QuantizedLinear(nn.Module):
    """An inference-only Linear with INT8 or packed INT4 weight storage."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bits: int,
        group_size: int,
        symmetric: bool,
        original_shape: tuple[int, int],
        padded_last_dim: int,
        packed: bool,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor | None,
        bias: torch.Tensor | None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        self.symmetric = symmetric
        self.original_shape = original_shape
        self.padded_last_dim = padded_last_dim
        self.packed = packed

        # Buffers are saved in state_dict and move with .to(device), but unlike
        # Parameters they are not trainable. There is no FP weight Parameter.
        self.register_buffer("qweight", qweight)
        self.register_buffer("scales", scales)
        self.register_buffer("zero_points", zero_points)
        self.register_buffer("bias", bias)

    @classmethod
    def from_float(
        cls,
        layer: nn.Linear,
        *,
        bits: int = 4,
        group_size: int = 32,
        symmetric: bool = True,
        scale_dtype: torch.dtype = torch.float16,
    ) -> "QuantizedLinear":
        """Quantize a trained `nn.Linear` and discard its float weight."""
        qt = quantize_tensor(
            layer.weight,
            bits=bits,
            group_size=group_size,
            symmetric=symmetric,
            pack=True,
            scale_dtype=scale_dtype,
        )
        bias = None if layer.bias is None else layer.bias.detach().clone()
        return cls.from_quantized_tensor(qt, bias=bias)

    @classmethod
    def from_quantized_tensor(
        cls, qt: QuantizedTensor, *, bias: torch.Tensor | None = None
    ) -> "QuantizedLinear":
        """Build directly from a quantized `[out, in]` weight tensor."""
        if len(qt.original_shape) != 2:
            raise ValueError(
                f"QuantizedLinear weight must be 2D, got {qt.original_shape}"
            )
        out_features, in_features = qt.original_shape
        if bias is not None and tuple(bias.shape) != (out_features,):
            raise ValueError(
                f"bias must have shape {(out_features,)}, got {tuple(bias.shape)}"
            )
        return cls(
            in_features,
            out_features,
            bits=qt.bits,
            group_size=qt.group_size,
            symmetric=qt.symmetric,
            original_shape=qt.original_shape,
            padded_last_dim=qt.padded_last_dim,
            packed=qt.packed,
            qweight=qt.data,
            scales=qt.scales,
            zero_points=qt.zero_points,
            bias=bias,
        )

    def as_quantized_tensor(self) -> QuantizedTensor:
        """Expose the buffers through the reusable QuantizedTensor API."""
        return QuantizedTensor(
            data=self.qweight,
            scales=self.scales,
            zero_points=self.zero_points,
            original_shape=self.original_shape,
            padded_last_dim=self.padded_last_dim,
            bits=self.bits,
            group_size=self.group_size,
            symmetric=self.symmetric,
            packed=self.packed,
        )

    def dequantize_weight(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Rebuild the approximate `[out, in]` floating-point weight."""
        return self.as_quantized_tensor().dequantize(dtype=dtype)

    @property
    def weight_storage_nbytes(self) -> int:
        """Packed weight payload plus scale/zero-point metadata."""
        return self.as_quantized_tensor().storage_nbytes

    @property
    def total_storage_nbytes(self) -> int:
        """Weight storage plus optional unquantized bias."""
        return self.weight_storage_nbytes + tensor_nbytes(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Portable reference path: unpack -> dequantize -> float matmul.

        Production runtimes fuse these stages in optimized kernels. This
        readable implementation chooses transparency over speed.
        """
        weight = self.dequantize_weight(dtype=x.dtype)
        bias = None if self.bias is None else self.bias.to(dtype=x.dtype)
        return F.linear(x, weight, bias)

    def extra_repr(self) -> str:
        mode = "symmetric" if self.symmetric else "asymmetric"
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits={self.bits}, group_size={self.group_size}, {mode}, "
            f"weight_storage={self.weight_storage_nbytes} B"
        )


class QATLinear(nn.Module):
    """Trainable Linear that simulates low-bit weight numerics in forward."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        bits: int = 4,
        group_size: int = 32,
        symmetric: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        self.symmetric = symmetric

        # QAT still trains full-precision "master" weights. Quantization is
        # simulated in forward; real low-bit storage appears only at convert.
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Match nn.Linear's standard initialization."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @classmethod
    def from_float(
        cls,
        layer: nn.Linear,
        *,
        bits: int = 4,
        group_size: int = 32,
        symmetric: bool = True,
    ) -> "QATLinear":
        """Clone one trained float layer into its QAT form."""
        qat = cls(
            layer.in_features,
            layer.out_features,
            bias=layer.bias is not None,
            bits=bits,
            group_size=group_size,
            symmetric=symmetric,
        )
        with torch.no_grad():
            qat.weight.copy_(layer.weight)
            if layer.bias is not None:
                qat.bias.copy_(layer.bias)
        return qat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward sees rounded/clipped weights; backward sees identity gradient
        # through fake_quantize_ste and can adapt the FP master weight.
        quantized_weight = fake_quantize_ste(
            self.weight,
            bits=self.bits,
            group_size=self.group_size,
            symmetric=self.symmetric,
        )
        return F.linear(x, quantized_weight, self.bias)

    def to_quantized(
        self, *, scale_dtype: torch.dtype = torch.float16
    ) -> QuantizedLinear:
        """Convert the trained FP master weight to real low-bit storage."""
        float_layer = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        with torch.no_grad():
            float_layer.weight.copy_(self.weight)
            if self.bias is not None:
                float_layer.bias.copy_(self.bias)
        return QuantizedLinear.from_float(
            float_layer,
            bits=self.bits,
            group_size=self.group_size,
            symmetric=self.symmetric,
            scale_dtype=scale_dtype,
        )


# ---------------------------------------------------------------------------
# Model-tree replacement helpers
# ---------------------------------------------------------------------------


def _replace_module(
    model: nn.Module, qualified_name: str, new_module: nn.Module
) -> None:
    """Replace `a.b.c` inside a module tree, including ModuleList indices."""
    *parents, leaf = qualified_name.split(".")
    obj = model
    for part in parents:
        obj = getattr(obj, part)
    setattr(obj, leaf, new_module)


def _selected_linear_names(
    model: nn.Module,
    *,
    include: tuple[str, ...] | str,
    exclude: tuple[str, ...],
) -> list[tuple[str, nn.Linear]]:
    """Resolve target names before mutating the module tree."""
    selected = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.split(".")[-1]
        if name in exclude or leaf in exclude:
            continue
        if include == "all" or name in include or leaf in include:
            selected.append((name, module))
    return selected


@dataclass
class ModelQuantizationReport:
    """Accounting returned after replacing a model's Linear layers."""

    layer_names: list[str]
    weight_count: int
    fp32_weight_bytes: int
    quantized_weight_bytes: int

    @property
    def compression_ratio(self) -> float:
        return self.fp32_weight_bytes / self.quantized_weight_bytes

    @property
    def effective_bits_per_weight(self) -> float:
        return 8.0 * self.quantized_weight_bytes / self.weight_count

    def __str__(self) -> str:
        return (
            f"{len(self.layer_names)} layers, {self.weight_count:,} weights: "
            f"FP32 {self.fp32_weight_bytes:,} B -> quantized "
            f"{self.quantized_weight_bytes:,} B "
            f"({self.compression_ratio:.2f}x, "
            f"{self.effective_bits_per_weight:.3f} bpw)"
        )


def quantize_model_linears(
    model: nn.Module,
    *,
    bits: int = 4,
    group_size: int = 32,
    symmetric: bool = True,
    include: tuple[str, ...] | str = "all",
    exclude: tuple[str, ...] = ("lm_head",),
    scale_dtype: torch.dtype = torch.float16,
) -> ModelQuantizationReport:
    """Replace selected `nn.Linear`s with inference-only QuantizedLinear.

    `lm_head` is excluded by default because TinyQwen ties it to the embedding
    weight. Replacing only one end would silently break that sharing.
    """
    selected = _selected_linear_names(
        model, include=include, exclude=exclude
    )
    if not selected:
        raise ValueError("no nn.Linear layer matched include/exclude settings")

    names, weight_count, fp32_bytes, quantized_bytes = [], 0, 0, 0
    for name, layer in selected:
        quantized = QuantizedLinear.from_float(
            layer,
            bits=bits,
            group_size=group_size,
            symmetric=symmetric,
            scale_dtype=scale_dtype,
        )
        _replace_module(model, name, quantized)
        names.append(name)
        weight_count += layer.weight.numel()
        fp32_bytes += layer.weight.numel() * 4
        quantized_bytes += quantized.weight_storage_nbytes

    return ModelQuantizationReport(
        layer_names=names,
        weight_count=weight_count,
        fp32_weight_bytes=fp32_bytes,
        quantized_weight_bytes=quantized_bytes,
    )


def prepare_model_for_qat(
    model: nn.Module,
    *,
    bits: int = 4,
    group_size: int = 32,
    symmetric: bool = True,
    include: tuple[str, ...] | str = "all",
    exclude: tuple[str, ...] = ("lm_head",),
) -> list[str]:
    """Replace selected float Linears with trainable QATLinear modules."""
    selected = _selected_linear_names(
        model, include=include, exclude=exclude
    )
    if not selected:
        raise ValueError("no nn.Linear layer matched include/exclude settings")
    names = []
    for name, layer in selected:
        _replace_module(
            model,
            name,
            QATLinear.from_float(
                layer,
                bits=bits,
                group_size=group_size,
                symmetric=symmetric,
            ),
        )
        names.append(name)
    return names


def convert_qat_model(
    model: nn.Module, *, scale_dtype: torch.dtype = torch.float16
) -> ModelQuantizationReport:
    """Replace every QATLinear with its packed inference form."""
    selected = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, QATLinear)
    ]
    if not selected:
        raise ValueError("model contains no QATLinear modules")

    names, weight_count, fp32_bytes, quantized_bytes = [], 0, 0, 0
    for name, layer in selected:
        quantized = layer.to_quantized(scale_dtype=scale_dtype)
        _replace_module(model, name, quantized)
        names.append(name)
        weight_count += layer.weight.numel()
        fp32_bytes += layer.weight.numel() * 4
        quantized_bytes += quantized.weight_storage_nbytes

    return ModelQuantizationReport(
        layer_names=names,
        weight_count=weight_count,
        fp32_weight_bytes=fp32_bytes,
        quantized_weight_bytes=quantized_bytes,
    )


def print_quantized_layers(model: nn.Module) -> None:
    """Inspect every packed layer and its exact storage."""
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            print("  " + format_quantized_tensor(name, module.as_quantized_tensor()))
