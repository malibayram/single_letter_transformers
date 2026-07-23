"""demo.py — existing TinyQwen checkpoint üzerinde gerçek packed quantization.

Karşılaştırma:
  * FP32 baseline
  * symmetric INT8 weight-only
  * symmetric packed INT4 weight-only

Quantize edilen katmanlar modelin attention/MLP `nn.Linear` weight'leridir.
`lm_head`, token embedding ile aynı weight'i paylaştığı (weight tying) için
bilerek dışarıda bırakılır. Norm ve embedding FP32 kalır.

Ölçülenler:
  * exact weight storage bytes ve effective bits per weight;
  * cross-entropy loss ve perplexity;
  * baseline'a göre logit drift;
  * aynı random seed ile üretilen Türkçe isimler;
  * quantized state_dict round-trip doğrulaması.

Önemli: QuantizedLinear weight'i gerçekten INT8/packed INT4 saklar; ancak bu
okunabilir PyTorch referansı forward sırasında dequantize edip float matmul
yapar. Production INT4 kernel'i olmadığı için speedup ölçülmez/iddia edilmez.

Çalıştır:
    PYTHONPATH=quantization .venv/bin/python quantization/demo.py
    PYTHONPATH=quantization .venv/bin/python quantization/demo.py --group-size 16
"""

from __future__ import annotations

import argparse
import copy
import io
import math
from pathlib import Path
import sys

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
QWEN3_DIR = ROOT / "qwen3"
CHECKPOINT = QWEN3_DIR / "tiny_qwen.pt"
DATA_FILE = ROOT / "data" / "temiz_isimler.txt"

# qwen3 uses flat imports (`from config import ...`), so its directory must be
# first on sys.path before importing the model.
sys.path.insert(0, str(QWEN3_DIR))
from model import TinyQwen  # noqa: E402
from tokenizer import CharTokenizer  # noqa: E402

from linear import (  # noqa: E402
    ModelQuantizationReport,
    QuantizedLinear,
    print_quantized_layers,
    quantize_model_linears,
)


EVAL_BLOCK = 16
EVAL_BATCH = 64
GENERATION_SEED = 2026


def load_checkpoint() -> tuple[dict, CharTokenizer]:
    """Read the existing checkpoint without changing it."""
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    tokenizer = CharTokenizer(checkpoint["chars"])
    return checkpoint, tokenizer


def fresh_float_model(checkpoint: dict) -> TinyQwen:
    """Build one independent model copy from the same FP32 weights."""
    model = TinyQwen(checkpoint["cfg"])
    model.load_state_dict(checkpoint["model"])
    return model.eval()


def fixed_eval_batch(tokenizer: CharTokenizer) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic, evenly-spaced corpus windows (no random sampling)."""
    text = DATA_FILE.read_text(encoding="utf-8")
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    max_start = len(data) - EVAL_BLOCK - 1
    starts = torch.linspace(0, max_start, EVAL_BATCH).long()
    x = torch.stack([data[i : i + EVAL_BLOCK] for i in starts])
    y = torch.stack([data[i + 1 : i + 1 + EVAL_BLOCK] for i in starts])
    return x, y


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor):
    logits, loss = model(x, y)
    return logits, loss.item(), math.exp(loss.item())


@torch.no_grad()
def generate_names(
    model: TinyQwen, tokenizer: CharTokenizer, n: int = 12
) -> list[str]:
    """Reset seed so every model samples from the same random-number stream."""
    torch.manual_seed(GENERATION_SEED)
    start = torch.full((n, 1), tokenizer.eos_id, dtype=torch.long)
    out = model.generate(
        start,
        max_new_tokens=model.cfg.max_seq_len,
        temperature=0.8,
        top_k=None,
        eos_id=tokenizer.eos_id,
    )
    names = [
        tokenizer.decode(row[1:]).split("\n")[0]
        for row in out.tolist()
    ]
    return [name for name in names if name]


def unique_parameter_nbytes(model: nn.Module) -> int:
    """Count tied Parameters once (embedding/lm_head share one object)."""
    seen, total = set(), 0
    for parameter in model.parameters():
        key = id(parameter)
        if key not in seen:
            total += parameter.numel() * parameter.element_size()
            seen.add(key)
    return total


def model_weight_storage_nbytes(model: nn.Module) -> int:
    """Unique remaining Parameters + packed QuantizedLinear weight buffers."""
    total = unique_parameter_nbytes(model)
    total += sum(
        module.total_storage_nbytes
        for module in model.modules()
        if isinstance(module, QuantizedLinear)
    )
    return total


def make_quantized_model(
    checkpoint: dict, *, bits: int, group_size: int
) -> tuple[TinyQwen, ModelQuantizationReport]:
    """Create an independent quantized copy; FP checkpoint remains untouched."""
    model = fresh_float_model(checkpoint)
    report = quantize_model_linears(
        model,
        bits=bits,
        group_size=group_size,
        symmetric=True,
        exclude=("lm_head",),
        scale_dtype=torch.float16,
    )
    return model.eval(), report


def round_trip_state_dict(
    checkpoint: dict,
    model: TinyQwen,
    *,
    bits: int,
    group_size: int,
    x: torch.Tensor,
) -> tuple[int, float]:
    """Serialize packed buffers, reload into same structure, compare logits."""
    memory_file = io.BytesIO()
    torch.save(model.state_dict(), memory_file)
    serialized_bytes = memory_file.tell()
    memory_file.seek(0)
    state = torch.load(memory_file, map_location="cpu", weights_only=True)

    rebuilt, _ = make_quantized_model(
        checkpoint, bits=bits, group_size=group_size
    )
    rebuilt.load_state_dict(state)
    with torch.no_grad():
        original_logits, _ = model(x)
        rebuilt_logits, _ = rebuilt(x)
    max_diff = (original_logits - rebuilt_logits).abs().max().item()
    return serialized_bytes, max_diff


def drift(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[float, float]:
    difference = (candidate - reference).abs()
    return difference.mean().item(), difference.max().item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--group-size",
        type=int,
        default=32,
        help="weights per scale (default: 32)",
    )
    parser.add_argument(
        "--show-layers",
        action="store_true",
        help="print exact payload/metadata for every packed layer",
    )
    args = parser.parse_args()
    if args.group_size <= 0:
        parser.error("--group-size must be positive")

    checkpoint, tokenizer = load_checkpoint()
    x, y = fixed_eval_batch(tokenizer)

    fp32 = fresh_float_model(checkpoint)
    int8, int8_report = make_quantized_model(
        checkpoint, bits=8, group_size=args.group_size
    )
    int4, int4_report = make_quantized_model(
        checkpoint, bits=4, group_size=args.group_size
    )

    fp_logits, fp_loss, fp_ppl = evaluate(fp32, x, y)
    i8_logits, i8_loss, i8_ppl = evaluate(int8, x, y)
    i4_logits, i4_loss, i4_ppl = evaluate(int4, x, y)
    i8_mean, i8_max = drift(fp_logits, i8_logits)
    i4_mean, i4_max = drift(fp_logits, i4_logits)

    fp_bytes = model_weight_storage_nbytes(fp32)
    i8_bytes = model_weight_storage_nbytes(int8)
    i4_bytes = model_weight_storage_nbytes(int4)

    print("TinyQwen weight-only quantization")
    print(f"  checkpoint       : {CHECKPOINT.relative_to(ROOT)}")
    print(f"  unique parameters: {sum(p.numel() for p in fp32.parameters()):,}")
    print(f"  group size       : {args.group_size}")
    print("  excluded         : lm_head (tied to FP32 embedding)")
    print(f"\n  INT8 layers: {int8_report}")
    print(f"  INT4 layers: {int4_report}")

    print("\nWhole-model weight storage (unchanged FP32 tensors included)")
    print(f"  {'model':<8} {'bytes':>10} {'vs FP32':>10}")
    print(f"  {'FP32':<8} {fp_bytes:>10,} {'1.00x':>10}")
    print(f"  {'INT8':<8} {i8_bytes:>10,} {fp_bytes/i8_bytes:>9.2f}x")
    print(f"  {'INT4':<8} {i4_bytes:>10,} {fp_bytes/i4_bytes:>9.2f}x")

    print("\nNumerical evaluation on the same 64 corpus windows")
    print(
        f"  {'model':<8} {'loss':>9} {'ppl':>9} "
        f"{'mean|Δlogit|':>15} {'max|Δlogit|':>14}"
    )
    print(f"  {'FP32':<8} {fp_loss:>9.5f} {fp_ppl:>9.4f} {0.0:>15.6f} {0.0:>14.6f}")
    print(
        f"  {'INT8':<8} {i8_loss:>9.5f} {i8_ppl:>9.4f} "
        f"{i8_mean:>15.6f} {i8_max:>14.6f}"
    )
    print(
        f"  {'INT4':<8} {i4_loss:>9.5f} {i4_ppl:>9.4f} "
        f"{i4_mean:>15.6f} {i4_max:>14.6f}"
    )

    print(f"\nGenerated names (sampling seed={GENERATION_SEED})")
    for label, model in (("FP32", fp32), ("INT8", int8), ("INT4", int4)):
        print(f"  {label:<5}: {generate_names(model, tokenizer)}")

    i4_serialized, i4_reload_diff = round_trip_state_dict(
        checkpoint,
        int4,
        bits=4,
        group_size=args.group_size,
        x=x,
    )
    print("\nPacked state_dict round-trip")
    print(f"  serialized INT4 state_dict: {i4_serialized:,} bytes")
    print(f"  reload max|logit diff|     : {i4_reload_diff:.3e}")

    if args.show_layers:
        print("\nINT4 layer details")
        print_quantized_layers(int4)

    print(
        "\nInterpretation: storage is genuinely low-bit, but this educational\n"
        "forward unpacks to float. Use llama.cpp/another optimized runtime for\n"
        "a real tokens/s benchmark; do not infer speed from this script."
    )

    # Verification: expected structure and deterministic numerical sanity.
    assert int4_report.layer_names == int8_report.layer_names
    assert len(int4_report.layer_names) == 14
    assert i4_bytes < i8_bytes < fp_bytes
    assert i8_mean < i4_mean
    assert i4_reload_diff == 0.0
    assert all(
        module.qweight.dtype == torch.uint8
        for module in int4.modules()
        if isinstance(module, QuantizedLinear)
    )


if __name__ == "__main__":
    main()
