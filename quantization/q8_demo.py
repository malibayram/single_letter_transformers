"""q8_demo.py — detailed Q8_0 experiment on the existing TinyQwen checkpoint.

This compares:

  * FP32 baseline;
  * the folder's generic symmetric group-wise INT8 implementation;
  * a dedicated, structurally faithful llama.cpp Q8_0 implementation.

With group size 32 and FP16 scales, generic symmetric INT8 and Q8_0 use the
same mathematical representation: 32 int8 codes plus one FP16 scale. The
script proves that equivalence on every converted TinyQwen layer, then measures
storage, loss, perplexity, logit drift, generation, and state_dict reload.

Run:
    PYTHONPATH=quantization .venv/bin/python quantization/q8_demo.py
    PYTHONPATH=quantization .venv/bin/python quantization/q8_demo.py --show-layers
"""

from __future__ import annotations

import argparse
import io

import torch

from demo import (
    CHECKPOINT,
    GENERATION_SEED,
    drift,
    evaluate,
    fixed_eval_batch,
    fresh_float_model,
    generate_names,
    load_checkpoint,
    make_quantized_model,
    unique_parameter_nbytes,
)
from linear import QuantizedLinear
from q8_0 import Q8_0Linear, quantize_model_q8_0, q8_0_layout_report


def model_weight_storage_nbytes(model: torch.nn.Module) -> int:
    """Count unique Parameters plus any low-bit Linear buffers."""
    total = unique_parameter_nbytes(model)
    total += sum(
        module.total_storage_nbytes
        for module in model.modules()
        if isinstance(module, (QuantizedLinear, Q8_0Linear))
    )
    return total


def make_q8_0_model(checkpoint: dict):
    model = fresh_float_model(checkpoint)
    report = quantize_model_q8_0(model, exclude=("lm_head",))
    return model.eval(), report


def prove_generic_int8_equivalence(
    generic_model: torch.nn.Module,
    q8_model: torch.nn.Module,
) -> tuple[int, int, float]:
    """Compare every code and scale in corresponding quantized layers."""
    generic_layers = [
        (name, module)
        for name, module in generic_model.named_modules()
        if isinstance(module, QuantizedLinear)
    ]
    q8_layers = [
        (name, module)
        for name, module in q8_model.named_modules()
        if isinstance(module, Q8_0Linear)
    ]
    if [name for name, _ in generic_layers] != [name for name, _ in q8_layers]:
        raise AssertionError("generic INT8 and Q8_0 converted different layers")

    code_mismatches = 0
    scale_mismatches = 0
    maximum_weight_difference = 0.0
    for (_, generic), (_, q8) in zip(generic_layers, q8_layers):
        generic_codes = generic.qweight.reshape(-1, 32)
        generic_scales = generic.scales.reshape(-1)
        code_mismatches += int((generic_codes != q8.qweight).sum().item())
        scale_mismatches += int((generic_scales != q8.scales).sum().item())
        maximum_weight_difference = max(
            maximum_weight_difference,
            (
                generic.dequantize_weight() - q8.dequantize_weight()
            ).abs().max().item(),
        )
    return code_mismatches, scale_mismatches, maximum_weight_difference


def q8_state_dict_round_trip(
    checkpoint: dict,
    q8_model: torch.nn.Module,
    x: torch.Tensor,
) -> tuple[int, float]:
    memory_file = io.BytesIO()
    torch.save(q8_model.state_dict(), memory_file)
    serialized_bytes = memory_file.tell()
    memory_file.seek(0)
    state = torch.load(memory_file, map_location="cpu", weights_only=True)

    rebuilt, _ = make_q8_0_model(checkpoint)
    rebuilt.load_state_dict(state)
    with torch.no_grad():
        original_logits, _ = q8_model(x)
        rebuilt_logits, _ = rebuilt(x)
    return serialized_bytes, (original_logits - rebuilt_logits).abs().max().item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--show-layers",
        action="store_true",
        help="print Q8_0 block/byte accounting for every converted layer",
    )
    args = parser.parse_args()

    checkpoint, tokenizer = load_checkpoint()
    x, y = fixed_eval_batch(tokenizer)

    fp32 = fresh_float_model(checkpoint)
    generic_i8, generic_report = make_quantized_model(
        checkpoint, bits=8, group_size=32
    )
    q8, q8_report = make_q8_0_model(checkpoint)

    fp_logits, fp_loss, fp_ppl = evaluate(fp32, x, y)
    generic_logits, generic_loss, generic_ppl = evaluate(generic_i8, x, y)
    q8_logits, q8_loss, q8_ppl = evaluate(q8, x, y)
    generic_mean, generic_max = drift(fp_logits, generic_logits)
    q8_mean, q8_max = drift(fp_logits, q8_logits)

    fp_bytes = model_weight_storage_nbytes(fp32)
    generic_bytes = model_weight_storage_nbytes(generic_i8)
    q8_bytes = model_weight_storage_nbytes(q8)
    code_diff, scale_diff, weight_diff = prove_generic_int8_equivalence(
        generic_i8, q8
    )
    serialized_bytes, reload_diff = q8_state_dict_round_trip(
        checkpoint, q8, x
    )

    print("TinyQwen — detailed Q8_0 study")
    print(f"  checkpoint             : {CHECKPOINT.name}")
    print("  Q8_0 block             : 32 × int8 code + 1 × FP16 scale")
    print("  exact block cost       : 34 bytes = 8.5 bits/weight")
    print("  quantized target       : 14 attention/MLP Linear layers")
    print("  excluded               : tied lm_head/embedding and norms\n")

    print("Converted Linear storage")
    print(f"  generic symmetric INT8 : {generic_report}")
    print(f"  dedicated Q8_0         : {q8_report}")

    print("\nWhole-model weight storage (unchanged tensors included)")
    print(f"  {'model':<14} {'bytes':>10} {'vs FP32':>10}")
    print(f"  {'FP32':<14} {fp_bytes:>10,} {'1.00x':>10}")
    print(
        f"  {'generic INT8':<14} {generic_bytes:>10,} "
        f"{fp_bytes/generic_bytes:>9.2f}x"
    )
    print(f"  {'Q8_0':<14} {q8_bytes:>10,} {fp_bytes/q8_bytes:>9.2f}x")

    print("\nNumerical evaluation on the same 64 corpus windows")
    print(
        f"  {'model':<14} {'loss':>9} {'ppl':>9} "
        f"{'mean|Δlogit|':>15} {'max|Δlogit|':>14}"
    )
    print(
        f"  {'FP32':<14} {fp_loss:>9.5f} {fp_ppl:>9.4f} "
        f"{0.0:>15.6f} {0.0:>14.6f}"
    )
    print(
        f"  {'generic INT8':<14} {generic_loss:>9.5f} {generic_ppl:>9.4f} "
        f"{generic_mean:>15.6f} {generic_max:>14.6f}"
    )
    print(
        f"  {'Q8_0':<14} {q8_loss:>9.5f} {q8_ppl:>9.4f} "
        f"{q8_mean:>15.6f} {q8_max:>14.6f}"
    )

    print("\nRepresentation equivalence proof")
    print(f"  differing int8 codes   : {code_diff}")
    print(f"  differing FP16 scales  : {scale_diff}")
    print(f"  max dequant weight diff: {weight_diff:.3e}")
    print(f"  max model logit diff   : {(generic_logits-q8_logits).abs().max().item():.3e}")
    print(
        "  conclusion              : group32 symmetric INT8 + FP16 scale is "
        "Q8_0's tensor math"
    )

    print(f"\nGenerated Turkish names (sampling seed={GENERATION_SEED})")
    print(f"  {'FP32':<14}: {generate_names(fp32, tokenizer)}")
    print(f"  {'Q8_0':<14}: {generate_names(q8, tokenizer)}")

    print("\nQ8_0 state_dict round-trip")
    print(f"  serialized state_dict  : {serialized_bytes:,} bytes")
    print(f"  reload max|logit diff| : {reload_diff:.3e}")

    if args.show_layers:
        print("\nLayer-by-layer Q8_0 storage")
        for name, module in q8.named_modules():
            if isinstance(module, Q8_0Linear):
                print(f"  {name:<33} {q8_0_layout_report(module.as_quantized_tensor())}")

    print(
        "\nDo not read Python runtime from this script as a speed benchmark. "
        "Q8_0 can\n"
        "reduce storage and bandwidth, but actual tokens/s depends on the "
        "runtime,\n"
        "hardware, kernel support, model shape, context, and offload settings."
    )

    assert generic_report.layer_names == q8_report.layer_names
    assert generic_report.effective_bits_per_weight == 8.5
    assert q8_report.effective_bits_per_weight == 8.5
    assert generic_bytes == q8_bytes
    assert code_diff == scale_diff == 0
    assert weight_diff == 0.0
    assert torch.equal(generic_logits, q8_logits)
    assert reload_diff == 0.0


if __name__ == "__main__":
    main()
