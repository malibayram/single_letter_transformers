# quantization — low-bit LLM weights, from one rounded number to packed INT4

Quantization implemented from scratch so every step can be **calculated by
hand, inspected byte by byte, trained with QAT, and tested on the repository's
TinyQwen model**.

The code is deliberately small, but the storage is real:

- INT8 weights are stored as `torch.int8`; Q8_0 additionally implements the
  exact 32-code + FP16-scale block layout used by llama.cpp.
- INT4 weights are packed two signed values per `torch.uint8` byte.
- Scale and zero-point metadata are counted explicitly.
- `QuantizedLinear` contains no hidden floating-point copy of its weight.
- Q4_K's 256-weight, 144-byte structure is built and unpacked.
- NF4 uses the real non-uniform 16-level codebook and packed indices.

The compute path is intentionally educational: `QuantizedLinear` unpacks and
dequantizes its weight before calling PyTorch's floating-point matrix
multiplication. It proves storage and numerical behavior, but it is **not an
optimized integer kernel** and makes no speedup claim.

For the long Turkish explanation, read
[`quantization_turkce_anlatim.md`](quantization_turkce_anlatim.md).

> **The one-sentence idea.** Replace many precise floating-point values with a
> small set of integer/codebook levels plus a little metadata; save memory and
> memory bandwidth, accepting controlled approximation error.

---

## Table of contents

1. [Three layers people often mix up](#1-three-layers-people-often-mix-up)
2. [The affine quantization formula](#2-the-affine-quantization-formula)
3. [Symmetric and asymmetric quantization](#3-symmetric-and-asymmetric-quantization)
4. [Granularity and group size](#4-granularity-and-group-size)
5. [Real INT4 packing](#5-real-int4-packing)
6. [`QuantizedTensor` and `QuantizedLinear`](#6-quantizedtensor-and-quantizedlinear)
7. [PTQ and QAT](#7-ptq-and-qat)
8. [Q8 and Q8_0 in detail](#8-q8-and-q8_0-in-detail)
9. [Q4_K and Q4_K_M](#9-q4_k-and-q4_k_m)
10. [NF4 and QLoRA](#10-nf4-and-qlora)
11. [GPTQ, AWQ, and SmoothQuant](#11-gptq-awq-and-smoothquant)
12. [TinyQwen results](#12-tinyqwen-results)
13. [File-by-file guide](#13-file-by-file-guide)
14. [How to run everything](#14-how-to-run-everything)
15. [What is real and what is simplified](#15-what-is-real-and-what-is-simplified)
16. [Sources](#16-sources)

---

## 1. Three layers people often mix up

Quantization discussions become much easier when these are kept separate:

| Layer | Example | What it answers |
| --- | --- | --- |
| **Container** | GGUF | How are model tensors and metadata stored in one file? |
| **Tensor encoding** | `Q4_K`, `Q8_0`, NF4 | How is one tensor represented? |
| **Model recipe** | `Q4_K_M`, `Q5_K_M` | Which encoding is chosen for each model tensor? |
| **Compute kernel** | llama.cpp CPU/Metal/CUDA matmul | How are encoded weights used efficiently? |

This folder implements tensor encodings and model replacement in readable
PyTorch. It does not write GGUF and does not implement fused CPU/GPU kernels.

`Q4_K_M` is therefore **not** a Python dtype and not “all weights are four
bits.” It is a llama.cpp model-level recipe that uses a Q4_K base while
promoting selected sensitive tensors to higher precision. The exact selection
can depend on architecture, tensor shape, and llama.cpp version.

---

## 2. The affine quantization formula

The same two equations explain most linear integer quantization:

```text
q     = clip(round(x / scale) + zero_point, qmin, qmax)
x_hat = scale * (q - zero_point)
```

Where:

- `x` is the original float value.
- `q` is the stored integer code.
- `x_hat` is the reconstructed approximation.
- `scale` says how much one integer step is worth.
- `zero_point` says which integer code represents real zero.
- `clip` prevents codes from leaving the available bit range.

Quantization error is:

```text
error = x_hat - x
MSE   = mean(error²)
```

### Hand example

Use seven symmetric codes `[-3, -2, -1, 0, 1, 2, 3]`:

```text
x     = [-1.2, -0.7, 0.1, 0.8, 1.1]
scale = max(abs(x)) / 3 = 0.4
q     = round(x / 0.4) = [-3, -2, 0, 2, 3]
x_hat = q * 0.4        = [-1.2, -0.8, 0.0, 0.8, 1.2]
MSE   = 0.006
```

If calibration fixed the range at `[-1.2, 1.2]`, a later value `1.6` is clipped
to code `3` and reconstructed as `1.2`. That is clipping error.

Every number is printed and asserted in [`by_hand.py`](by_hand.py).

---

## 3. Symmetric and asymmetric quantization

### Symmetric

For signed `b`-bit quantization in this module:

```text
qmax  = 2^(b-1) - 1
qmin  = -qmax
scale = max(abs(x)) / qmax
zero_point = 0
```

At 4-bit we use `[-7, 7]`. The unused `-8` code keeps positive and negative
ranges symmetric and makes the explanation straightforward.

Advantages:

- one scale, no stored zero-point;
- simple and common for neural-network weights;
- zero is always exact.

Cost:

- wastes levels if the distribution is strongly one-sided.

### Asymmetric

For unsigned `b`-bit quantization:

```text
qmin = 0
qmax = 2^b - 1
scale = (xmax - xmin) / (qmax - qmin)
zero_point = round(qmin - xmin / scale)
```

The observed range is expanded to include real zero, ensuring `zero_point`
stays representable. At 4-bit the codes are `[0, 15]`.

Advantages:

- can use levels more efficiently for shifted/non-symmetric data.

Costs:

- must store a zero-point per tensor/channel/group;
- extra metadata and slightly more arithmetic.

Use:

```python
from core import quantize_tensor

qt = quantize_tensor(
    weight,
    bits=4,
    group_size=32,
    symmetric=True,
)
weight_hat = qt.dequantize()
```

---

## 4. Granularity and group size

One scale can serve different amounts of data:

| Granularity | Scales | Error tendency | Metadata |
| --- | ---: | --- | --- |
| Per-tensor | one for whole tensor | highest when outliers exist | smallest |
| Per-channel/row | one per output row | usually lower | larger |
| Group-wise | one per small weight group | lower again | larger again |
| Per-element | one per value | no useful compression | enormous |

This module splits the final dimension into groups. A Linear weight has shape
`[out_features, in_features]`, so `group_size=32` gives one scale for every 32
input weights in each output row.

### Why smaller groups help

Suppose:

```text
[0.1, 0.2, 0.3, 10.0]
```

One group uses a scale determined by `10.0`; the small values can all round to
zero. Splitting into `[0.1, 0.2]` and `[0.3, 10.0]` protects the first pair.

### Why groups are not free

With packed INT4 and one FP16 scale:

```text
effective bpw = 4 + 16 / group_size
```

For dimensions that require no padding:

| Group size | Payload | FP16 scale overhead | Effective bpw |
| ---: | ---: | ---: | ---: |
| 64 | 4.0 | 0.25 | 4.25 |
| 32 | 4.0 | 0.50 | 4.50 |
| 16 | 4.0 | 1.00 | 5.00 |
| 8 | 4.0 | 2.00 | 6.00 |

The payload remains four bits. Better local scales cost metadata.

Padding also matters: if the last dimension is not divisible by `group_size`,
the final stored group is padded, increasing effective bpw for a small tensor.

---

## 5. Real INT4 packing

PyTorch has `int8`, but a normal tensor of values in `[-7, 7]` still consumes
one full byte per element. It is not compressed just because the values happen
to fit in four bits.

[`core.py`](core.py) packs two signed values into one byte:

```text
values  = [-7, -1, 0, 7]
nibbles = [0x9, 0xF, 0x0, 0x7]   # signed two's complement
bytes   = [0xF9, 0x70]            # first value in low nibble
```

Code:

```python
from core import pack_int4, unpack_int4

packed = pack_int4(values)        # uint8, half as many elements
restored_codes = unpack_int4(packed)
```

Packing is lossless. Quantization caused approximation earlier when floats were
mapped to integer codes; packing only rearranges those codes into bytes.

That distinction is important:

```text
float -> integer codes     lossy quantization
codes -> packed bytes      lossless encoding
packed bytes -> codes      lossless decoding
codes -> approximate float dequantization
```

---

## 6. `QuantizedTensor` and `QuantizedLinear`

### `QuantizedTensor`

[`core.py`](core.py) returns a dataclass containing only:

- packed integer payload;
- FP16/FP32 group scales;
- optional integer zero-points;
- shape, padding, bit width, and group metadata.

It reports:

```python
qt.payload_nbytes
qt.metadata_nbytes
qt.storage_nbytes
qt.effective_bits_per_weight
```

No original float tensor is stored inside it.

### `QuantizedLinear`

[`linear.py`](linear.py) converts an existing `nn.Linear`:

```python
from linear import QuantizedLinear

qlinear = QuantizedLinear.from_float(
    float_linear,
    bits=4,
    group_size=32,
    symmetric=True,
)
```

Its buffers are:

```text
qweight       uint8 packed INT4, or int8/uint8 for 8-bit
scales        one FP16/FP32 value per group
zero_points   absent for symmetric, int16 for asymmetric
bias          unchanged float bias, if the source layer had one
```

There is no `weight` Parameter:

```python
assert not hasattr(qlinear, "weight")
assert qlinear.qweight.dtype == torch.uint8
```

The reference forward path is:

```text
packed qweight
    -> unpack integer codes
    -> dequantize approximate float weight
    -> F.linear(input, weight_hat, bias)
```

A production kernel fuses or reorganizes this work. This module keeps each step
visible so a student can inspect it.

### Replacing a model

```python
from linear import quantize_model_linears

report = quantize_model_linears(
    model,
    bits=4,
    group_size=32,
    exclude=("lm_head",),
)
print(report)
```

For TinyQwen, `lm_head` is excluded because it shares its Parameter with the
token embedding. Replacing only the head would break weight tying.

---

## 7. PTQ and QAT

### PTQ — Post-Training Quantization

Train a float model first, then quantize:

```text
float training -> finished float weight -> quantize once -> inference
```

Advantages:

- fast and simple;
- no training dataset required for naive weight-only quantization;
- calibration-aware techniques can use a small representative dataset.

Risk:

- the model never had an opportunity to adapt to rounding/clipping noise.

`QuantizedLinear.from_float()` is the direct PTQ path.

### QAT — Quantization-Aware Training

QAT keeps a trainable floating-point master weight but simulates quantization in
the forward pass:

```text
FP master weight
    -> fake quantize/dequantize
    -> forward sees low-bit numerical values
    -> backward updates FP master weight
    -> convert to real packed storage after training
```

Rounding has zero derivative almost everywhere. A straight-through estimator
(STE) uses:

```python
output = weight + (quantized_weight - weight).detach()
```

Forward:

```text
output = quantized_weight
```

Backward:

```text
d(output)/d(weight) = 1
```

[`QATLinear`](linear.py) implements this, and [`qat_demo.py`](qat_demo.py)
shows:

```text
Float model          : 98.7%
Direct PTQ INT4      : 96.6%
QAT -> packed INT4   : 97.0%
```

This one toy run illustrates adaptation; it is not a universal guarantee. Real
QAT systems add observers, staged enable/freeze schedules, learned or calibrated
qparams, and backend-specific conversion.

---

## 8. Q8 and Q8_0 in detail

### “Q8” is not one complete specification

People often use “Q8” to mean “approximately eight-bit quantization,” but a
working format must answer more questions:

- signed or unsigned codes?
- symmetric or asymmetric?
- one scale per tensor, row, token, or block?
- what scale dtype is stored?
- are weights, activations, or both quantized?
- which runtime/kernel understands the layout?

Keep these common labels separate:

| Label | Meaning |
| --- | --- |
| Generic INT8 | A broad family of eight-bit integer representations |
| `Q8_0` | A specific llama.cpp/GGML block-wise tensor encoding |
| W8A8 | A compute setup with 8-bit weights and 8-bit activations |
| `Q8_K` | A different K-quant helper/intermediate type; not another name for `Q8_0` |
| GGUF | The container that can hold Q8_0 tensors and model metadata |

This module's generic symmetric INT8 becomes mathematically equivalent to
Q8_0 when it uses `group_size=32`, FP16 scales, and the signed range
`[-127,127]`. The dedicated [`q8_0.py`](q8_0.py) implementation makes the
on-disk block fields and constraints explicit.

### The exact Q8_0 block

The current llama.cpp type is conceptually:

```c
#define QK8_0 32
struct block_q8_0 {
    fp16  d;       // one scale/delta
    int8  qs[32];  // 32 signed codes
};
```

Byte accounting:

```text
32 int8 codes × 1 byte = 32 bytes
 1 FP16 scale × 2 bytes =  2 bytes
----------------------------------
one Q8_0 block           = 34 bytes

34 × 8 / 32 = 8.5 effective bits per weight
```

So “Q8” describes the eight-bit payload, not the complete storage cost.
Metadata adds 0.5 bit per weight. A whole GGUF can differ further because
model-level conversion can leave unsupported or sensitive tensors in other
types and adds file metadata/alignment.

### Quantization and dequantization equations

For every independent 32-value block:

```text
amax = max(abs(x))
d    = amax / 127
q    = clamp(round(x / d), -127, 127)
x_hat = q × stored_FP16(d)
```

Important implementation details:

1. `zero_point=0`, so real zero maps to code zero.
2. The code `-128` is unused; `[-127,127]` keeps the range symmetric.
3. Each block has its own scale, limiting an outlier's effect to 32 values.
4. llama.cpp calculates integer codes from the FP32 scale, then stores `d` as
   FP16. Dequantization uses that stored FP16 value.
5. Quantized row length must be divisible by 32. [`q8_0.py`](q8_0.py) raises
   an error instead of silently pretending padding is GGUF-compatible.

For an all-zero block, `d=0`, every code is zero, and the reconstructed block
is exactly zero; the implementation avoids division by zero explicitly.

### Pencil-friendly block example

Let the largest magnitude be `12.7`:

```text
d_FP32 = 12.7 / 127 ≈ 0.1
```

Selected values from one 32-value block:

| `x` | `x/d` | stored `q` | `x_hat` using FP16 `d` |
| ---: | ---: | ---: | ---: |
| -12.70 | -127.0 | -127 | -12.696899 |
| -6.35 | -63.5 | -64 | -6.398438 |
| -3.20 | -32.0 | -32 | -3.199219 |
| -1.05 | -10.5 | -11 | -1.099731 |
| 0.00 | 0.0 | 0 | 0.000000 |
| 0.14 | 1.4 | 1 | 0.099976 |
| 1.25 | 12.5 | 13 | 1.299683 |
| 4.40 | 44.0 | 44 | 4.398926 |
| 8.88 | 88.8 | 89 | 8.897827 |
| 12.70 | 127.0 | 127 | 12.696899 |

The stored FP16 scale is `0.099975586`, not exactly decimal `0.1`; this is why
even the endpoint has a small reconstruction error. The script prints these
values and asserts the exact codes.

### Why 32-value blocks beat one global scale

[`q8_0.py`](q8_0.py) concatenates:

```text
block 0: 32 values between -0.25 and 0.25
block 1: 32 values between -20 and 20
```

Measured result:

| Method | Scales | Small-block MSE | Total MSE | Storage |
| --- | ---: | ---: | ---: | ---: |
| Q8_0 | 2 | 0.000000313 | 0.001001081 | 68 B |
| One global INT8 group | 1 | 0.002284769 | 0.002143309 | 66 B |

The global scale's step is set by `20.0`, so small numbers receive poor
resolution. Q8_0 spends two additional metadata bytes to isolate the two
ranges. This is the same granularity-versus-metadata trade-off seen in INT4.

### Q8_0 is weight encoding, not automatically W8A8

A Q8_0 GGUF commonly stores model weight tensors in Q8_0. That statement alone
does not say activations or the KV cache are eight-bit. Our `Q8_0Linear`:

1. stores the weight as int8 blocks plus FP16 scales;
2. reconstructs a floating-point weight during `forward`;
3. calls PyTorch `F.linear`.

This proves storage and numerical behavior, but not an integer compute speedup.
A production runtime can use optimized CPU/GPU kernels directly on the Q8_0
layout. Actual speed depends on backend, hardware, tensor shapes, offload,
batch/context, and memory bandwidth.

### When Q8_0 is a good choice

Q8_0 is useful when:

- you want a high-quality quantized baseline close to the source model;
- Q4/Q5 changes a quality-sensitive task too much;
- memory can afford roughly half the raw FP16 weight payload;
- the target runtime has a good Q8_0 kernel;
- you need a convenient high-quality quantized storage/conversion stage.

It may be a poor first choice when:

- RAM/VRAM is tight and Q4_K_M/Q5_K_M quality is already sufficient;
- long-context KV cache, rather than weights, dominates memory;
- a specific backend accelerates another format better;
- the source model already has equal or lower native precision, so eight bits
  cannot recreate information absent from the source.

Do not assume Q8_0 is lossless. Do not assume fewer-bit formats are always
faster. Benchmark the actual model, workload, runtime, and machine.

### Runnable APIs

One tensor:

```python
from q8_0 import quantize_q8_0, q8_0_layout_report

q8 = quantize_q8_0(weight)  # last dimension must be divisible by 32
print(q8.qs.dtype)          # torch.int8
print(q8.d.dtype)           # torch.float16
print(q8_0_layout_report(q8))
weight_hat = q8.dequantize()
```

One Linear:

```python
from q8_0 import Q8_0Linear

qlinear = Q8_0Linear.from_float(float_linear)
assert not hasattr(qlinear, "weight")
output = qlinear(input)
```

Whole TinyQwen:

```bash
.venv/bin/python quantization/q8_demo.py --show-layers
```

The model demo proves code/scale equivalence with generic group-32 INT8,
measures loss/perplexity/logit drift, generates names, and verifies an exact
Q8_0 `state_dict` reload.

---

## 9. Q4_K and Q4_K_M

### Q4_K tensor structure

A Q4_K super-block contains 256 weights:

```text
128 bytes = 256 4-bit codes
 12 bytes = 8 six-bit scale codes + 8 six-bit minimum codes
  2 bytes = FP16 d
  2 bytes = FP16 dmin
--------------------------------
144 bytes
```

Therefore:

```text
144 * 8 / 256 = 4.5 bits per weight
```

[`q4_k.py`](q4_k.py) implements:

- the 8 × 32 sub-block layout;
- Q4 code nibble packing;
- the exact 12-byte packing shape for 6-bit scale/minimum codes;
- FP16 super-block multipliers;
- dequantization;
- exact byte accounting.

Its simple min/max qparam fitting is for teaching. It does not reproduce
llama.cpp's optimized quantizer, importance weighting, or GGUF output
byte-for-byte.

### Q4_K_M model recipe

`Q4_K_M` is a model-level mixed-precision recipe:

- Q4_K is the base for many eligible weight tensors.
- Sensitive output/attention/FFN tensors may be promoted to Q5_K/Q6_K.
- one-dimensional norms and incompatible shapes may remain in other types;
- architecture, layer position, expert count, tensor shape, imatrix use, and
  llama.cpp version can affect the exact selection.

The correct mental model is:

```text
Q4_K     = one tensor encoding family
Q4_K_M   = a rule choosing encodings across a whole model
GGUF     = the file containing those tensors and metadata
```

Do not calculate a `Q4_K_M` file as `parameter_count * 4 / 8`. Mixed tensors,
scale/min metadata, alignment, and GGUF metadata all matter.

### Practical starting point

For local GGUF inference, `Q4_K_M` is a sensible first candidate because it
usually balances size and quality, and current llama.cpp tooling uses it as a
default remote quant when available. It is not automatically best for every
model/backend.

Measure:

- task quality or perplexity;
- total model and runtime memory;
- prompt processing speed;
- generation tokens/s;
- long-context KV-cache memory;
- backend support.

---

## 10. NF4 and QLoRA

Uniform INT4 spends levels at equal intervals. NF4 (NormalFloat4) uses a
non-uniform 16-value codebook chosen for normally distributed weights.

[`nf4.py`](nf4.py) contains the codebook and:

1. splits weights into 64-value blocks;
2. stores one FP16 `absmax` scale per block;
3. normalizes each block to `[-1, 1]`;
4. chooses the nearest NF4 code;
5. packs two codebook indices per byte.

Storage without double quantization:

```text
4 payload bits + 16 scale bits / 64 weights = 4.25 bpw
```

Actual demo:

```text
distribution         uniform INT4 MSE       NF4 MSE     winner
normal N(0,1)              0.01226308    0.00832188        NF4
uniform [-1,1]             0.00163648    0.00273948    uniform
```

NF4's advantage comes from matching a distribution. It is not universally
better for arbitrary data.

QLoRA combines:

- a frozen 4-bit NF4 base model;
- double quantization of quantization constants;
- paged optimizers;
- trainable LoRA adapters.

This folder implements basic NF4. The repository's [`lora/`](../lora/) module
implements LoRA separately. A full QLoRA training stack would combine those
ideas with backend kernels and memory-management features.

---

## 11. GPTQ, AWQ, and SmoothQuant

[`advanced_methods.py`](advanced_methods.py) reduces each method to its central
working idea. These are toy experiments, not drop-in production quantizers.

### GPTQ

Goal: weight-only PTQ that minimizes layer-output damage, not just independent
weight error.

The toy implementation:

1. builds an approximate Hessian from calibration inputs;
2. quantizes one weight at a time;
3. propagates its error into remaining weights using the inverse-Hessian
   Cholesky factor.

Example:

```text
naive calibration output MSE     : 0.298211
GPTQ-like compensated output MSE : 0.222547
```

Full GPTQ adds layer/block processing, group quantization, ordering, damping,
efficient memory use, serialization, and optimized kernels.

### AWQ

Goal: protect weight channels that matter to large/important activations.

The toy implementation:

1. measures channel magnitudes on calibration inputs;
2. searches an exponent `alpha`;
3. scales important weight columns up and inversely scales activations down;
4. quantizes the scaled weights;
5. selects the scale producing smallest calibration output error.

The paired transformation is exact before quantization:

```text
X W^T = (X / s) (W * s)^T
```

Example:

```text
naive held-out test MSE            : 0.141348
activation-aware held-out test MSE : 0.059957
```

### SmoothQuant

Goal: make W8A8 practical when activation outliers stretch the INT8 range.

It migrates quantization difficulty from activations to weights:

```text
X W^T = (X / s) (W * s)^T
s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)
```

Example:

```text
naive per-tensor W8A8 MSE    : 0.949966
smoothed per-tensor W8A8 MSE : 0.123037
```

The dramatic toy gap is intentional: large activation outliers make the
mechanism visible.

---

## 12. TinyQwen results

Run:

```bash
.venv/bin/python quantization/demo.py --show-layers
```

The existing `qwen3/tiny_qwen.pt` has 19,584 unique Parameters. Fourteen
attention/MLP Linear layers contain 18,432 weights; tied `lm_head`/embedding and
norms remain FP32.

With `group_size=32` and FP16 scales, generic symmetric INT8 has the same
codes, scales, storage, and reconstructed weights as the dedicated Q8_0 path:

```text
Q8_0 linears: 73,728 B FP32 -> 19,584 B, 8.500 bpw
INT4 linears: 73,728 B FP32 -> 10,368 B, 4.500 bpw
```

Whole-model weight storage:

| Model | Bytes | Compression vs FP32 |
| --- | ---: | ---: |
| FP32 | 78,336 | 1.00× |
| Q8_0 | 24,192 | 3.24× |
| INT4 | 14,976 | 5.23× |

Evaluation on the same 64 corpus windows:

| Model | Loss | Perplexity | Mean absolute logit drift |
| --- | ---: | ---: | ---: |
| FP32 | 0.79982 | 2.2251 | 0 |
| Q8_0 | 0.79810 | 2.2213 | 0.086648 |
| INT4 | 1.43681 | 4.2072 | 1.554702 |

This tiny model is sensitive to simple INT4 round-to-nearest quantization. That
is a useful result, not a failure: fewer bits create a real quality trade-off,
and sophisticated recipes/calibration exist for a reason.

[`q8_demo.py`](q8_demo.py) additionally proves across all 14 layers:

```text
differing int8 codes      = 0
differing FP16 scales     = 0
max dequant weight diff   = 0
max model logit diff      = 0
Q8_0 state_dict reload Δ  = 0
```

This is a precise equivalence for this module's group-32 symmetric INT8
configuration, not a claim that every product called “INT8” is Q8_0.

---

## 13. File-by-file guide

| File | Read it for |
| --- | --- |
| [`README.md`](README.md) | Main map, commands, and results |
| [`quantization_turkce_anlatim.md`](quantization_turkce_anlatim.md) | Long Turkish theory + implementation walkthrough |
| [`core.py`](core.py) | Affine qparams, packing, dequantization, STE |
| [`linear.py`](linear.py) | Packed `QuantizedLinear`, `QATLinear`, model replacement |
| [`q8_0.py`](q8_0.py) | Exact Q8_0 block layout, quantizer, `Q8_0Linear`, hand example |
| [`q8_demo.py`](q8_demo.py) | Dedicated Q8_0 TinyQwen evaluation and equivalence proof |
| [`q4_k.py`](q4_k.py) | Q4_K 256-value/144-byte structure |
| [`nf4.py`](nf4.py) | NF4 codebook quantization and comparison |
| [`by_hand.py`](by_hand.py) | Eight pencil-and-paper proofs |
| [`advanced_methods.py`](advanced_methods.py) | GPTQ/AWQ/SmoothQuant core ideas |
| [`qat_demo.py`](qat_demo.py) | PTQ vs QAT on a trainable task |
| [`demo.py`](demo.py) | FP32/INT8/INT4 on the real TinyQwen checkpoint |
| [`CHEAT_SHEET.md`](CHEAT_SHEET.md) | One-page formulas and selection guide |
| [`EXERCISES.md`](EXERCISES.md) | Practice questions with solutions |
| [`GUIDELINE.md`](GUIDELINE.md) | Teaching strategy |
| [`PLAN.md`](PLAN.md) | Original implementation plan and verification contract |

Recommended reading order:

```text
README
  -> by_hand.py
  -> core.py
  -> linear.py
  -> q8_0.py / q8_demo.py
  -> demo.py
  -> q4_k.py / nf4.py
  -> advanced_methods.py
  -> qat_demo.py
  -> Turkish deep dive + exercises
```

---

## 14. How to run everything

From the repository root:

```bash
# 1. Every basic equation, packing byte, and Q4_K size
.venv/bin/python quantization/by_hand.py

# 2. Q8_0 block arithmetic, outlier isolation, and one real Linear
.venv/bin/python quantization/q8_0.py

# 3. Dedicated TinyQwen FP32 vs Q8_0 study
.venv/bin/python quantization/q8_demo.py --show-layers

# 4. Existing TinyQwen: FP32 vs INT8 vs packed INT4
.venv/bin/python quantization/demo.py

# Exact storage for every converted layer
.venv/bin/python quantization/demo.py --show-layers

# Study the quality/metadata effect of another group size
.venv/bin/python quantization/demo.py --group-size 16

# 5. PTQ vs QAT, followed by real packed conversion
.venv/bin/python quantization/qat_demo.py

# 6. NF4 codebook vs uniform INT4
.venv/bin/python quantization/nf4.py

# 7. GPTQ, AWQ, SmoothQuant distinguishing ideas
.venv/bin/python quantization/advanced_methods.py
```

No external model download, GPU, transformers, bitsandbytes, or llama.cpp build
is required. Only the repository's existing `.venv`, PyTorch, data, and
checkpoint are used.

### Import in a notebook

```python
import sys
sys.path.insert(0, "quantization")

from core import quantize_tensor, quantization_error
from linear import QuantizedLinear
```

One tensor:

```python
import torch

w = torch.randn(32, 64)
qw = quantize_tensor(
    w,
    bits=4,
    group_size=32,
    symmetric=True,
    scale_dtype=torch.float16,
)

print(qw.data.dtype)                    # torch.uint8
print(qw.storage_nbytes)
print(qw.effective_bits_per_weight)     # 4.5 when no padding
print(quantization_error(w, qw))
```

One Linear:

```python
layer = torch.nn.Linear(64, 32, bias=False)
qlayer = QuantizedLinear.from_float(
    layer,
    bits=4,
    group_size=32,
)

x = torch.randn(4, 64)
y = qlayer(x)
print(qlayer)
```

---

## 15. What is real and what is simplified

### Implemented for real

- affine symmetric/asymmetric qparams;
- group-wise quantization;
- clipping and dequantization;
- signed and unsigned nibble packing;
- actual INT8/packed INT4 tensor storage;
- structurally faithful Q8_0 blocks and exact 8.5-bpw accounting;
- `Q8_0Linear` and model-tree conversion with no FP weight copy;
- exact payload/metadata byte accounting;
- inference module with no float weight copy;
- serializable quantized buffers;
- QAT fake quantization with STE and packed conversion;
- NF4 codebook/index packing;
- Q4_K structural packing and 4.5 bpw accounting;
- calibration-based AWQ/SmoothQuant/GPTQ toy mathematics;
- real TinyQwen loss, perplexity, logits, and generation comparison.

### Simplified or deliberately absent

- no fused integer CPU/GPU matrix-multiplication kernel;
- no GGUF reader/writer;
- Q8_0 bytes are represented in separate PyTorch buffers rather than emitted
  as an interleaved GGUF byte stream;
- educational Q4_K qparam fitting is not llama.cpp byte-compatible;
- no complete architecture-specific Q4_K_M recipe clone;
- no full GPTQ/AWQ library implementation;
- no QLoRA double quantization, paged optimizer, or fused NF4 kernel;
- no activation/KV-cache runtime integration;
- no claim that Python reference timing predicts production speed.

These boundaries are printed in the scripts because honest separation is part
of learning the implementation.

---

## 16. Sources

Primary papers and official project references:

- [llama.cpp quantization guide](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)
- [llama.cpp tensor encoding schemes](https://github.com/ggml-org/llama.cpp/wiki/Tensor-Encoding-Schemes)
- [llama.cpp Q8_0 reference quantizer](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c)
- [llama.cpp Q8_0 block definition](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h)
- [llama.cpp backend feature matrix](https://github.com/ggml-org/llama.cpp/wiki/Feature-matrix)
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)
- [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://proceedings.mlr.press/v202/xiao23c.html)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [PyTorch quantization documentation](https://docs.pytorch.org/docs/stable/quantization.html)

Because llama.cpp model recipes evolve, inspect the current source before
presenting an exact per-tensor `Q4_K_M` allocation as permanent behavior.
