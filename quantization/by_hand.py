"""by_hand.py — quantization'ı kalemle kontrol edilebilen sayılarla öğren.

Çalıştır:
    PYTHONPATH=quantization .venv/bin/python quantization/by_hand.py

Sekiz bölüm:
  1. symmetric quantize -> dequantize -> error
  2. asymmetric quantization ve zero-point
  3. outlier: per-tensor yerine küçük group neden işe yarar?
  4. gerçek INT4 packing: iki sayı bir byte
  5. quantized Linear çıktısı nasıl değişir?
  6. group size: hata ile metadata arasındaki değiş tokuş
  7. Q4_K neden tam 4.5 bits per weight?
  8. QAT için straight-through estimator (STE)

Her önemli sonuç `assert` ile doğrulanır. Kod çalışmayı bitirirse, ekrana
yazdırılan matematik de testlerden geçmiş demektir.
"""

import torch

from core import (
    fake_quantize_ste,
    pack_int4,
    quantization_error,
    quantize_tensor,
    unpack_int4,
)
from q4_k import (
    pack_scale_min_6bit,
    q4_k_layout_report,
    quantize_q4_k,
    unpack_scale_min_6bit,
)


torch.set_printoptions(precision=4, sci_mode=False)


def rule(title: str) -> None:
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)


def row(label: str, values: torch.Tensor) -> None:
    print(f"  {label:14s} {values.tolist()}")


# ===========================================================================
rule("1. Symmetric quantization: q = round(x/scale), x_hat = q*scale")
# ---------------------------------------------------------------------------

# Öğretim için 7 symmetric seviye: -3,-2,-1,0,1,2,3.
x = torch.tensor([-1.2, -0.7, 0.1, 0.8, 1.1])
qmin, qmax = -3, 3
scale = x.abs().max() / qmax                       # 1.2 / 3 = 0.4
q = torch.sign(x / scale) * torch.floor((x / scale).abs() + 0.5)
q = q.clamp(qmin, qmax).to(torch.int8)
x_hat = q.float() * scale
error = x_hat - x
mse = error.square().mean()

row("x", x)
print(f"  scale          max(|x|)/3 = {scale.item():.1f}")
row("q", q)
row("x_hat", x_hat)
row("error", error)
print(f"  MSE            {mse.item():.3f}")

assert torch.equal(q, torch.tensor([-3, -2, 0, 2, 3], dtype=torch.int8))
assert torch.allclose(x_hat, torch.tensor([-1.2, -0.8, 0.0, 0.8, 1.2]))
assert abs(mse.item() - 0.006) < 1e-7

# Calibration bu aralığı sabitledikten sonra gelen 1.6 temsil edilemez.
new_value = torch.tensor(1.6)
new_q = torch.round(new_value / scale).clamp(qmin, qmax)
new_hat = new_q * scale
print(
    f"\n  clipping: yeni x=1.6 -> q={new_q.item():.0f} -> "
    f"x_hat={new_hat.item():.1f}; 0.4 bilgi kaybı"
)
assert new_q.item() == 3 and abs(new_hat.item() - 1.2) < 1e-6


# ===========================================================================
rule("2. Asymmetric quantization: zero-point gerçek sıfırı temsil eder")
# ---------------------------------------------------------------------------

# 4-bit unsigned kodlar 0..15. Aralık [-1, 3] sıfırı zaten içeriyor.
x2 = torch.tensor([-1.0, 0.0, 1.0, 3.0])
qt2 = quantize_tensor(
    x2, bits=4, group_size=4, symmetric=False, pack=False
)
q2 = qt2.integer_values().flatten()[:4]
x2_hat = qt2.dequantize()

scale2 = qt2.scales.item()
zp2 = int(qt2.zero_points.item())
print("  qmin=0, qmax=15")
print(f"  scale          (3-(-1))/15 = {scale2:.6f}")
print(f"  zero_point     round(0 - (-1)/scale) = {zp2}")
row("x", x2)
row("q", q2)
row("x_hat", x2_hat)
print("  x=0 tam olarak q=zero_point koduna gider ve tekrar 0 olur.")

assert zp2 == 4
assert q2[1].item() == zp2
assert x2_hat[1].item() == 0.0
assert (x2_hat - x2).abs().max().item() < 0.14


# ===========================================================================
rule("3. Outlier ve granularity: küçük group küçük değerleri kurtarır")
# ---------------------------------------------------------------------------

x3 = torch.tensor([[0.1, 0.2, 0.3, 10.0]])
one_group = quantize_tensor(x3, bits=4, group_size=4, symmetric=True)
two_groups = quantize_tensor(x3, bits=4, group_size=2, symmetric=True)

one_hat = one_group.dequantize()
two_hat = two_groups.dequantize()
one_mse = quantization_error(x3, one_group)["mse"]
two_mse = quantization_error(x3, two_groups)["mse"]

row("orijinal", x3.flatten())
row("group=4", one_hat.flatten())
row("group=2", two_hat.flatten())
print(f"  MSE group=4: {one_mse:.6f}")
print(f"  MSE group=2: {two_mse:.6f}")
print("  10.0 outlier'ı artık yalnız kendi iki-elemanlı group'unu etkiliyor.")

assert two_mse < one_mse
assert abs(two_hat[0, 0].item() - 0.1) < abs(one_hat[0, 0].item() - 0.1)


# ===========================================================================
rule("4. INT4 packing: iki signed değer gerçekten bir uint8 byte olur")
# ---------------------------------------------------------------------------

signed_values = torch.tensor([[-7, -1, 0, 7]], dtype=torch.int8)
packed = pack_int4(signed_values)
unpacked = unpack_int4(packed)

row("INT4 values", signed_values.flatten())
print("  two's complement nibbles: -7=0x9, -1=0xF, 0=0x0, 7=0x7")
print("  low nibble first -> packed bytes:", [f"0x{b:02X}" for b in packed.flatten()])
row("unpacked", unpacked.flatten())
print(
    f"  {signed_values.numel()} int8 slots would use "
    f"{signed_values.numel()} bytes; packed INT4 uses {packed.numel()} bytes."
)

assert packed.flatten().tolist() == [0xF9, 0x70]
assert torch.equal(unpacked, signed_values)
assert packed.numel() * 2 == signed_values.numel()


# ===========================================================================
rule("5. Quantized weight bir Linear layer çıktısını nasıl değiştirir?")
# ---------------------------------------------------------------------------

W = torch.tensor(
    [
        [0.10, 0.20, -0.30, 0.40],
        [1.20, -0.80, 0.50, -0.10],
    ]
)
input_x = torch.tensor([1.0, 2.0, -1.0, 0.5])
Wq = quantize_tensor(W, bits=4, group_size=4, symmetric=True)
W_hat = Wq.dequantize()
y = W @ input_x
y_hat = W_hat @ input_x

print("  W:")
print(W)
print("  dequantized W_hat:")
print(W_hat)
row("y = W x", y)
row("y_hat", y_hat)
row("output error", y_hat - y)
print("  Quantization weight'leri değiştirir; bu hata matrix product'e taşınır.")

assert Wq.data.dtype == torch.uint8
assert y_hat.shape == y.shape
assert (y_hat - y).abs().max().item() < 0.2


# ===========================================================================
rule("6. Group size küçülür: error azalabilir, scale metadata büyür")
# ---------------------------------------------------------------------------

torch.manual_seed(7)
weights = torch.randn(16, 64) * 0.2
weights[0, 0] = 3.0                                  # görünür bir outlier

print(f"  {'group':>8} | {'MSE':>11} | {'payload B':>9} | {'metadata B':>10} | {'bpw':>7}")
previous_mse = None
for group_size in (64, 32, 16, 8):
    qt = quantize_tensor(
        weights,
        bits=4,
        group_size=group_size,
        symmetric=True,
        scale_dtype=torch.float16,
    )
    metrics = quantization_error(weights, qt)
    print(
        f"  {group_size:>8} | {metrics['mse']:>11.8f} | "
        f"{qt.payload_nbytes:>9} | {qt.metadata_nbytes:>10} | "
        f"{qt.effective_bits_per_weight:>7.3f}"
    )
    if previous_mse is not None:
        # Smaller groups have at least as much freedom on this fixed tensor.
        assert metrics["mse"] <= previous_mse + 1e-10
    previous_mse = metrics["mse"]

print("  Payload hep 4-bit; daha çok group daha çok FP16 scale demektir.")


# ===========================================================================
rule("7. Q4_K: 256 weight için 128+12+2+2 = 144 byte = 4.5 bpw")
# ---------------------------------------------------------------------------

torch.manual_seed(3)
block = torch.randn(256)
q4k = quantize_q4_k(block)
q4k_hat = q4k.dequantize()

print(" ", q4_k_layout_report(q4k))
print("  integer codes : 256*4 bit = 128 byte")
print("  8 scale + 8 min: 16*6 bit = 96 bit = 12 byte")
print("  d + dmin      : 2 FP16 = 4 byte")
print("  total         : 144 byte * 8 / 256 = 4.5 bits/weight")
print(f"  educational min/max fitter MSE: {(q4k_hat-block).square().mean().item():.6f}")

assert q4k.storage_nbytes == 144
assert q4k.effective_bits_per_weight == 4.5

# 6-bit metadata packing kendi başına da kayıpsız olmalı.
s_codes = torch.tensor([[0, 1, 2, 3, 15, 31, 47, 63]], dtype=torch.uint8)
m_codes = torch.tensor([[63, 47, 31, 15, 3, 2, 1, 0]], dtype=torch.uint8)
metadata = pack_scale_min_6bit(s_codes, m_codes)
s_back, m_back = unpack_scale_min_6bit(metadata)
assert metadata.numel() == 12
assert torch.equal(s_codes, s_back) and torch.equal(m_codes, m_back)

print(
    "\n  Not: Bu dosya Q4_K storage yapısını öğretir. llama.cpp'nin optimized\n"
    "  parameter search'ünü, importance matrix'ini veya GGUF writer'ını kopyalamaz.\n"
    "  Q4_K_M ise bunun da üstünde, tensorlere farklı tür seçen model recipe'sidir."
)


# ===========================================================================
rule("8. QAT: forward quantized, backward gradient düz geçer (STE)")
# ---------------------------------------------------------------------------

master_weight = torch.tensor([0.12, -0.37, 0.88], requires_grad=True)
forward_weight = fake_quantize_ste(
    master_weight, bits=4, group_size=3, symmetric=True
)
loss = forward_weight.sum()
loss.backward()

row("FP master", master_weight.detach())
row("forward quant", forward_weight.detach())
row("gradient", master_weight.grad)
print(
    "  Forward yuvarlanmış değerleri gördü; backward d(output)/d(weight)=1\n"
    "  varsayımıyla master weight'e gradient ulaştırdı."
)

assert not torch.equal(master_weight.detach(), forward_weight.detach())
assert torch.equal(master_weight.grad, torch.ones_like(master_weight))


print("\nBütün elle-kontrol testleri geçti.\n")
