# LLM Quantization — Hızlı Referans

## Ana formül

```text
q     = clip(round(x / scale) + zero_point, qmin, qmax)
x_hat = scale × (q - zero_point)
error = x_hat - x
```

Symmetric `b`-bit:

```text
qmax = 2^(b-1)-1, qmin = -qmax
scale = max(abs(x))/qmax
zero_point = 0
```

Asymmetric unsigned `b`-bit:

```text
qmin=0, qmax=2^b-1
scale=(xmax-xmin)/(qmax-qmin)
zero_point=round(qmin-xmin/scale)
```

## Hangi sayı quantize ediliyor?

| Hedef | Kazanç | Dikkat |
| --- | --- | --- |
| Weight | Model dosyası ve RAM/VRAM | Kalite, kernel desteği |
| Activation | Compute + ara bellek | Outlier, dynamic range |
| KV cache | Uzun context/concurrency belleği | Long-context kalite |

```text
W4A16 = yaklaşık 4-bit weight, 16-bit activation
W8A8  = 8-bit weight, 8-bit activation
```

## Granularity

```text
per-tensor -> per-channel -> group-wise
az metadata / daha çok error  ---->  çok metadata / daha az error
```

Packed INT4 + bir FP16 scale:

```text
effective bpw = 4 + 16/group_size
group 64 -> 4.25, 32 -> 4.50, 16 -> 5.00, 8 -> 6.00 bpw
```

Q8_0:

```text
32 signed int8 code + 1 FP16 scale
32 B payload + 2 B metadata = 34 B
34×8/32 = 8.5 bpw

d = max(abs(block))/127
q = clip(round(x/d), -127, 127)
x_hat = q × stored_FP16(d)
```

```text
Q8_0 = weight tensor encoding
W8A8 = 8-bit weight + 8-bit activation compute hedefi
GGUF = container
```

## Workflow'lar ve yöntemler

| İsim | Tek cümle |
| --- | --- |
| PTQ | Training bittikten sonra quantize et |
| Dynamic PTQ | Activation qparam'ını runtime'da hesapla |
| Static PTQ | Calibration ile qparam'ı önceden sabitle |
| QAT | Training forward'ında fake quantization göster |
| GPTQ | Hessian/second-order bilgiyle output error'ını telafi et |
| AWQ | Önemli activation channel weight'lerini daha iyi koru |
| SmoothQuant | Activation outlier zorluğunu weight'e taşı; W8A8 hedefle |
| NF4 | Normal weight dağılımına uygun non-uniform 4-bit codebook |
| QLoRA | Frozen NF4 base + trainable LoRA adapter |

## GGUF isimleri

| Etiket | Pratik yorum |
| --- | --- |
| `Q4_0` | Basit/legacy 32-weight block 4-bit quant |
| `Q4_K_S` | Daha küçük Q4 K-quant recipe |
| `Q4_K_M` | Dengeli mixed Q4 K-quant başlangıcı |
| `Q5_K_M` | Daha fazla kalite, daha fazla bellek |
| `Q6_K` | Yüksek precision K-quant |
| `Q8_0` | 32 code + FP16 scale; 8.5 bpw, yüksek kalite baseline |
| `IQ4_XS` | Nonlinear/importance-aware aile; backend hızını ölç |

```text
GGUF   = container
Q4_K   = tensor encoding
Q4_K_M = model-level mixed recipe
```

Q4_K:

```text
256 weight
128 B 4-bit codes + 12 B scale/min codes + 4 B FP16 metadata
= 144 B = 4.5 bits/weight
```

## Seçim akışı

1. Model + runtime + KV cache belleğini hesapla.
2. Backend'in quant kernel desteğini kontrol et.
3. Local GGUF için `Q4_K_M` ile başla.
4. Kalite yetersizse `Q5_K_M`/`Q6_K`/`Q8_0` karşılaştır.
5. Bellek yetersizse `Q4_K_S`/Q3/IQ veya küçük model dene.
6. Aynı inputlarla kalite, peak memory, prompt t/s ve generation t/s ölç.

## Beş uyarı

- “4-bit” demek gerçek modelin tam 4.0 bpw olduğu anlamına gelmez.
- `torch.int8` içinde küçük değer saklamak gerçek INT4 packing değildir.
- Daha az bit her backend'de daha hızlı değildir.
- Weight quantization KV-cache büyümesini çözmez.
- Birkaç güzel generation örneği kalite benchmark'ı değildir.
- Q8_0 quantized weight demek activation ve KV cache de 8-bit demek değildir.
- Q8_0 lossless değildir; rounding ve FP16 scale küçük error üretir.

## Çalıştır

```bash
.venv/bin/python quantization/by_hand.py
.venv/bin/python quantization/q8_0.py
.venv/bin/python quantization/q8_demo.py --show-layers
.venv/bin/python quantization/demo.py --show-layers
.venv/bin/python quantization/qat_demo.py
.venv/bin/python quantization/nf4.py
.venv/bin/python quantization/advanced_methods.py
```
