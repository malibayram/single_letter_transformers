# Quantization — Sıfırdan, Basit ve Gerçek Uygulamayla Türkçe Anlatım

Bu belge yalnızca 90 dakikalık bir ders için değildir. Amaç, quantization
konusunu okuyup geçmek yerine:

- matematiğini elle yapmak,
- integer kodları görmek,
- iki INT4 değeri gerçekten tek byte'a paketlemek,
- bir `nn.Linear` katmanının float weight'ini kaldırmak,
- PTQ ve QAT akışlarını çalıştırmak,
- TinyQwen modelinde loss ve generation farkını ölçmek,
- `Q4_K_M`, NF4, GPTQ, AWQ ve SmoothQuant isimlerini doğru yere koymak

için uzun süre kullanılabilecek bir çalışma rehberi sağlamaktır.

Kodlar bilerek küçük tutulmuştur. Production framework'lerin yüzlerce
optimizasyon satırı arasında kaybolmadan, her adımın neden var olduğunu
görebilirsin.

---

## İçindekiler

1. [Quantization hangi problemi çözer?](#1-quantization-hangi-problemi-çözer)
2. [Önce üç kavramı ayıralım](#2-önce-üç-kavramı-ayralım)
3. [Ana formül](#3-ana-formül)
4. [Rounding ve clipping](#4-rounding-ve-clipping)
5. [Symmetric quantization](#5-symmetric-quantization)
6. [Asymmetric quantization ve zero-point](#6-asymmetric-quantization-ve-zero-point)
7. [Granularity: per-tensor, per-channel, group-wise](#7-granularity-per-tensor-per-channel-group-wise)
8. [Group size neden kaliteyi ve boyutu birlikte değiştirir?](#8-group-size-neden-kaliteyi-ve-boyutu-birlikte-değiştirir)
9. [Quantization ile packing aynı şey değildir](#9-quantization-ile-packing-aynı-şey-değildir)
10. [`core.py` adım adım](#10-corepy-adım-adım)
11. [`QuantizedLinear`: gerçek düşük-bit storage](#11-quantizedlinear-gerçek-düşük-bit-storage)
12. [PTQ workflow'ları](#12-ptq-workflowları)
13. [QAT ve straight-through estimator](#13-qat-ve-straight-through-estimator)
14. [Weight, activation ve KV-cache quantization](#14-weight-activation-ve-kv-cache-quantization)
15. [W4A16 ve W8A8 nasıl okunur?](#15-w4a16-ve-w8a8-nasıl-okunur)
16. [GPTQ](#16-gptq)
17. [AWQ](#17-awq)
18. [SmoothQuant](#18-smoothquant)
19. [NF4 ve QLoRA](#19-nf4-ve-qlora)
20. [GGUF nedir, ne değildir?](#20-gguf-nedir-ne-değildir)
21. [Q4_0, K-quants ve I-quants](#21-q4_0-k-quants-ve-i-quants)
22. [Q4_K neden 4.5 bpw?](#22-q4_k-neden-45-bpw)
23. [Q4_K_M tam olarak nasıl düşünülmeli?](#23-q4_k_m-tam-olarak-nasıl-düşünülmeli)
24. [TinyQwen uygulaması](#24-tinyqwen-uygulaması)
25. [Bir quantization nasıl seçilir?](#25-bir-quantization-nasıl-seçilir)
26. [Kendi quantizer'ını geliştirirken kontrol listesi](#26-kendi-quantizerını-geliştirirken-kontrol-listesi)
27. [Yaygın hatalar ve debugging](#27-yaygın-hatalar-ve-debugging)
28. [Çalışma sırası](#28-çalışma-sırası)
29. [Terimler sözlüğü](#29-terimler-sözlüğü)
30. [Q8 ve Q8_0 derinlemesine](#30-q8-ve-q8_0-derinlemesine)
31. [Q8_0 gerçek Python implementation walkthrough](#31-q8_0-gerçek-python-implementation-walkthrough)

---

## 1. Quantization hangi problemi çözer?

Bir modelin büyük bölümü weight matrix'lerinden oluşur. Her weight FP32 ise
32 bit, yani 4 byte kullanır.

Örneğin bir milyar weight:

```text
1,000,000,000 × 4 byte = yaklaşık 4 GB
```

Aynı weight'ler saf 8-bit payload olsaydı:

```text
1,000,000,000 × 1 byte = yaklaşık 1 GB
```

Saf 4-bit payload:

```text
1,000,000,000 × 0.5 byte = yaklaşık 0.5 GB
```

Fakat gerçek quantized storage bundan biraz büyüktür. Çünkü integer kodların
yanında `scale`, bazen `zero_point`, minimum veya codebook metadata'sı gerekir.
Model recipe bazı hassas tensorleri daha yüksek precision'da da tutabilir.

Quantization üç potansiyel kazanç sağlar:

1. **Daha küçük model dosyası**
2. **Daha düşük RAM/VRAM kullanımı**
3. **Uygun kernel varsa daha yüksek inference hızı**

Üçüncüsü otomatik değildir. Kernel yoksa düşük-bit weight önce float'a
açılabilir ve Python örneğinde olduğu gibi daha yavaş bile çalışabilir.

Quantization'ın bedeli:

```text
orijinal sayı -> en yakın temsil seviyesi -> küçük numerical error
```

Bu küçük hatalar bütün layer'lardan geçerken model davranışını etkileyebilir.
Amaç sıfır hata değil; kabul edilebilir kaliteyle büyük verim kazanmaktır.

---

## 2. Önce üç kavramı ayıralım

### Container

GGUF gibi bir container:

- tensor isimlerini,
- tensor shape'lerini,
- tensor byte'larını,
- tokenizer/config metadata'sını

bir dosyada taşır.

### Tensor encoding

Bir tensorün byte'lara nasıl çevrildiğini anlatır:

- INT8
- uniform INT4
- `Q4_0`
- `Q4_K`
- `IQ4_XS`
- NF4

### Model recipe

Bütün modelde hangi tensore hangi encoding uygulanacağını anlatır:

- `Q4_K_S`
- `Q4_K_M`
- `Q5_K_M`

`Q4_K_M` tek bir tensor dtype değildir. Modelde farklı tensorlerin farklı
encoding'ler kullanabildiği mixed-precision tariftir.

### Compute kernel

Packed weight ile matrix multiplication'ı hangi CPU/GPU kodunun yaptığıdır.
Storage format doğru olsa bile hızlı kernel yoksa hız kazanılmaz.

Bu dört katmanı ayır:

```text
GGUF file
  └── model recipe: Q4_K_M
        ├── tensor A: Q4_K
        ├── tensor B: Q6_K
        └── tensor C: F32/F16/başka tür

runtime kernel
  └── bu tensorlerle CPU/GPU computation yapar
```

---

## 3. Ana formül

Affine quantization:

```text
q = clip(round(x / scale) + zero_point, qmin, qmax)
```

Dequantization:

```text
x_hat = scale × (q - zero_point)
```

Hata:

```text
error = x_hat - x
```

Burada:

- `x`: float değer
- `q`: integer kod
- `x_hat`: yaklaşık geri açılmış değer
- `scale`: iki komşu integer seviyenin float dünyasındaki mesafesi
- `zero_point`: gerçek `0.0` değerinin integer kodu
- `qmin/qmax`: bit sayısının izin verdiği sınırlar

Bir sayı örneği:

```text
scale = 0.1
zero_point = 0
x = 0.34

q = round(0.34 / 0.1) = round(3.4) = 3
x_hat = 3 × 0.1 = 0.3
error = 0.3 - 0.34 = -0.04
```

Quantizer artık `0.34` saklamaz. `3` kodunu ve `0.1` scale'ını saklar.

---

## 4. Rounding ve clipping

### Rounding error

`0.34`, en yakın seviye olan `0.3`e gider. Aralık içinde olmasına rağmen
seviye tam eşleşmediği için hata vardır.

### Clipping error

Temsil aralığı `[-1.2, 1.2]`, yeni değer `1.6` olsun:

```text
round(1.6 / 0.4) = 4
qmax = 3
clip(4, -3, 3) = 3
x_hat = 3 × 0.4 = 1.2
```

Hata `-0.4` olur. Clipping genellikle büyük outlier değerlerde ciddi olabilir.

### Round davranışı

`2.5` tam ortada olduğunda farklı sistemler:

- ties-to-even: `2`
- half-away-from-zero: `3`

sonucunu verebilir. [`core.py`](core.py), örnekleri deterministic yapmak için
half-away-from-zero uygular:

```python
sign(x) * floor(abs(x) + 0.5)
```

Production format uyumluluğu gerekiyorsa hedef kernel'in rounding kuralı da
eşleşmelidir.

---

## 5. Symmetric quantization

Signed 4-bit teorik two's-complement aralığı:

```text
[-8, 7]
```

Bu eğitim kodu simetri için:

```text
[-7, 7]
```

kullanır. Böylece:

```text
qmax = 7
qmin = -7
zero_point = 0
scale = max(abs(x)) / 7
```

Örnek:

```text
x = [-1.0, -0.2, 0.0, 0.4, 0.9]
max_abs = 1.0
scale = 1/7 ≈ 0.142857
```

Kodlar:

```text
-1.0 / scale -> -7
-0.2 / scale -> -1.4 -> -1
 0.0 / scale -> 0
 0.4 / scale -> 2.8 -> 3
 0.9 / scale -> 6.3 -> 6
```

Avantaj:

- zero-point metadata yok;
- weight dağılımları çoğu zaman sıfır çevresinde;
- implementation basit.

Dezavantaj:

- yalnız pozitif veya kaymış dağılımda bazı seviyeler boşa gidebilir.

---

## 6. Asymmetric quantization ve zero-point

Unsigned 4-bit:

```text
qmin = 0
qmax = 15
```

Örnek aralık:

```text
xmin = -1
xmax = 3
scale = (3 - (-1)) / 15 = 4/15 ≈ 0.266667
zero_point = round(0 - (-1)/scale)
           = round(3.75)
           = 4
```

Gerçek sıfır:

```text
q = round(0/scale) + 4 = 4
x_hat = scale × (4 - 4) = 0
```

Tam olarak geri gelir.

`core.py`, range yalnız pozitif veya yalnız negatif olsa bile `0.0` değerini
aralığa ekler. Bunun nedeni geçerli bir zero-point elde etmektir:

```python
mins = minimum(observed_min, 0)
maxs = maximum(observed_max, 0)
```

Asymmetric quantization daha fazla metadata kullanır:

```text
her group: scale + zero_point
```

Bu nedenle weight quantization'da symmetric sık görülür; activation ve bazı
deployment formatlarında asymmetric tercih edilebilir.

---

## 7. Granularity: per-tensor, per-channel, group-wise

Bir scale kaç weight'i temsil ediyor?

### Per-tensor

Tek tensor, tek scale:

```text
W [4096, 4096] -> 1 scale
```

Bir outlier bütün matrix'in adım boyunu büyütebilir.

### Per-channel

Linear weight shape:

```text
[out_features, in_features]
```

Her output row için bir scale:

```text
4096 output row -> 4096 scale
```

Her neuron kendi range'ini kullanır.

### Group-wise

Her row'u daha küçük parçalara ayır:

```text
row uzunluğu 4096
group_size = 128
4096/128 = 32 scale per row
```

Outlier yalnız bulunduğu group'u etkiler.

### Per-token activation

Activation tensoründe her token için ayrı scale hesaplanabilir. Bu dynamic
range'e uyum sağlar, fakat runtime qparam hesabı gerektirir.

### Per-tensor mi group-wise mı?

Tek doğru cevap yoktur:

- büyük group: az metadata, fazla quantization error
- küçük group: çok metadata, az quantization error

---

## 8. Group size neden kaliteyi ve boyutu birlikte değiştirir?

Packed INT4 payload:

```text
her weight = 4 bit
```

Her group bir FP16 scale kullanıyorsa:

```text
scale overhead = 16 bit / group_size
effective bpw = 4 + 16/group_size
```

Tablo:

| Group | Payload bpw | Scale bpw | Total |
| ---: | ---: | ---: | ---: |
| 64 | 4 | 0.25 | 4.25 |
| 32 | 4 | 0.50 | 4.50 |
| 16 | 4 | 1.00 | 5.00 |
| 8 | 4 | 2.00 | 6.00 |

`by_hand.py` gerçek bir `[16,64]` weight ile şunu ölçer:

```text
group | MSE        | payload | metadata | bpw
64    | 0.00135883 | 512 B   | 32 B     | 4.25
32    | 0.00088203 | 512 B   | 64 B     | 4.50
16    | 0.00060086 | 512 B   | 128 B    | 5.00
8     | 0.00030878 | 512 B   | 256 B    | 6.00
```

Kalite arttı, fakat storage avantajı azaldı.

### Padding

Last dimension 35, group size 32:

```text
group 1: 32 gerçek weight
group 2: 3 gerçek + 29 padding
```

İkinci group yine payload ve scale kullanır. Küçük tensorlerde gerçek effective
bpw formülden daha yüksek olabilir.

`QuantizedTensor.original_shape` ve `padded_last_dim` bu farkı saklar.

---

## 9. Quantization ile packing aynı şey değildir

Şu tensor:

```python
q = torch.tensor([-7, -1, 0, 7], dtype=torch.int8)
```

değerleri dört bite sığsa bile PyTorch'ta:

```text
4 element × 1 byte = 4 byte
```

kullanır.

Gerçek INT4 packing iki **nibble**'ı bir byte'a koyar.

Two's-complement 4-bit:

```text
-7 -> 1001 -> 0x9
-1 -> 1111 -> 0xF
 0 -> 0000 -> 0x0
 7 -> 0111 -> 0x7
```

İlk değer low nibble:

```text
[-7, -1] -> 1111 1001 -> 0xF9
[ 0,  7] -> 0111 0000 -> 0x70
```

Artık:

```text
4 code -> 2 byte
```

### Kayıp nerede oldu?

Packing kayıpsızdır:

```text
integer codes -> bytes -> integer codes
```

Bilgi kaybı daha önce oluşmuştur:

```text
float -> integer code
```

Bu ayrım debugging için çok önemlidir.

---

## 10. `core.py` adım adım

Ana API:

```python
from core import quantize_tensor

qt = quantize_tensor(
    tensor,
    bits=4,
    group_size=32,
    symmetric=True,
    pack=True,
    scale_dtype=torch.float16,
)
```

### 1. Shape'i satırlara aç

```python
original_shape = tensor.shape
flat = tensor.reshape(rows, last_dim)
```

Linear `[out,in]` zaten satır düzenindedir. Daha yüksek boyutlu tensorlerde
leading dimension'lar `rows` içinde birleşir.

### 2. Son group için padding

```python
n_groups = ceil(last_dim / group_size)
padded_last_dim = n_groups * group_size
```

### 3. Qparam hesapla

Symmetric:

```python
max_abs = groups.abs().amax(-1)
scales = max_abs / qmax
```

Tamamen sıfır group:

```python
scale = 1
q = 0
x_hat = 0
```

Division-by-zero önlenir.

### 4. Quantize

```python
q = round(groups / scale)
q = clamp(q, qmin, qmax)
```

### 5. Pack

4-bit:

```python
data = pack_int4(q)
```

8-bit:

```python
data = q.to(torch.int8)
```

### 6. Float weight'i saklama

`QuantizedTensor` içinde yalnız:

- `data`
- `scales`
- optional `zero_points`
- shape/config metadata

bulunur.

### 7. Dequantize

```python
q = unpack_int4(data)
x_hat = (q - zero_point) * scale
```

Padding kesilir, original shape geri verilir.

### Storage ölçümü

```python
qt.payload_nbytes
qt.metadata_nbytes
qt.storage_nbytes
qt.effective_bits_per_weight
```

Bu sayılar tensorün gerçekten sakladığı PyTorch buffer byte'larından gelir.

---

## 11. `QuantizedLinear`: gerçek düşük-bit storage

Normal Linear:

```python
layer = nn.Linear(64, 32, bias=False)
layer.weight                  # Parameter [32,64], FP32
```

Quantized:

```python
qlayer = QuantizedLinear.from_float(
    layer,
    bits=4,
    group_size=32,
)
```

İçerik:

```text
qweight  : packed uint8
scales   : FP16
bias     : varsa float
weight   : YOK
```

Kontrol:

```python
assert not hasattr(qlayer, "weight")
assert qlayer.qweight.dtype == torch.uint8
```

### Forward neden dequantize ediyor?

PyTorch'un genel `F.linear` fonksiyonu bizim özel packed nibble düzenimizi
doğrudan kullanmaz. Okunabilir reference:

```python
weight_hat = qlayer.dequantize_weight(dtype=x.dtype)
return F.linear(x, weight_hat, bias)
```

Bu implementation:

- doğru low-bit storage gösterir,
- numerical output verir,
- her byte'ı inceletir,
- CPU'da her yerde çalışır.

Fakat:

- her forward weight'i açar,
- fused kernel değildir,
- production hızını temsil etmez.

Gerçek runtime kernel şu işleri birleştirebilir:

```text
load packed bytes
unpack vectorized
apply scales
multiply/accumulate
```

### Modelde katman değiştirme

LoRA'daki `inject()` gibi:

```python
report = quantize_model_linears(
    model,
    bits=4,
    group_size=32,
    exclude=("lm_head",),
)
```

Her seçili `nn.Linear`, `QuantizedLinear` olur.

TinyQwen'de `lm_head.weight` ile `embed_tokens.weight` aynıdır:

```python
model.lm_head.weight is model.embed_tokens.weight
```

Sadece `lm_head` değiştirilirse sharing bozulur. Bu nedenle demo onu dışarıda
bırakır.

---

## 12. PTQ workflow'ları

PTQ = **Post-Training Quantization**.

### Weight-only PTQ

```text
trained float model
  -> weight qparams hesapla
  -> weight'leri pack et
  -> inference
```

Activation float kalabilir:

```text
W4A16
```

Local LLM inference'da çok yaygındır; weight memory/bandwidth büyük bottleneck
olabilir.

### Dynamic activation quantization

Weight önceden quantized. Activation scale input geldiğinde hesaplanır:

```text
her batch/token:
  activation range ölç
  activation quantize et
  compute
```

Avantaj:

- değişen input dağılımına uyum.

Maliyet:

- runtime range/qparam hesabı.

### Static activation quantization

Önce representative calibration verisi:

```text
calibration samples
  -> observers min/max/statistics toplar
  -> scale'lar sabitlenir
  -> inference sırasında sabit qparams
```

Avantaj:

- daha az runtime qparam overhead;
- backend için daha kolay optimize edilebilir.

Risk:

- gerçek input dağılımı calibration'dan farklıysa clipping.

### Calibration dataset nasıl olmalı?

- gerçek deployment inputlarını temsil etmeli;
- chat template/tokenization aynı olmalı;
- sequence length dağılımı benzer olmalı;
- MoE modelde mümkün olduğunca farklı expert'leri uyarmalı;
- çok küçük veya tek tip olmamalı.

---

## 13. QAT ve straight-through estimator

QAT = **Quantization-Aware Training**.

QAT sırasında gerçek weight hâlâ FP32 master Parameter olabilir:

```text
FP master W
  -> fake quantize/dequantize
  -> W_hat ile forward
  -> loss
  -> gradient master W'ye
```

### Sorun: round differentiable değil

Matematiksel olarak rounding'in gradient'i neredeyse her yerde sıfırdır. Normal
gradient kullanılırsa weight hareket etmeyebilir.

### STE çözümü

```python
quantized = fake_quantize(weight.detach())
forward_weight = weight + (quantized - weight).detach()
```

Forward değer:

```text
weight + quantized - weight = quantized
```

Backward:

```text
detach içinden gradient gelmez
d(forward_weight)/d(weight) = 1
```

Bu yaklaşık gradient'e straight-through estimator denir.

### `QATLinear`

```python
qat = QATLinear.from_float(
    trained_linear,
    bits=4,
    group_size=32,
)
```

Training:

```python
loss = criterion(qat(x), y)
loss.backward()
optimizer.step()
```

Convert:

```python
packed = qat.to_quantized()
```

Convert sonrası:

- FP master weight atılır,
- packed `qweight` kalır,
- scale metadata kalır.

[`qat_demo.py`](qat_demo.py) deterministic binary classification görevinde
direct PTQ ile kısa QAT adaptasyonunu karşılaştırır.

### Gerçek QAT sistemlerinde ek parçalar

Bu basit örnekte olmayanlar:

- observer enable/disable
- qparam freeze
- activation fake quantization
- learned scale
- per-token/per-channel backend config
- warm-up schedule
- batch norm folding
- export graph conversion

Temel STE fikrini anladıktan sonra bu ayrıntılar daha anlamlı olur.

---

## 14. Weight, activation ve KV-cache quantization

### Weight

Training sonrası çoğunlukla sabit:

```text
Linear/Embedding matrix'leri
```

Fayda:

- model storage ve RAM/VRAM düşer.

### Activation

Inputa bağlı, forward sırasında üretilir:

```text
hidden states, Linear inputs/outputs, attention ara değerleri
```

Fayda:

- uygun low-bit GEMM ile compute hızlanabilir.

Zorluk:

- activation outlier'ları;
- inputa göre değişen range.

### KV cache

Autoregressive generation sırasında eski tokenların attention Key/Value
tensorleri saklanır.

Yaklaşık büyüme:

```text
layers × sequence_length × kv_heads × head_dim × 2(K,V) × bytes
```

Weight quantization model dosyasını küçültse de KV cache context length ile
büyümeye devam eder. Uzun context veya çok concurrent user olduğunda KV-cache
quantization ayrı değerlendirilmelidir.

---

## 15. W4A16 ve W8A8 nasıl okunur?

Genel etiket:

```text
W<weight bits>A<activation bits>
```

### W4A16

```text
weight yaklaşık 4-bit
activation FP16/BF16 gibi 16-bit
```

Weight-only inference için yaygın.

### W8A8

```text
weight 8-bit
activation 8-bit
```

Uygun INT8 matrix multiplication kernel'iyle compute hızını hedefler.

### Etiket neyi söylemez?

- symmetric/asymmetric
- group size
- scale dtype
- per-tensor/per-channel/per-token
- accumulators INT32 mi float mı
- hangi layer'lar hariç
- kernel/backend

İki model aynı `W4A16` etiketini taşıyıp farklı kalite ve hız verebilir.

---

## 16. GPTQ

Naive quantization her weight'i bağımsız en yakın seviyeye koyar:

```text
minimize |w - q(w)|
```

Fakat önemli olan tek weight hatası değil, layer output hatasıdır:

```text
minimize ||W X - Q X||²
```

Burada `X` calibration activation'larıdır.

GPTQ:

- approximate second-order/Hessian bilgisi kullanır;
- weight'leri sıra ile quantize eder;
- mevcut weight'teki hatayı kalan weight'lere taşır;
- output error'ı azaltmaya çalışır.

`advanced_methods.py` tek weight row için:

```text
H ≈ 2 X X^T
H^-1 Cholesky factor
```

hesaplar. Bir weight quantize edilince:

```text
error = (w_i - q_i) / Hinv[i,i]
remaining_weights -= error × Hinv[i,i:]
```

Toy sonuç:

```text
naive output MSE = 0.298211
GPTQ-like MSE    = 0.222547
```

Full GPTQ bundan daha kapsamlıdır:

- layer/block batching
- group scale
- act-order/permutation
- damping
- memory management
- serialized format
- CUDA kernel

Bu klasör fikir matematiğini gerçek sayılarla gösterir, production GPTQ
converter'ı olduğunu iddia etmez.

---

## 17. AWQ

AWQ = **Activation-aware Weight Quantization**.

Ana fikir:

> Bütün weight'ler eşit önemli değildir. Büyük/önemli activation üreten
> channel'ların weight quantization hatası output'u daha fazla etkiler.

Calibration inputlarından channel importance:

```python
importance = mean(abs(X), dim=tokens)
```

Paired scale:

```text
X W^T = (X / s) (W * s)^T
```

Weight'in önemli sütununu büyütürsek quantizer o sütuna daha çok resolution
verebilir. Input channel tersine bölündüğü için float function değişmez.

Toy kod alpha arar:

```text
s = importance^alpha
alpha ∈ [0,1]
```

Her candidate için scaled weight quantize edilir, calibration output MSE
ölçülür.

Toy sonuç:

```text
naive held-out MSE = 0.141348
AWQ-like MSE       = 0.059957
```

Full AWQ:

- gerçek LLM layer'ları,
- calibration sample seçimi,
- clipping/search,
- group quantization,
- hardware-friendly kernels

gibi ayrıntılar içerir.

---

## 18. SmoothQuant

Activation outlier bir per-tensor INT8 scale'ı büyütürse küçük activation'lar
az seviyeye sıkışır.

SmoothQuant şu eşitliği kullanır:

```text
Y = X W^T
  = (X / s) (W * s)^T
```

Channel scale:

```text
s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)
```

`alpha` yükün ne kadarının activation'dan weight'e taşınacağını belirler.

Quantization'dan önce iki expression matematiksel olarak eşittir. Dönüşüm:

- activation range'i düzleştirir,
- weight range'ini büyütebilir,
- toplam W8A8 error için daha iyi denge bulur.

Toy sonuç, bilerek büyük activation outlier'larıyla:

```text
naive W8A8 MSE    = 0.949966
smoothed W8A8 MSE = 0.123037
```

SmoothQuant özellikle weight-only değil, weight + activation quantization
problemini hedefler.

---

## 19. NF4 ve QLoRA

### Uniform INT4

Seviyeler eşit aralıklı:

```text
-1.0, -0.857, ..., 0, ..., 0.857, 1.0
```

### NF4

NF4 = **NormalFloat4**. Normal dağılıma yakın weight'lerin sık bulunduğu
bölgelerde daha çok seviye kullanır.

16 codebook değeri:

```text
[-1.0000, -0.6962, -0.5251, -0.3949,
 -0.2844, -0.1848, -0.0911,  0.0000,
  0.0796,  0.1609,  0.2461,  0.3379,
  0.4407,  0.5626,  0.7230,  1.0000]
```

`nf4.py`:

1. 64-weight block alır.
2. `absmax` FP16 scale saklar.
3. Weight'i `[-1,1]` normalize eder.
4. En yakın codebook index'ini bulur.
5. Index'i 4-bit pack eder.

Storage:

```text
payload 4 bit
scale 16/64 = 0.25 bit
total 4.25 bpw
```

Normal sample:

```text
uniform INT4 MSE = 0.012263
NF4 MSE          = 0.008322
```

Uniform sample:

```text
uniform INT4 MSE = 0.001636
NF4 MSE          = 0.002739
```

NF4 distribution assumption'dır; her veri için sihirli değildir.

### QLoRA

QLoRA:

- frozen NF4 base model,
- double quantization,
- paged optimizer,
- trainable LoRA adapter

birleşimidir.

Bu repository'de:

- [`quantization/nf4.py`](nf4.py) NF4 fikrini,
- [`lora/`](../lora/) LoRA adapter'ını

ayrı ve okunabilir biçimde gösterir. Full QLoRA stack için fused backend ve
optimizer memory sistemi de gerekir.

---

## 20. GGUF nedir, ne değildir?

GGUF:

- model tensorlerini,
- tensor encoding türlerini,
- architecture metadata'sını,
- tokenizer metadata'sını

taşıyan container formatıdır.

GGUF:

- tek başına quantization algoritması değildir;
- her tensorün aynı bit width'te olduğunu garanti etmez;
- tek başına hız garantisi vermez.

Dosya adı:

```text
Model-8B-Instruct-Q4_K_M.gguf
```

şunu söyler:

- model ailesi/size/instruct çeşidi,
- GGUF container,
- `Q4_K_M` model quantization recipe etiketi.

Gerçek tensor listesini görmek için GGUF inspection tool veya runtime logları
kullanılmalıdır.

---

## 21. Q4_0, K-quants ve I-quants

### Q4_0

Legacy/simple block quant ailesi:

- 32-weight block,
- signed-ish 4-bit codes,
- block scale.

Basit ve yaygın; `Q4_K` ile aynı düzen değildir.

### K-quants

256-weight super-block içinde:

- alt block'lar,
- alt block scale/min codes,
- super-block metadata

kullanır.

Yaygın:

- `Q2_K`
- `Q3_K`
- `Q4_K`
- `Q5_K`
- `Q6_K`

Nominal bit sayısı tek başına effective bpw değildir.

Resmi llama.cpp encoding tablosundaki değerler:

| Type | Yaklaşık bpw |
| --- | ---: |
| `Q2_K` | 2.5625 |
| `Q3_K` | 3.4375 |
| `Q4_K` | 4.5 |
| `Q5_K` | 5.5 |
| `Q6_K` | 6.5625 |

### I-quants

I-quant ailesi:

- nonlinear codebook,
- importance matrix'ten yararlanabilen düşük-bit yöntemler

ile daha düşük boyutta kaliteyi korumayı hedefler.

Örnek:

- `IQ4_NL`
- `IQ4_XS`
- `IQ3_S`
- `IQ2_XS`

Backend desteği kritik olabilir. Benzer boyuttaki K-quant'tan daha iyi kalite
verip daha yavaş çalışabilir. Her cihazda benchmark gerekir.

---

## 22. Q4_K neden 4.5 bpw?

Bir super-block:

```text
256 weight
8 sub-block × 32 weight
```

Payload:

```text
256 × 4 bit = 1024 bit = 128 byte
```

Her sub-block için:

```text
8 scale code × 6 bit = 48 bit
8 min code   × 6 bit = 48 bit
toplam = 96 bit = 12 byte
```

Super-block:

```text
d    FP16 = 2 byte
dmin FP16 = 2 byte
```

Toplam:

```text
128 + 12 + 2 + 2 = 144 byte
144 × 8 / 256 = 4.5 bits per weight
```

### `q4_k.py` ne yapar?

- 4-bit q codes pack eder.
- Her 64'lü bölümde bir 32'lik sub-block low nibble, diğeri high nibble olur.
- 16 adet 6-bit scale/min code'u 12 byte'a pack eder.
- FP16 `d`/`dmin` saklar.
- `x_hat = d*scale_code*q - dmin*min_code` açılımını yapar.

### Ne yapmaz?

- llama.cpp optimized qparam search'ünü kopyalamaz.
- importance matrix kullanmaz.
- GGUF dosyası yazmaz.
- byte-for-byte llama.cpp output garantisi vermez.

Storage yapısı gerçektir; qparam fitting eğitim amaçlı basitleştirilmiştir.

---

## 23. Q4_K_M tam olarak nasıl düşünülmeli?

Yanlış:

> Bütün model weight'leri tam 4 bit.

Doğru:

> Çoğu uygun tensore Q4_K tabanı uygulayan, bazı hassas tensorleri daha yüksek
> precision'a çıkaran model-level mixed recipe.

Örnek zihinsel model:

```text
attention q/k/v tensorleri -> bazıları Q5_K/Q6_K olabilir
ffn_down tensorleri        -> layer'a göre daha yüksek olabilir
output tensor              -> daha yüksek precision olabilir
norm 1D tensorleri         -> quantize edilmeyebilir
uyumsuz shape              -> fallback type olabilir
```

Exact seçim:

- architecture
- model size/type
- GQA oranı
- MoE expert sayısı
- layer position
- tensor shape
- imatrix
- llama.cpp version

ile değişebilir.

Bu nedenle:

```text
Q4_K_M file size != parameter_count × 4/8
```

### S/M/L

Genel sezgi:

- `S`: smaller recipe
- `M`: medium/balanced mixed recipe
- `L`: bulunduğu ailelerde higher-quality/larger taraf

Fakat her quant ailesinde her suffix yoktur ve exact recipe source code'dan
kontrol edilmelidir.

---

## 24. TinyQwen uygulaması

Çalıştır:

```bash
.venv/bin/python quantization/demo.py --show-layers
```

Model:

```text
19,584 unique Parameter
2 transformer block
14 attention/MLP Linear
```

Quantize edilen:

```text
18,432 Linear weight
```

FP32 kalan:

- tied embedding/lm_head
- norm weight'leri

### Linear storage

`group_size=32`, FP16 scale:

```text
INT8:
73,728 byte FP32 -> 19,584 byte
8.5 bpw

INT4:
73,728 byte FP32 -> 10,368 byte
4.5 bpw
```

### Bütün model weight storage

```text
FP32  78,336 byte
INT8  24,192 byte
INT4  14,976 byte
```

INT4 neden 9,792 byte (`19584/2`) değil?

- bazı weight'ler FP32 kaldı;
- her group FP16 scale sakladı.

### Numerical kalite

Sabit 64 corpus window:

```text
FP32 loss  0.79982
INT8 loss  0.79810
INT4 loss  1.43681
```

INT8 küçük numerical drift ile baseline'a yakın kaldı. Basit INT4 bu minik
modelde loss'u belirgin artırdı.

Bu şunları öğretir:

- model size küçülmesi gerçektir;
- kalite kaybı mümkündür;
- bit sayısı seçimdir, garanti değil;
- GPTQ/AWQ/imatrix/mixed precision neden geliştirilmiştir.

### State dict round-trip

Demo:

1. quantized model state dict'ini memory buffer'a yazar;
2. aynı quantized module yapısını yeniden kurar;
3. packed buffer'ları yükler;
4. logits farkının sıfır olduğunu assert eder.

Bu, storage'ın yalnız geçici hesap olmadığını gösterir.

---

## 25. Bir quantization nasıl seçilir?

### 1. Bütçeyi hesapla

Yalnız model dosyası değil:

- runtime workspace
- KV cache
- batch/concurrency
- GPU offload buffer
- OS/uygulama payı

hesaba katılmalıdır.

### 2. Backend'i kontrol et

CPU, Metal, CUDA, ROCm, Vulkan, SYCL:

- farklı type desteği,
- farklı optimized kernel,
- farklı prompt/generation performansı

verebilir.

### 3. Dengeli başlangıç

Local GGUF için:

```text
önce Q4_K_M dene
```

makul pratik başlangıçtır.

### 4. Kalite düşükse

```text
Q5_K_M -> Q6_K -> Q8_0
```

tarafına kıyasla.

### 5. Bellek yetmiyorsa

```text
Q4_K_S / Q3 / IQ seçenekleri
veya daha küçük base model
```

denenebilir.

Bazen daha küçük ama aşırı quantized büyük model yerine daha temiz quantized
küçük model daha iyi olabilir.

### 6. Göreve özel ölç

- language modeling: perplexity
- QA: exact match/F1
- code: unit test pass rate
- classification: accuracy/F1
- generation: human/rubric evaluation

### 7. Sistem ölç

- peak RAM/VRAM
- load time
- prompt tokens/s
- generation tokens/s
- first-token latency
- energy/power gerekiyorsa

Tek bir dosya boyutu tablosu seçim için yeterli değildir.

---

## 26. Kendi quantizer'ını geliştirirken kontrol listesi

### Numerical

- `scale == 0` group ne olacak?
- rounding kuralı ne?
- clipping sınırları doğru mu?
- zero-point geçerli range'de mi?
- padding min/max istatistiğini bozuyor mu?
- scale hangi dtype'ta saklanıyor?
- dequantized shape tam original mı?

### Storage

- 4-bit kodlar gerçekten pack edildi mi?
- padding byte'ları sayıldı mı?
- scale/zero-point metadata sayıldı mı?
- tied weight iki kez mi sayıldı?
- state dict içinde float weight yanlışlıkla kaldı mı?

### Model

- hangi layer'lar quantize edildi?
- norm/embedding/output hariç mi?
- weight tying bozuldu mu?
- bias ne precision'da?
- activation dtype ne?

### Evaluation

- FP32 baseline aynı checkpoint mi?
- aynı input batch mi?
- generation için aynı seed mi?
- loss/perplexity finite mi?
- task metric var mı?
- sadece birkaç güzel output seçilmedi mi?

### Performance

- warm-up yapıldı mı?
- aynı thread/backend mi?
- prompt ve generation ayrı mı?
- Python dequant overhead'i production kernel diye sunuluyor mu?
- model gerçekten memory'e sığıyor mu?

---

## 27. Yaygın hatalar ve debugging

### “INT4 yaptım ama tensor hâlâ aynı boyutta”

Muhtemelen code'ları `torch.int8` içinde saklıyorsun. Değer range'i küçük olsa
da her element 1 byte'tır.

Çözüm:

```python
pack_int4()
```

ile iki nibble'ı bir byte'a koy.

### “Dequantization shape yanlış”

- padded last dimension
- number of groups
- pack/unpack sıra düzeni

kontrol et.

Round-trip testi:

```python
assert equal(q_codes, unpack(pack(q_codes)))
```

### “Asymmetric positive tensor kötü quantize oluyor”

Real zero'yu range'e dahil et:

```text
xmin = min(observed_min, 0)
xmax = max(observed_max, 0)
```

Yoksa zero-point range dışına düşebilir.

### “Daha küçük group, dosyayı beklediğimden çok büyüttü”

Her group yeni scale/zero-point metadata getirir. Effective bpw hesapla.

### “INT4 demo INT8'den çok daha kötü”

Olası ve normal:

- model çok küçük/hassas;
- naive round-to-nearest kullanılıyor;
- sensitive layer'lar korunmuyor;
- calibration yok;
- group size büyük.

GPTQ/AWQ/mixed precision veya daha yüksek bit dene.

### “Packed model daha yavaş”

Bu klasörün `QuantizedLinear` katmanı her forward'da dequantize eder.
Speedup için optimized backend kernel gerekir.

### “lm_head değiştirdikten sonra embedding farklı davrandı”

Weight tying bozulmuş olabilir. Shared Parameter'ları quantize etmeden önce
data pointer/object identity kontrol et.

### “QAT loss dalgalanıyor”

QAT approximate gradient kullanır. Şunlar önemlidir:

- düşük learning rate
- observer/qparam schedule
- calibration
- fake quant enable zamanı
- clipping
- regularization

Toy örnek QAT'nin bütün production stabilizasyonlarını içermez.

---

## 28. Çalışma sırası

### Seviye 1 — Sayılar

```bash
.venv/bin/python quantization/by_hand.py
```

Şunları elle tekrar yap:

- scale
- q
- x_hat
- error/MSE
- nibble bytes

### Seviye 2 — Core implementation

Oku:

1. [`core.py`](core.py)
2. [`linear.py`](linear.py)

Notebook'ta küçük tensor dene:

```python
w = torch.randn(4, 16)
for bits in (8, 4):
    for group in (16, 8, 4):
        ...
```

### Seviye 3 — Model

```bash
.venv/bin/python quantization/demo.py --show-layers
```

Sonra:

```bash
.venv/bin/python quantization/demo.py --group-size 16
```

Storage ve loss değişimini kaydet.

Q8'i ayrı ve daha ayrıntılı çalış:

```bash
.venv/bin/python quantization/q8_0.py
.venv/bin/python quantization/q8_demo.py --show-layers
```

İlk script tek block'u byte ve sayı düzeyinde gösterir. İkincisi Q8_0'ı gerçek
TinyQwen checkpoint'ine uygular.

### Seviye 4 — Training

```bash
.venv/bin/python quantization/qat_demo.py
```

STE satırını değiştirip gradient'i gözle.

### Seviye 5 — Formatlar

```bash
.venv/bin/python quantization/nf4.py
```

Oku:

- [`q4_k.py`](q4_k.py)
- bu belgedeki GGUF bölümleri

### Seviye 6 — Advanced PTQ

```bash
.venv/bin/python quantization/advanced_methods.py
```

Her yöntemin hangi error'ı azalttığını ayrı yaz:

- GPTQ: second-order output error compensation
- AWQ: activation-aware weight channel protection
- SmoothQuant: activation outlier migration

### Alıştırmalar

[`EXERCISES.md`](EXERCISES.md) içindeki soruları önce çözüm bölümünü kapatarak
yap.

---

## 29. Terimler sözlüğü

| Terim | Kısa anlam |
| --- | --- |
| Quantization | Sayıları daha az temsil seviyesine eşleme |
| Dequantization | Integer/codebook temsilinden yaklaşık float'a dönüş |
| Scale | Bir integer adımının float karşılığı |
| Zero-point | Gerçek sıfırın integer kodu |
| Clipping | Range dışı değeri sınıra çekme |
| Calibration | Representative veriyle qparam/statistics toplama |
| Observer | Activation/weight range istatistiği toplayan yapı |
| Granularity | Bir qparam setinin kaç değeri paylaştığı |
| Group size | Bir scale/zero-point paylaşan weight sayısı |
| Codebook | Kod index'lerinin işaret ettiği değer listesi |
| Packing | Birkaç düşük-bit kodu byte'lara sıkıştırma |
| bpw | Bits per weight |
| PTQ | Post-Training Quantization |
| QAT | Quantization-Aware Training |
| STE | Straight-Through Estimator |
| Weight-only | Esas olarak yalnız weight quantized |
| W4A16 | 4-bit weight, 16-bit activation hedefi |
| W8A8 | 8-bit weight ve activation hedefi |
| KV cache | Önceki token attention Key/Value belleği |
| GGUF | Model tensor/metadata container formatı |
| K-quant | llama.cpp 256-value super-block quant ailesi |
| I-quant | Nonlinear/importance-aware düşük-bit ailesi |
| NF4 | Normal dağılıma göre 16 seviyeli NormalFloat4 |
| GPTQ | Second-order, calibration-aware weight PTQ |
| AWQ | Activation-aware weight quantization |
| SmoothQuant | Activation outlier yükünü weight'e taşıyan W8A8 PTQ |
| QLoRA | Frozen NF4 base üzerinde LoRA fine-tuning yaklaşımı |

---

## 30. Q8 ve Q8_0 derinlemesine

### Önce en önemli ayrım: “Q8” tek başına format değildir

Günlük konuşmada Q8 çoğu zaman “yaklaşık 8-bit quantization” anlamında
kullanılır. Fakat gerçek bir tensor encoding için aşağıdaki soruların yanıtı
gerekir:

- Kodlar signed mı unsigned mı?
- Symmetric mi asymmetric mi?
- Kaç sayı aynı scale'ı paylaşıyor?
- Scale FP32 mi FP16 mı saklanıyor?
- Zero-point var mı?
- Weight mi, activation mı, KV cache mi quantize ediliyor?
- Runtime bu byte düzeni için optimized kernel içeriyor mu?

Şu etiketleri birbirine karıştırma:

| Etiket | Ne anlatır? |
| --- | --- |
| Generic INT8 | Geniş bir 8-bit integer quantization ailesi |
| `Q8_0` | llama.cpp/GGML içindeki belirli block tensor encoding |
| W8A8 | Weight ve activation'ın 8-bit olduğu compute hedefi |
| `Q8_K` | K-quant içindeki farklı/helper tür; `Q8_0` ile aynı değildir |
| GGUF | Q8_0 tensorlerini ve model metadata'sını taşıyabilen container |

Bu klasördeki generic symmetric INT8 şu ayarlarla Q8_0'ın matematiğine eşit
olur:

```text
group_size = 32
scale dtype = FP16
integer range = [-127, 127]
zero_point = 0
```

Fakat [`q8_0.py`](q8_0.py) bunun genel bir INT8 ayarı olmadığını, belirli bir
block formatı olduğunu class ve field isimleriyle açıkça gösterir.

### Bir Q8_0 block tam olarak ne saklar?

Current llama.cpp yapısının sade karşılığı:

```c
#define QK8_0 32

struct block_q8_0 {
    fp16 d;        // delta/scale
    int8 qs[32];   // 32 quantized code
};
```

Byte hesabı:

```text
32 adet int8 code = 32 × 1 byte = 32 byte
1 adet FP16 scale =  1 × 2 byte =  2 byte
------------------------------------------------
toplam                           = 34 byte
```

Effective bits per weight:

```text
34 byte × 8 bit/byte / 32 weight = 8.5 bpw
```

Bu yüzden “Q8_0 tam 8 bpw'dir” cümlesi yanlıştır. Payload sekiz bittir;
block metadata'sı eklendiğinde gerçek maliyet 8.5 bpw olur.

Bir milyar uygun weight yalnız Q8_0 block storage olarak yaklaşık:

```text
1,000,000,000 × 8.5 / 8 ≈ 1.0625 GB
```

kullanır. Bu hesap GGUF file metadata/alignment, quantize edilmeyen tensorler,
KV cache ve runtime buffer'larını içermez.

### Neden `[-127,127]` ve neden `-128` kullanılmıyor?

Symmetric quantization sıfırın iki tarafında eşit büyüklükte range ister:

```text
negative: -127 ... -1
zero    : 0
positive: 1 ... 127
```

`-128` kullanılsa negatif tarafta bir fazla magnitude seviyesi olurdu. Q8_0
reference quantizer symmetric `[-127,127]` range kullanır.

Bu, her INT8 sisteminin `-128`i bırakacağı anlamına gelmez. Örneğin bazı
kernel/formatlar full signed range kullanabilir. Yine “INT8” etiketinin tek
başına yeterli olmadığını görüyoruz.

### Q8_0 quantization formülü

Her 32-value block bağımsız işlenir:

```text
amax = max(abs(x))
d = amax / 127
q = clip(round(x / d), -127, 127)
```

Saklanan:

```text
FP16(d)
32 adet int8 q
```

Dequantization:

```text
x_hat = q × FP16(d)
```

Buradaki ince implementation detayı:

1. `amax` ve `d` FP32 hesaplanır.
2. Integer code, FP32 `d` kullanılarak seçilir.
3. Dosyada/buffer'da `d` FP16 saklanır.
4. Decode sırasında stored FP16 `d` kullanılır.

Yani:

```text
code seçen scale ≈ decode eden stored scale
```

ama FP16 rounding nedeniyle bit düzeyinde aynı sayı olmak zorunda değildir.
[`q8_0.py`](q8_0.py) bu ayrıntıyı korur.

### Elle yapılabilir 32-value block örneği

Block içindeki maximum absolute value:

```text
amax = 12.7
d_FP32 = 12.7 / 127 ≈ 0.1
stored_FP16(d) = 0.099975586
```

İlk değerler:

| index | `x` | `x/d` | `q` | `x_hat=q×FP16(d)` | error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | -12.70 | -127.0 | -127 | -12.696899 | 0.003100 |
| 1 | -6.35 | -63.5 | -64 | -6.398438 | -0.048438 |
| 2 | -3.20 | -32.0 | -32 | -3.199219 | 0.000781 |
| 3 | -1.05 | -10.5 | -11 | -1.099731 | -0.049731 |
| 4 | 0.00 | 0.0 | 0 | 0.000000 | 0 |
| 5 | 0.14 | 1.4 | 1 | 0.099976 | -0.040024 |
| 6 | 1.25 | 12.5 | 13 | 1.299683 | 0.049683 |
| 7 | 4.40 | 44.0 | 44 | 4.398926 | -0.001074 |
| 8 | 8.88 | 88.8 | 89 | 8.897827 | 0.017827 |
| 9 | 12.70 | 127.0 | 127 | 12.696899 | -0.003100 |

Kalan 22 değer sıfır seçilmiştir. Script'in ölçtüğü bütün block MSE:

```text
0.000288392
```

Burada half-away-from-zero rounding kullanıldığı için:

```text
-63.5 -> -64
+12.5 -> +13
```

olur. Bu tabloyu ezberlemek yerine her satırda `q` ve `x_hat` hesabını
tekrarla.

### Tamamen sıfır block

```text
amax = 0
d = 0
```

Doğrudan `x/d` yapmak division-by-zero üretir. Correct implementation:

```text
d == 0 ise inverse_scale = 0
q = 0
x_hat = 0
```

seçer. Test dosyası üç sıfır block'un exact zero döndüğünü doğrular.

### Block-wise scale neden önemli?

İki block düşün:

```text
block 0: 32 sayı, range [-0.25, 0.25]
block 1: 32 sayı, range [-20, 20]
```

Tek global INT8 scale:

```text
d ≈ 20/127 ≈ 0.1575
```

olur. Bu step `0.25` büyüklüğündeki küçük değerler için çok kabadır.

Q8_0:

```text
block 0 d ≈ 0.0019684
block 1 d ≈ 0.1574707
```

kullanır. Runnable sonuç:

| Yöntem | Küçük block MSE | Total MSE | Storage |
| --- | ---: | ---: | ---: |
| Q8_0, iki scale | 0.000000313 | 0.001001081 | 68 B |
| Global INT8, tek scale | 0.002284769 | 0.002143309 | 66 B |

İki byte daha metadata ile küçük block çok daha iyi korunur. Aynı trade-off:

```text
daha küçük group -> daha local range -> genellikle daha az error
daha küçük group -> daha fazla scale -> daha çok metadata
```

### Q8_0 ile W8A8 aynı şey değildir

Bir modelin GGUF dosyasındaki weight tensorleri Q8_0 olabilir. Bu cümle:

- activation'ların 8-bit olduğunu,
- INT8 × INT8 GEMM kullanıldığını,
- accumulator'ın 8-bit olduğunu,
- KV cache'in 8-bit olduğunu

söylemez.

Bu klasördeki `Q8_0Linear`:

```text
Q8_0 weight storage
    -> forward'da FP weight'e dequantize
    -> PyTorch float F.linear
```

yapar. Storage gerçektir; compute path eğitim için okunabilirdir. Production
runtime aynı layout için fused/vectorized kernel kullanabilir.

W8A8 ise weight ve activation'ın ikisini de 8-bit compute'a hazırlayan daha
geniş bir hedefi anlatır. Activation outlier'ları nedeniyle calibration,
per-token/per-channel scale veya SmoothQuant gibi teknikler gerekebilir.

### Q8_0 ne zaman seçilir?

Mantıklı durumlar:

- Source'a yakın yüksek kaliteli quantized baseline istiyorsun.
- Q4/Q5 görev kalitesini kabul edilemez ölçüde düşürüyor.
- Raw FP16 weight boyutunun yaklaşık yarısına yakın storage sığıyor.
- Runtime Q8_0 için iyi kernel sunuyor.
- Daha küçük quantları karşılaştırmak için güçlü reference noktası istiyorsun.

Daha düşük bit düşün:

- RAM/VRAM sınırı Q8_0'ı taşımıyor.
- Q4_K_M veya Q5_K_M aynı görev kalitesini yeterli koruyor.
- Modelden kalan bellek KV cache/context için yetmiyor.
- Backend düşük-bit formatta daha iyi throughput veriyor.

Daha yüksek/source precision düşün:

- Numerically sensitive bilimsel/kod görevinde Q8_0 farkı kabul edilmiyor.
- Quantization validation başarısız.
- Memory budget sorun değil.

Her durumda aynı prompt/dataset üzerinde:

- loss/perplexity veya task metric,
- peak RAM/VRAM,
- prompt processing tokens/s,
- generation tokens/s,
- long-context davranışı

ölç.

### Q8_0 hakkında yaygın yanlışlar

**“8-bit olduğuna göre exact/lossless.”**

Hayır. Scale grid'ine rounding ve FP16 scale storage nedeniyle error vardır.

**“Q8_0 model FP16'nın tam yarısıdır.”**

Raw eligible weight payload'a yakındır ama 8.5 bpw metadata, mixed tensorler,
file metadata, alignment ve runtime buffers tam oranı değiştirir.

**“Q8_0 yazıyorsa activation da INT8.”**

Hayır. Tensor storage encoding ile compute configuration ayrı kavramlardır.

**“Q8_0 her zaman Q4'ten hızlıdır/yavaştır.”**

Evrensel değil. Daha basit decode, daha fazla bandwidth, kernel kalitesi ve
hardware birlikte sonucu belirler.

**“Q8_K ile Q8_0 aynı.”**

Hayır. İsimdeki sekiz aynı payload bit fikrini çağrıştırsa da block field'ları,
amaç ve kullanım farklıdır. Format adını tam oku.

---

## 31. Q8_0 gerçek Python implementation walkthrough

### `Q8_0Tensor`: float kopyası olmayan storage object

[`q8_0.py`](q8_0.py) içindeki ana fields:

```python
@dataclass
class Q8_0Tensor:
    qs: torch.Tensor       # int8, [n_blocks, 32]
    d: torch.Tensor        # float16, [n_blocks]
    original_shape: tuple
```

Bilerek saklanmayan:

```text
original FP32 weight
zero_point
minimum
codebook
```

Storage property'leri:

```python
payload_nbytes  = qs.numel() * qs.element_size()
metadata_nbytes = d.numel() * d.element_size()
storage_nbytes  = payload_nbytes + metadata_nbytes
bpw             = 8 * storage_nbytes / logical_weight_count
```

Bir block için:

```text
qs: 32 × int8    -> 32 byte
d :  1 × float16 ->  2 byte
```

### Input shape validation

Real Q8_0 row complete 32-value block'lardan oluşur:

```python
if tensor.shape[-1] % 32 != 0:
    raise ValueError(...)
```

Neden sessiz padding yapmıyoruz?

Eğitim amaçlı bir quantizer herhangi shape'i padding ile saklayabilir. Fakat
bu padding'i llama.cpp/GGUF encoding davranışı gibi göstermek yanlış olur.
Implementation hatayı erken ve görünür yapar.

### Qparam hesabı

```python
blocks = tensor.float().reshape(-1, 32)
amax = blocks.abs().amax(dim=1)
d_float = amax / 127
inverse_scale = where(d_float > 0, 1/d_float, 0)
```

Burada her row 32 sayı içerir ve her row'a bir `amax/d` düşer.

### Integer code üretimi

```python
q = round_half_away_from_zero(
    blocks * inverse_scale[:, None]
)
q = q.clamp(-127, 127).to(torch.int8)
```

Önce FP32 scale kullanılır. Sonra scale storage'a:

```python
d = d_float.to(torch.float16)
```

çevrilir.

### Dequantization

```python
x_hat = qs.float() * d.float().unsqueeze(1)
x_hat = x_hat.reshape(original_shape)
```

Her code yalnız kendi block scale'ıyla çarpılır.

### `Q8_0Linear`

`nn.Linear` normalde:

```text
weight: Parameter[output, input], çoğunlukla FP32/FP16
bias  : optional Parameter
```

`Q8_0Linear`:

```text
qweight: int8 buffer
scales : FP16 buffer
bias   : optional unquantized buffer
```

saklar. `weight` attribute yoktur:

```python
qlinear = Q8_0Linear.from_float(float_linear)
assert not hasattr(qlinear, "weight")
```

Örnek `Linear(64,16)`:

```text
logical weight count = 64 × 16 = 1024
block count          = 1024 / 32 = 32
int8 payload         = 1024 byte
FP16 scales          = 32 × 2 = 64 byte
Q8_0 weight total    = 1088 byte
FP32 weight total    = 1024 × 4 = 4096 byte
```

Script ölçümü:

```text
output MSE = 0.000006043
```

Bias bu weight hesabına dahil değildir ve quantize edilmeden korunur.

### Whole-model replacement

```python
model = fresh_float_model(checkpoint)
report = quantize_model_q8_0(
    model,
    exclude=("lm_head",),
)
```

Helper önce bütün matching `nn.Linear` isimlerini toplar, sonra her layer'ı
`Q8_0Linear` ile değiştirir. TinyQwen'de:

```text
14 attention/MLP Linear
18,432 weight
```

quantize edilir. `lm_head` embedding ile tied olduğu için dışarıda kalır.

### TinyQwen Q8_0 sonucu

Komut:

```bash
.venv/bin/python quantization/q8_demo.py
```

Quantize edilen Linear'lar:

```text
FP32   73,728 byte
Q8_0   19,584 byte
ratio  3.76×
bpw    8.500
```

Quantize edilmeyen embedding/norm dahil whole-model weight storage:

| Model | Byte | FP32'ye göre |
| --- | ---: | ---: |
| FP32 | 78,336 | 1.00× |
| Q8_0 | 24,192 | 3.24× |

64 deterministic corpus window sonucu:

| Model | Loss | Perplexity | Mean `abs(Δlogit)` | Max `abs(Δlogit)` |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 0.79982 | 2.2251 | 0 | 0 |
| Q8_0 | 0.79810 | 2.2213 | 0.086648 | 1.433005 |

Quantized loss'un bu küçük örnekte FP32 loss'tan çok az düşük çıkması Q8_0'ın
FP32'den genel olarak “daha iyi” olduğunu kanıtlamaz. Evaluation sample'ı küçük
ve model tiny'dir; quantization noise bazen belirli batch'te loss'u tesadüfen
düşürebilir. Daha geniş validation/task benchmark kullan.

Generation aynı sampling seed ile çoğunlukla aynı, bir örnekte farklıdır:

```text
FP32 : ... 'samelike' ...
Q8_0 : ... 'samerin'  ...
```

Küçük logit farkı autoregressive sampling'de bir token seçimini değiştirip
sonraki bütün token yolunu değiştirebilir. Bu yüzden generation equality
quantization'ın zorunlu kriteri değildir.

### Generic INT8 ile Q8_0 equivalence testi

`q8_demo.py` mevcut generic INT8'i şu ayarla kurar:

```text
symmetric=True
group_size=32
scale_dtype=FP16
```

Sonra 14 layer'da karşılaştırır:

```text
differing int8 codes    = 0
differing FP16 scales   = 0
max dequant weight diff = 0
max model logit diff    = 0
```

Bu çok önemli bir öğrenme noktasıdır:

```text
aynı matematik ve aynı metadata layout
    -> aynı quantized representation
```

Fakat “bütün INT8 yöntemleri Q8_0'dır” sonucu çıkmaz. Group size, qrange,
zero-point, scale dtype veya granularity değişirse representation değişir.

### Save/load

`qweight` ve `scales` buffer olduğu için `state_dict` içine girer:

```python
torch.save(q8_model.state_dict(), file)
rebuilt.load_state_dict(torch.load(file, weights_only=True))
```

Ölçülen:

```text
serialized state_dict = 37,041 byte
reload max logit diff = 0
```

Serialized file, raw weight storage hesabından büyüktür; key isimleri, pickle/
zip container metadata'sı ve quantize edilmemiş tensorler de vardır.

### Kodla çalışma sırası

1. Tek block ve Linear:

   ```bash
   .venv/bin/python quantization/q8_0.py
   ```

2. Her layer'ın block/byte hesabı:

   ```bash
   .venv/bin/python quantization/q8_demo.py --show-layers
   ```

3. Testler:

   ```bash
   PYTHONPATH=quantization .venv/bin/python -m unittest \
     discover -s quantization/tests -v
   ```

4. Kendi tensorün:

   ```python
   import torch
   from q8_0 import quantize_q8_0

   weight = torch.randn(16, 64)
   q8 = quantize_q8_0(weight)
   restored = q8.dequantize()

   print(q8.qs.dtype, q8.qs.shape)
   print(q8.d.dtype, q8.d.shape)
   print(q8.storage_nbytes)
   print(((weight-restored)**2).mean())
   ```

5. Outlier experiment:

   - Bir 32-value block üret.
   - Tek değeri `100.0` yap.
   - Block MSE ve yalnız outlier dışındaki değerlerin MSE'sini ayrı ölç.
   - Aynı tensorü 64-value global scale ile karşılaştır.

### Q8_0 validation checklist

- `qweight.dtype == torch.int8`
- `scale.dtype == torch.float16`
- her scale tam 32 code paylaşıyor
- code range `[-127,127]`
- 32 weight için storage `34 byte`
- no-padding durumda `8.5 bpw`
- zero block finite ve exact zero
- quantized Linear içinde float `weight` yok
- save/load sonrası output/logit exact aynı
- speed sonucu optimized runtime dışında iddia edilmiyor
- Q8_0, W8A8 ve KV-cache quantization birbirinden ayrılıyor

---

## Kaynaklar

- [llama.cpp quantization guide](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)
- [llama.cpp tensor encoding schemes](https://github.com/ggml-org/llama.cpp/wiki/Tensor-Encoding-Schemes)
- [llama.cpp Q8_0 reference quantizer](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c)
- [llama.cpp Q8_0 block definition](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h)
- [llama.cpp backend feature matrix](https://github.com/ggml-org/llama.cpp/wiki/Feature-matrix)
- [GPTQ paper](https://arxiv.org/abs/2210.17323)
- [AWQ paper](https://arxiv.org/abs/2306.00978)
- [SmoothQuant paper](https://proceedings.mlr.press/v202/xiao23c.html)
- [QLoRA paper](https://arxiv.org/abs/2305.14314)
- [PyTorch quantization documentation](https://docs.pytorch.org/docs/stable/quantization.html)

llama.cpp recipe'leri zamanla değişebildiği için exact `Q4_K_M` tensor
dağılımını anlatmadan önce güncel source code kontrol edilmelidir.
