# Quantization Alıştırmaları ve Cevap Anahtarı

Öneri: Önce yalnız “Sorular” bölümünü çöz. Sonra
[cevap anahtarına](#cevap-anahtarı) geç. Kod sorularında repository root'undan
çalış.

---

## Sorular

## A. Temel matematik

### A1 — Symmetric 3-bit oyuncak quantizer

Kodlar `[-3,3]`, değerler:

```text
x = [-1.5, -0.4, 0.0, 0.7, 1.2]
```

1. `scale = max(abs(x))/3` kaçtır?
2. Her `q` kodunu hesapla.
3. `x_hat` değerlerini hesapla.
4. Maximum absolute error kaçtır?

### A2 — Clipping

A1'deki scale sabitken yeni input `2.2` gelirse:

1. Clipping öncesi kod nedir?
2. Clipping sonrası kod nedir?
3. `x_hat` ve error nedir?

### A3 — Asymmetric 4-bit

```text
xmin = -2
xmax = 6
qmin = 0
qmax = 15
```

1. Scale'ı hesapla.
2. Zero-point'i hesapla.
3. Gerçek `0` hangi integer koda gider?
4. Bu kod neden önemlidir?

### A4 — MSE

```text
x     = [0.0, 0.2, 0.7, 1.0]
x_hat = [0.0, 0.25, 0.75, 1.0]
```

MSE'yi hesapla.

---

## B. Storage ve packing

### B1 — INT4 packing

Signed INT4 değerleri:

```text
[-1, 2, -8, 7]
```

Two's-complement nibble'ları ve iki packed byte'ı hexadecimal yaz.
İlk değer low nibble'a yerleşsin.

### B2 — “4-bit” ama 8-bit storage

Şu tensor neden gerçekten 4-bit storage değildir?

```python
q = torch.tensor([1, 2, 3, 4], dtype=torch.int8)
```

Gerçek 4-bit storage için ne yapılmalıdır?

### B3 — Effective bpw

Packed INT4 ve her group için bir FP16 scale:

1. `group_size=32` için effective bpw?
2. `group_size=16` için?
3. Hangisi daha fazla metadata kullanır?

### B4 — Padding

Shape `[8, 35]`, group size `32`.

1. Her row kaç group kullanır?
2. Her row kaç weight slot'u saklar?
3. Kaç padding slot oluşur?
4. Padding effective bpw'yi nasıl etkiler?

### B5 — Q4_K

Bir Q4_K super-block:

- 256 adet 4-bit code,
- 12 byte scale/min metadata,
- iki FP16 değer

saklıyor. Toplam byte ve bpw hesapla.

---

## C. Kavram eşleştirme

Aşağıdaki yöntemleri açıklamalarla eşleştir:

1. GPTQ
2. AWQ
3. SmoothQuant
4. QAT
5. NF4
6. QLoRA

Açıklamalar:

a. Normal dağılıma uygun non-uniform 16-level codebook  
b. Activation outlier yükünü weight'lere taşıyan W8A8 yöntemi  
c. Frozen 4-bit base üzerinden LoRA adapter eğitimi  
d. Second-order bilgiyle weight quantization error compensation  
e. Training sırasında fake quantization  
f. Activation istatistiğiyle önemli weight channel'larını koruma  

---

## D. GGUF ve isimler

### D1

Şu üç terimin farkını tek cümleyle yaz:

- GGUF
- Q4_K
- Q4_K_M

### D2

“`Q4_K_M` modelindeki her weight 4 bittir” cümlesi neden yanlış?

### D3

`Q4_K_S`, `Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q8_0` seçeneklerini genel olarak
küçükten yüksek precision tarafına sırala. Bunun evrensel kalite garantisi
olmadığını da açıkla.

### D4

Bir `.gguf` dosyasının isminde `Q4_K_M` yazması gerçek ortalama bpw'yi neden
tek başına vermez?

---

## E. Model ve runtime

### E1 — Weight tying

TinyQwen'de:

```python
model.lm_head.weight = model.embed_tokens.weight
```

Yalnız `lm_head`i `QuantizedLinear` ile değiştirmenin iki olası sakıncası nedir?

### E2 — Speed

`QuantizedLinear` gerçek packed INT4 sakladığı halde neden FP32 modelden daha
yavaş çalışabilir?

### E3 — KV cache

Model weight'lerini 4-bit yapmak neden uzun context belleğini tamamen çözmez?

### E4 — Kalite metriği

Neden yalnız beş generation örneğine bakmak yeterli değildir? En az iki daha
güçlü evaluation yöntemi söyle.

---

## F. Senaryolar

### F1 — 8 GB laptop

FP16 model sığmıyor. Local GGUF inference yapmak istiyorsun. İlk denenecek
quant ve kontrol edilecek dört ölçüm nedir?

### F2 — Kod üretimi

`Q4_K_M` ile model sığıyor fakat unit test pass rate düşüyor. Sonraki iki
mantıklı adım nedir?

### F3 — Uzun belge

Model rahatça sığıyor fakat 100k-token context sırasında bellek taşıyor. Weight
quantization dışında hangi parçaya bakmalısın?

### F4 — Beklenmeyen yavaşlık

`IQ4_XS`, `Q4_K_M`den küçük fakat cihazında daha yavaş. Bu çelişki midir?
Neden?

---

## G. Kod çalışmaları

### G1 — Group-size sweep

[`core.py`](core.py) kullanarak shape `[64,128]` random weight üret. INT4 için
group size `128, 64, 32, 16` dene. Her biri için:

- MSE
- payload bytes
- metadata bytes
- effective bpw

yazdır.

Beklenen genel trend nedir?

### G2 — Symmetric/asymmetric

Tamamı pozitif bir tensor üret:

```python
x = torch.rand(1024) * 3 + 2
```

Symmetric ve asymmetric 4-bit MSE'yi karşılaştır. Sonucu açıkla.

### G3 — Outlier

Normal random weight'in tek bir değerini `20.0` yap. Per-tensor ve group-wise
quantization'ı karşılaştır. Hangi group'lar etkilenir?

### G4 — Packing round-trip

Rastgele `[-7,7]` INT4 kodları üret:

```python
q = torch.randint(-7, 8, (10, 32), dtype=torch.int8)
```

Şunu doğrula:

```python
q == unpack_int4(pack_int4(q))
```

Packed tensor kaç byte olmalıdır?

### G5 — TinyQwen

Çalıştır:

```bash
.venv/bin/python quantization/demo.py
.venv/bin/python quantization/demo.py --group-size 16
```

İki çalışmadaki:

- INT4 storage
- effective bpw
- loss
- logit drift

farkını tabloya yaz. Küçük group her metriği aynı yönde mi değiştiriyor?

### G6 — QAT

[`qat_demo.py`](qat_demo.py) içinde `QAT_STEPS` değerini `0`, `50`, `250` yap.
Packed accuracy'yi kaydet. QAT adımı arttıkça sonuç her zaman monoton artıyor
mu? Neden garanti değildir?

### G7 — NF4

[`nf4.py`](nf4.py) örneğine Laplace dağılımı ekle:

```python
dist = torch.distributions.Laplace(0, 1)
x = dist.sample((4096,))
```

Uniform INT4 ve NF4 MSE'yi ölç. Codebook ile veri dağılımı arasındaki ilişkiyi
yorumla.

### G8 — Model recipe tasarla

TinyQwen için kendi “M” recipe'ni kağıt üzerinde tasarla:

- çoğu MLP weight INT4,
- attention `v_proj` INT8,
- output/embedding FP32.

Tahmini storage'ı hesapla. Sonra `quantize_model_linears` çağrılarını kullanarak
uygula ve loss'u ölç.

---

## H. Q8 ve Q8_0 çalışmaları

### H1 — Exact block storage

Bir Q8_0 block:

- 32 adet signed INT8 code,
- bir adet FP16 scale

saklıyor.

1. Bir block kaç byte?
2. Effective bits per weight kaç?
3. `Linear(64,16, bias=False)` weight'i kaç block ve kaç Q8_0 byte kullanır?
4. Aynı weight FP32 iken kaç byte kullanır?

### H2 — Elle code ve dequantization

Bir block için:

```text
amax = 12.7
d_FP32 = 12.7/127 ≈ 0.1
stored_FP16(d) = 0.099975586
```

Aşağıdaki değerler için half-away-from-zero rounding kullanarak `q`, `x_hat`
ve error hesapla:

```text
x = [-12.7, -6.35, -1.05, 0.0, 0.14, 1.25, 12.7]
```

Neden `x=12.7` bile dequantization sonrası tam `12.7` olmayabilir?

### H3 — `-128` neden kullanılmıyor?

Q8_0 integer payload dtype'ı `int8` olduğu halde reference quantizer neden
`[-127,127]` range kullanıyor? Bunun symmetric quantization ile ilişkisini
açıkla.

### H4 — Global scale ve block scale

64 value:

```text
ilk 32 : [-0.25, 0.25] aralığında
son 32 : [-20, 20] aralığında
```

1. Tek global INT8 scale yaklaşık kaçtır?
2. Q8_0 ilk block scale'ı yaklaşık kaçtır?
3. Hangi yöntem küçük değerleri daha iyi korur ve neden?
4. Q8_0 bu örnekte global yönteme göre kaç ek metadata byte kullanır?

### H5 — Kavramları ayır

Her birini bir cümleyle açıkla:

- generic INT8
- Q8_0
- W8A8
- Q8_K
- GGUF

“Q8_0 model kullanıyorum, dolayısıyla activation ve KV cache de 8-bittir”
cümlesindeki hatayı göster.

### H6 — Zero block ve edge case

Tamamen sıfır bir Q8_0 block için:

1. `amax` ve `d` nedir?
2. Naive `x/d` neden sorun çıkarır?
3. Güvenli implementation hangi `q` kodlarını üretmelidir?
4. Dequantization sonucu ne olmalıdır?

### H7 — Shape constraint

[`q8_0.py`](q8_0.py) neden last dimension'ın 32'ye bölünmesini zorunlu tutuyor?
Shape `[2,31]` için sessiz padding yapmak eğitim açısından kolay olsa da gerçek
formatı anlatırken neden yanıltıcı olabilir?

### H8 — Runnable TinyQwen analizi

Çalıştır:

```bash
.venv/bin/python quantization/q8_demo.py --show-layers
```

Şunları kaydet ve yorumla:

- quantize edilen layer/weight sayısı,
- Q8_0 effective bpw,
- whole-model FP32 ve Q8_0 byte,
- loss/perplexity,
- mean/max logit drift,
- generic group-32 INT8 ile code/scale fark sayısı,
- state_dict reload farkı,
- generated name'lerdeki olası fark.

Quantized loss FP32 loss'tan çok az düşükse neden “Q8_0 FP32'den daha
kalitelidir” sonucu çıkaramazsın?

### H9 — Q8_0 seçim senaryosu

İki sistem:

1. Q4_K_M ile kalite düşüyor ama Q8_0 belleğe sığıyor.
2. Q8_0 sığıyor fakat 100k-token context'te KV cache belleği taşıyor.

Her sistem için ilk mantıklı sonraki adımı yaz. Hangi metrikleri tekrar
ölçersin?

---

# Cevap anahtarı

## A cevapları

### A1

```text
max_abs = 1.5
scale = 1.5/3 = 0.5

x/scale = [-3.0, -0.8, 0.0, 1.4, 2.4]
q       = [-3, -1, 0, 1, 2]
x_hat   = [-1.5, -0.5, 0.0, 0.5, 1.0]
error   = [0.0, -0.1, 0.0, -0.2, -0.2]
max_abs_error = 0.2
```

Burada `round(1.4)=1`, `round(2.4)=2`.

### A2

```text
2.2/0.5 = 4.4 -> round = 4
clip(4,-3,3) = 3
x_hat = 3×0.5 = 1.5
error = 1.5-2.2 = -0.7
```

### A3

```text
scale = (6-(-2))/15 = 8/15 ≈ 0.533333
zero_point = round(0 - (-2)/0.533333)
           = round(3.75)
           = 4
```

Gerçek sıfır `q=4` koduna gider. Böylece dequantization:

```text
scale × (4-4) = 0
```

ile sıfırı tam korur.

### A4

```text
error = [0, 0.05, 0.05, 0]
MSE = (0 + 0.0025 + 0.0025 + 0)/4
    = 0.00125
```

## B cevapları

### B1

```text
-1 -> 0xF
 2 -> 0x2
-8 -> 0x8
 7 -> 0x7
```

Low nibble first:

```text
[-1, 2] -> 0x2F
[-8, 7] -> 0x78
```

### B2

`torch.int8` her element için bir byte kullanır. Değerlerin dört bite sığması
storage dtype'ını değiştirmez. İki 4-bit code bir `uint8` byte içine
paketlenmelidir.

### B3

```text
group32: 4 + 16/32 = 4.5 bpw
group16: 4 + 16/16 = 5.0 bpw
```

Group16 iki kat fazla scale metadata kullanır.

### B4

```text
ceil(35/32) = 2 group/row
2×32 = 64 slot/row
64-35 = 29 padding/row
```

Padding payload ve group metadata'sı kullanır ama gerçek weight değildir;
logical weight başına effective bpw yükselir.

### B5

```text
256×4 bit = 128 byte
metadata = 12 byte
2 FP16 = 4 byte
total = 144 byte
bpw = 144×8/256 = 4.5
```

## C cevapları

```text
1-d  GPTQ
2-f  AWQ
3-b  SmoothQuant
4-e  QAT
5-a  NF4
6-c  QLoRA
```

## D cevapları

### D1

- GGUF: tensor ve model metadata'sını taşıyan container.
- Q4_K: bir tensorün 256-weight super-block tabanlı encoding'i.
- Q4_K_M: bütün modelde hangi tensorün hangi precision'da tutulacağını seçen
  mixed recipe.

### D2

Q4_K_M hassas tensorleri Q5_K/Q6_K gibi daha yüksek türlerde tutabilir; norm
gibi tensorler quantize edilmeyebilir; shape fallback olabilir.

### D3

Genel olarak:

```text
Q4_K_S -> Q4_K_M -> Q5_K_M -> Q6_K -> Q8_0
```

boyut/precision artış yönüdür. Fakat gerçek kalite ve hız model/backend/task'e
bağlıdır; yalnız isimle garanti edilmez.

### D4

Scale/min metadata, mixed tensor türleri, padding/alignment ve GGUF metadata
gerçek bpw/file size'ı değiştirir.

## E cevapları

### E1

1. Embedding/output arasındaki sharing kırılır ve aynı logical weight iki kez
   saklanabilir.
2. Embedding FP32, output quantized farklı değerler kullanır; modelin tied
   weight varsayımı değişir.

### E2

Bu reference layer her forward'da unpack + dequantize yapıp float matmul
çağırır. Fused optimized INT4 kernel yoktur. Storage compression speedup
garantisi değildir.

### E3

KV cache weight'lerden ayrı, token sayısıyla büyüyen runtime activation
belleğidir. Weight quantization onu otomatik küçültmez.

### E4

Sampling stochastic'tir ve birkaç örnek cherry-pick olabilir. Perplexity/loss,
task benchmark, unit test pass rate, accuracy/F1 veya yeterli sayıda rubric
evaluation kullanılabilir.

## F cevapları

### F1

İlk pratik aday `Q4_K_M`. En az:

- task kalitesi/perplexity,
- peak RAM,
- prompt processing tokens/s,
- generation tokens/s

ölçülür. KV cache payı da bırakılır.

### F2

1. Aynı testte `Q5_K_M` veya `Q6_K` ile karşılaştır.
2. Hâlâ sorun varsa daha yüksek precision/küçük model veya görev için uygun
   quantizer/calibration seç.

### F3

KV cache size/type ve context/batch ayarlarına bak. Destek varsa KV-cache
quantization değerlendir.

### F4

Çelişki değildir. Nonlinear/I-quant decode kernel'i backend'de daha az optimize
olabilir. Küçük storage, otomatik yüksek compute throughput demek değildir.

## G için beklenen sonuçlar

### G1

Group küçüldükçe MSE çoğunlukla düşer; metadata ve effective bpw yükselir.
Payload aynı kalır. Rastgele küçük farklar olabilir, fakat daha küçük group daha
fazla local scale özgürlüğü verir.

### G2

Tamamı pozitif range'de asymmetric quantizer negatif seviyelere yer ayırmadığı
için genellikle daha düşük MSE verir. Bunun karşılığında zero-point metadata'sı
saklar.

### G3

Per-tensor scale bütün tensorde büyür. Group-wise durumda esas olarak outlier'ın
bulunduğu group etkilenir.

### G4

`10×32 = 320` code vardır. İki code/byte:

```text
160 byte
```

olmalıdır.

### G5

Group16 genellikle INT4 reconstruction/loss'u iyileştirebilir; scale metadata
arttığı için storage/effective bpw büyür. Bütün metrikler aynı yönde değişmez:
kalite iyileşirken compression azalabilir.

### G6

Monoton iyileşme garanti değildir. STE approximate gradient, optimizer noise,
learning rate, qparam değişimi ve over-training sonucu dalgalanabilir.

### G7

Sonuç Laplace sample'a göre ölçülmelidir. Ana yorum: codebook hangi
dağılım bölgelerine daha çok seviye ayırıyorsa o dağılımda avantajlı olabilir.

### G8

Tek doğru sayı yoktur; hesap layer shape'leri ve scale metadata'yı içermelidir.
Uygulama sonunda `v_proj`lerin `QuantizedLinear(bits=8)`, diğer seçili
katmanların `bits=4`, output/embedding'in float kaldığı inspection ile
kanıtlanmalıdır.

## H cevapları

### H1

```text
bir block = 32×1 + 1×2 = 34 byte
bpw = 34×8/32 = 8.5

Linear weight count = 64×16 = 1024
block count = 1024/32 = 32
Q8_0 byte = 32×34 = 1088
FP32 byte = 1024×4 = 4096
```

Bias verilmediği için hesaba başka tensor girmez.

### H2

Stored scale kullanılarak yaklaşık sonuç:

| `x` | `q` | `x_hat` | error |
| ---: | ---: | ---: | ---: |
| -12.70 | -127 | -12.696899 | 0.003100 |
| -6.35 | -64 | -6.398438 | -0.048438 |
| -1.05 | -11 | -1.099731 | -0.049731 |
| 0.00 | 0 | 0.000000 | 0 |
| 0.14 | 1 | 0.099976 | -0.040024 |
| 1.25 | 13 | 1.299683 | 0.049683 |
| 12.70 | 127 | 12.696899 | -0.003100 |

Code FP32 scale ile seçilir; decode stored FP16 scale ile yapılır. `0.1` FP16
içinde exact temsil edilmediği için endpoint'te bile küçük fark kalabilir.

### H3

`[-127,127]` sıfırın iki tarafında eşit sayıda ve eşit magnitude seviyeleri
verir. `-128` eklenirse negatif tarafta extra seviye olur. Q8_0 symmetric
quantizer bu nedenle `-128` kodunu kullanmaz. Başka INT8 formatları farklı
range seçebilir.

### H4

```text
global d ≈ 20/127 ≈ 0.15748
ilk Q8_0 block d ≈ 0.25/127 ≈ 0.0019685
```

Q8_0 küçük block'a çok daha küçük step verdiği için küçük değerleri daha iyi
korur. Global yöntem bir FP16 scale, Q8_0 iki FP16 scale kullanır; fark 2
byte'tır. Measured script sonuçları küçük-block MSE için yaklaşık
`0.000000313` ve `0.002284769` verir.

### H5

- Generic INT8: Birçok qrange/granularity/metadata seçeneğini kapsayan genel
  8-bit integer quantization sınıfı.
- Q8_0: Her 32 signed code için bir FP16 scale saklayan belirli tensor encoding.
- W8A8: Weight ve activation'ın 8-bit compute hedefi.
- Q8_K: K-quant ailesindeki farklı/helper representation; Q8_0 alias'ı değil.
- GGUF: Tensor byte'ları ile model/tokenizer metadata'sını taşıyan container.

Q8_0 etiketi weight encoding'i anlatabilir; activation veya KV-cache dtype'ını
tek başına belirlemez.

### H6

```text
amax = 0
d = 0
```

Naive division `0/0` ile NaN üretebilir. Güvenli path inverse scale'ı sıfır,
bütün code'ları `q=0` seçer. Dequantization exact zero block döndürür.

### H7

Q8_0 quantized row complete 32-value block'lardan oluşur. `[2,31]` row'ları
complete block değildir. Custom padding yapılabilir ama padding kuralı,
logical shape/accounting ve serialized layout artık öğreticinin tasarımı olur;
llama.cpp formatının exact davranışı değildir. Bu implementation ayrımı
görünür tutmak için hata verir.

### H8

Default repository sonucu:

```text
14 Linear, 18,432 quantized weight
Q8_0 linears: 19,584 byte, 8.5 bpw
whole FP32: 78,336 byte
whole Q8_0: 24,192 byte
FP32 loss/ppl: 0.79982 / 2.2251
Q8_0 loss/ppl: 0.79810 / 2.2213
mean/max logit drift: 0.086648 / 1.433005
generic INT8 code difference: 0
generic INT8 scale difference: 0
reload max logit difference: 0
```

Generation'da küçük logit farkı token sampling yolunu değiştirebilir.
Quantized loss'un küçük evaluation batch'inde biraz düşük olması genel kalite
kanıtı değildir; noise o batch'e tesadüfen yardımcı olmuş olabilir. Daha geniş
validation ve task benchmark gerekir.

### H9

1. Aynı task benchmark'ında Q8_0'ı dene; kalite, peak memory, prompt/generation
   speed ve latency'yi Q4_K_M ile karşılaştır.
2. Sorun weight değil büyüyen KV cache olabilir. Destek varsa KV-cache
   quantization, daha kısa context, daha az concurrency/batch veya daha küçük
   model değerlendir.

Her iki sistemde de yalnız file size değil total runtime memory ve gerçek
workload kalitesi/hızı yeniden ölçülmelidir.
