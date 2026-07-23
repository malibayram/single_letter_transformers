# LLM Quantization Dersi — Öğretim Rehberi

Bu belge, quantization konusunu ilk kez gören fakat tensor ve temel sinir ağı
kavramlarını bilen öğrenciler için hazırlanacak **90 dakikalık** dersin öğretim
stratejisini tanımlar. Açıklamalar Türkçe, sektörde karşılaşılacak teknik
terimler İngilizce verilecektir.

Bu dosya dersin öğretim stratejisidir. Tam öğrenme/pratik modülü
[`README.md`](README.md), uzun Türkçe anlatım, Python implementation'ları,
alıştırmalar ve kısa referans sayfası olarak ayrıca uygulanmıştır.

## 1. Dersin ana fikri

Ders tek bir cümle üzerine kurulacaktır:

> Quantization, çok sayıdaki hassas sayıyı daha az sayıda temsil seviyesine
> yuvarlayarak bellek ve bazen hız kazanır; bunun karşılığında kontrollü bir
> hata ekler.

Öğrenciler önce bu fikri gündelik sayılarla görecek, sonra aynı işlemi küçük
bir ağırlık vektörüne uygulayacak, en son `Q4_K_M` gibi gerçek LLM dosya
isimlerini okuyacaktır. Böylece isimler ve araçlar, temel fikir anlaşıldıktan
sonra tanıtılmış olur.

## 2. Hedef kitle ve önkoşullar

Öğrencilerin aşağıdakileri bildiği varsayılır:

- Bir tensorün sayı dizisi olduğu
- Ağırlıkların (`weights`) eğitim sırasında öğrenildiği
- Basit bir matrix multiplication fikri
- `float32` sayının bellekte yer kapladığı

Binary aritmetik, donanım kernel'leri, Hessian matrix veya GGUF iç yapısı
önkoşul değildir. Bu konular yalnızca meraklı öğrenciler için ileri okuma
olarak gösterilecektir.

## 3. Öğrenme hedefleri

Dersin sonunda öğrenci:

1. Quantization'ın hangi sayıları neden değiştirdiğini açıklayabilmeli.
2. `scale`, `zero-point`, rounding, clipping ve quantization error kavramlarını
   küçük sayılar üzerinde gösterebilmeli.
3. Weight, activation ve KV-cache quantization arasındaki farkı söyleyebilmeli.
4. PTQ ve QAT yaklaşımlarını; GPTQ, AWQ, SmoothQuant ve QLoRA isimlerinin hangi
   probleme odaklandığını kabaca ayırt edebilmeli.
5. `Q4_K_M` etiketinin “modeldeki her ağırlık tam olarak 4 bittir” anlamına
   gelmediğini açıklayabilmeli.
6. Bir quantization seçerken yalnızca dosya boyutuna değil; kalite, RAM/VRAM,
   context length, backend desteği ve gerçek hıza bakması gerektiğini bilmeli.

## 4. Öğretim ilkeleri

### Somuttan soyuta

Sıralama her zaman şu şekilde olacaktır:

1. Gündelik yuvarlama örneği
2. Kalem-kâğıt üzerinde küçük tensor
3. Bir model ağırlığı
4. Gerçek LLM yöntemleri
5. GGUF dosya isimleri

### Tek formül, tekrar eden yorum

Ders boyunca aynı temel affine quantization formülü kullanılacaktır:

```text
q     = clip(round(x / scale) + zero_point, qmin, qmax)
x_hat = scale * (q - zero_point)
error = x_hat - x
```

- `x`: orijinal floating-point değer
- `q`: saklanan küçük integer kod
- `x_hat`: inference sırasında kullanılan yaklaşık değer
- `scale`: bir integer adımının gerçek sayı dünyasındaki büyüklüğü
- `zero_point`: gerçek sıfırın integer kod karşılığı
- `clip`: temsil aralığının dışındaki değerleri sınıra çekme işlemi

İlk anlatımda symmetric quantization kullanılacak ve `zero_point = 0`
seçilecektir. Asymmetric quantization daha sonra, sıfır noktasına neden ihtiyaç
duyulabileceğini göstermek için eklenecektir.

### Kazancı ve bedeli birlikte göster

Her yöntem için aynı dört soru sorulacaktır:

- Ne küçülür?
- Ne hızlanabilir?
- Hata nerede oluşur?
- Hangi donanım/runtime bunu gerçekten destekler?

“Daha az bit her zaman daha hızlıdır” veya “4-bit model tam dört kat küçüktür”
gibi kesin ama yanlış genellemeler yapılmayacaktır.

### GGUF etiketi ile algoritmayı karıştırma

Üç ayrı katman sürekli ayrıştırılacaktır:

- **Container:** GGUF, tensorleri ve metadata'yı taşıyan dosya biçimidir.
- **Encoding/type:** `Q4_K`, `Q6_K`, `Q8_0` gibi tensor temsil biçimleridir.
- **Model recipe:** `Q4_K_M` gibi etiketler farklı tensorlere farklı
  encoding'ler uygulayan model düzeyi tarifler olabilir.

## 5. 90 dakikalık ders akışı

| Süre | Bölüm | Öğrenci etkinliği | Ana çıktı |
| ---: | --- | --- | --- |
| 0–8 dk | Neden quantization? | FP32 ve 4-bit için yaklaşık bellek hesabı | Problemi tanımlar |
| 8–23 dk | Sayıları az seviyeye yuvarlama | Küçük vektörü elle quantize/dequantize eder | `scale`, clipping ve error |
| 23–35 dk | Modelde ne quantize edilir? | Weight/activation/KV-cache kartlarını eşleştirir | Hedefleri ayırır |
| 35–50 dk | Temel workflow'lar | PTQ, dynamic/static ve QAT akışlarını karşılaştırır | Süreçleri ayırır |
| 50–63 dk | Yaygın LLM teknikleri | GPTQ, AWQ, SmoothQuant ve QLoRA'yı tek cümleyle eşler | İsim haritası oluşturur |
| 63–75 dk | GGUF isimlerini okuma | `Q4_K_M` ve `Q8_0`ı parçalar | Encoding ve mixed recipe |
| 75–84 dk | TinyQwen canlı örneği | FP32/Q8_0/INT4 sonuç tablosunu yorumlar | Gerçek trade-off görür |
| 84–90 dk | Seçim stratejisi ve çıkış soruları | Üç senaryo için format seçer | Öğrenme hedeflerini doğrular |

Canlı kod gecikirse yöntem bölümü kısaltılacak; elle yapılan örnek ve
`Q4_K_M` açıklaması atlanmayacaktır.

## 6. Kalem-kâğıt örneği

İlk örnek özellikle küçük ve elle kontrol edilebilir olacaktır.

```text
x = [-1.2, -0.7, 0.1, 0.8, 1.1]
```

Öğretim kolaylığı için 7 symmetric seviye kullanılacaktır:

```text
qmin = -3
qmax =  3
scale = 1.2 / 3 = 0.4
zero_point = 0
```

Hesap:

| `x` | `round(x / 0.4)` | `q` | `x_hat = q × 0.4` | `error` |
| ---: | ---: | ---: | ---: | ---: |
| -1.2 | -3 | -3 | -1.2 | 0.0 |
| -0.7 | -2 | -2 | -0.8 | -0.1 |
| 0.1 | 0 | 0 | 0.0 | -0.1 |
| 0.8 | 2 | 2 | 0.8 | 0.0 |
| 1.1 | 3 | 3 | 1.2 | 0.1 |

Bu örnekte:

```text
MSE = (0² + 0.1² + 0.1² + 0² + 0.1²) / 5 = 0.006
```

Ardından calibration sırasında aralığın `[-1.2, 1.2]` olarak sabitlendiği
varsayılıp yeni bir `1.6` değeri gösterilecektir. Kod `3` seviyesinde kalır,
geri açılan değer `1.2` olur. Bu, **clipping error** kavramını görünür kılar.

Bu oyuncak örneğin standart bir 3-bit dosya formatı olmadığı açıkça
söylenecektir; amaç formülü öğretmektir.

## 7. Kavram haritası

### Neyi quantize ediyoruz?

| Hedef | Basit açıklama | Temel fayda | Temel risk |
| --- | --- | --- | --- |
| Weights | Modelin öğrenilmiş, çoğunlukla sabit sayıları | Model dosyası ve RAM/VRAM küçülür | Model kalitesi düşebilir |
| Activations | Her input için çalışma anında üretilen sayılar | Uygun kernel ile compute hızlanabilir | Outlier değerler zorlayabilir |
| KV cache | Önceki tokenlardan saklanan attention K/V değerleri | Uzun context ve çoklu kullanıcı belleği azalır | Uzun bağlam kalitesi etkilenebilir |

`W4A16`, weight'lerin yaklaşık 4-bit, activation'ların 16-bit tutulduğu bir
şemayı; `W8A8`, iki tarafın da 8-bit hedeflendiği bir şemayı anlatır. Bu
etiketler tek başına granularity, group size veya kernel hakkında tüm bilgiyi
vermez.

### Ne zaman quantize ediyoruz?

| Yaklaşım | Basit açıklama | Calibration/training |
| --- | --- | --- |
| Dynamic PTQ | Weight önceden; activation aralığı çalışma anında belirlenir | Yeniden training yok |
| Static PTQ | Aralıklar temsilî calibration verisiyle önceden sabitlenir | Kısa calibration gerekir |
| Weight-only PTQ | Esas olarak weight'ler küçültülür | LLM local inference için yaygın |
| QAT | Training sırasında fake quantization ile hata simüle edilir | Training/fine-tuning gerekir |

PTQ, **Post-Training Quantization**; QAT, **Quantization-Aware Training**
olarak ilk kullanımda açılacaktır.

## 8. Yaygın LLM tekniklerini basit anlatma

Bu bölüm yöntemleri matematiksel ayrıntılarıyla değil, “hangi problemi
çözüyor?” sorusuyla tanıtır:

- **Naive round-to-nearest:** Her değeri en yakın seviyeye koyar. Basit
  baseline'dır; hangi ağırlığın daha önemli olduğunu bilmez.
- **GPTQ:** Training sonrası weight quantization yaparken bir weight'teki
  yuvarlama hatasının layer output'una etkisini yaklaşık second-order bilgiyle
  telafi etmeye çalışır.
- **AWQ:** Küçük bir calibration setindeki activation istatistiklerini
  kullanarak önemli weight channel'larını daha iyi korumayı hedefler.
- **SmoothQuant:** Quantize edilmesi zor activation outlier yükünün bir kısmını
  eşdeğer bir ölçekleme ile weight'lere taşır; özellikle `W8A8` inference'ı
  hedefler.
- **NF4 / QLoRA:** Normal dağılıma yakın weight'ler için 4-bit NormalFloat
  temsilini kullanır; frozen quantized base model üzerinden küçük LoRA
  adapter'larını eğiterek fine-tuning belleğini azaltır.
- **GGUF K-quants:** Weight'leri block/super-block grupları içinde scale ve
  gerektiğinde minimum değerlerle temsil eder.
- **GGUF I-quants:** Düşük bitlerde kaliteyi korumak için nonlinear codebook ve
  importance bilgisi kullanan/ondan yararlanabilen quant ailesidir. Backend'e
  göre benzer boyuttaki K-quants'tan daha yavaş olabilir.

Önemli ayrım: QLoRA bir fine-tuning yaklaşımıdır; `Q4_K_M` ise yaygın olarak
llama.cpp/GGUF inference dünyasında görülen bir recipe adıdır. Aynı kategoride
değillerdir.

## 9. `Q4_K_M` ve diğer GGUF isimlerini okuma

### `Q4_K` ne söyler?

- `Q4`: weight kodları nominal olarak 4-bit seviyelerdedir.
- `K`: block'ların 256-weight **super-block** yapısında gruplanan K-quant
  ailesidir.
- Temel `Q4_K`, 256 weight'i sekiz adet 32-weight block olarak düzenler.
- Block scale/minimum metadata'sı nedeniyle ortalama temsil yaklaşık
  **4.5 bits per weight (bpw)** olur; yalnızca “4” sayısına bakarak gerçek
  dosya boyutu hesaplanamaz.

`bpw`, **bits per weight** olarak ilk kullanımda açılacaktır.

### `_S`, `_M`, `_L` ne söyler?

Bu son ekler genellikle model düzeyi recipe'nin size/mixed-precision
tercihini anlatır:

- `S` — daha küçük (`small`) tarafa yakın recipe
- `M` — hassas tensorlerin bir bölümünü daha yüksek precision'da tutan dengeli
  (`medium`) recipe
- `L` — bu varyantı olan ailelerde daha yüksek kalite/boyut tarafı

Her quant ailesinde her son ek bulunmaz. Bu harfler evrensel bir dosya
standardı gibi yorumlanmamalıdır; gerçek tensor seçimi llama.cpp sürümüne ve
model mimarisine göre kontrol edilmelidir.

### `Q4_K_M` için doğru cümle

> `Q4_K_M`, çoğu uygun tensor için Q4_K tabanı kullanan; output, attention veya
> FFN gibi hassas tensorlerin bazılarını model ve layer'a göre Q5_K/Q6_K gibi
> daha yüksek precision türlerine çıkarabilen mixed-precision bir llama.cpp
> quantization recipe'sidir.

Dolayısıyla:

- Modelin her tensörü 4-bit değildir.
- Norm, küçük veya uyumsuz tensorler başka türlerde kalabilir.
- Gerçek ortalama bpw ve dosya boyutu model mimarisine göre değişir.
- Güncel llama.cpp CLI, uzaktan model seçerken mevcutsa `Q4_K_M`'yi varsayılan
  quant olarak kullanır; bu onu iyi bir başlangıç noktası yapar, mutlak olarak
  her model ve donanım için “en iyi” yapmaz.

### Sınıfta kullanılacak kısa karşılaştırma

| Etiket | Basit konumlandırma | Sınıfta vurgulanacak nokta |
| --- | --- | --- |
| `Q4_0` | Eski/basit 4-bit block quant | Basit ve yaygın; K-quant ile aynı recipe değildir |
| `Q4_K_S` | Daha küçük Q4 K-quant recipe | Bellek öncelikli, kaliteyi ölç |
| `Q4_K_M` | Dengeli mixed Q4 K-quant recipe | İlk denenecek pratik başlangıç |
| `Q5_K_M` | Daha fazla precision | Kalite için daha fazla bellek |
| `Q6_K` | Yüksek precision K-quant | Quantization kaybını azaltma tarafı |
| `Q8_0` | 8-bit block quant | Yüksek kalite, daha büyük model |
| `IQ4_XS` | Nonlinear/importance-aware 4-bit ailesi | Boyut-kalite iyi olabilir; backend hızını test et |

Bu tablo kalite garantisi veya evrensel benchmark sıralaması değildir.

### `Q8_0` için gösterilecek exact block

```text
32 × signed int8 code = 32 byte
 1 × FP16 scale       =  2 byte
--------------------------------
Q8_0 block            = 34 byte = 8.5 bpw
```

Her 32-value block için:

```text
d = max(abs(x))/127
q = clip(round(x/d), -127, 127)
x_hat = q × stored_FP16(d)
```

Sınıfta özellikle şu üç ifade birbirinden ayrılır:

- `Q8_0`: weight tensor encoding;
- W8A8: 8-bit weight ve activation compute hedefi;
- GGUF: tensorleri taşıyan container.

Uzun pratik için [`q8_0.py`](q8_0.py) tek block/Linear örneğini,
[`q8_demo.py`](q8_demo.py) ise TinyQwen loss, logits, storage, generation ve
save/load karşılaştırmasını çalıştırır.

## 10. Basit seçim stratejisi

Öğrencilere tek bir “en iyi quant” verilmek yerine aşağıdaki karar akışı
öğretilecektir:

1. **Model sığıyor mu?** Weight dosyası yanında runtime overhead ve KV cache
   için de bellek bırak.
2. **Backend destekliyor mu?** CPU, Metal, CUDA, ROCm ve Vulkan aynı quant
   türünde aynı hızı vermeyebilir.
3. **Başlangıç ölçümü:** Local GGUF inference için önce `Q4_K_M` dene.
4. **Kalite yetersizse:** `Q5_K_M`, `Q6_K` veya `Q8_0` ile karşılaştır.
5. **Bellek yetersizse:** `Q4_K_S`, uygun `Q3_K`/`IQ` seçeneklerini kontrollü
   benchmark et veya daha küçük model seç.
6. **Context uzunsa:** Weight quantization'dan ayrı olarak KV-cache belleğini
   ve desteklenen cache quant türlerini incele.
7. **Son kararı ölç:** Aynı prompt/dataset üzerinde kalite, peak memory,
   prompt processing speed ve tokens/s kaydet.

### Senaryo soruları

- 8 GB RAM'e sığmayan model: Önce model boyutu ve runtime payı hesaplanır,
  ardından `Q4_K_M` veya daha küçük seçenek ölçülür.
- Kod üretiminde küçük hatalar pahalı: `Q5_K_M`/`Q6_K` ile kalite farkı
  karşılaştırılır.
- Çok uzun sohbet: Weight dosyası kadar KV-cache büyümesi de hesaplanır.
- GPU'da küçük quant daha yavaş: Backend'in ilgili kernel desteği kontrol edilir;
  yalnızca bit sayısına bakılmaz.

## 11. Canlı demo ilkeleri

[`demo.py`](demo.py) mevcut `qwen3/tiny_qwen.pt` checkpoint'ini kullanır.
Karşılaştırma aynı input ve seed ile yapılır:

- FP32 baseline
- Gerçek INT8 weight storage
- Gerçek packed symmetric INT4 weight storage
- Exact payload/scale metadata byte hesabı
- Logit farkı
- Loss/perplexity farkı
- Üretilen Türkçe isim örnekleri
- Packed state dict save/load kontrolü

Q8'e ayrılmış [`q8_demo.py`](q8_demo.py), dedicated `Q8_0Linear` ile generic
group-32 symmetric INT8'in bütün TinyQwen layer'larında aynı code, FP16 scale,
dequantized weight ve logits ürettiğini de doğrular.

Demo weight'leri gerçekten düşük-bit buffer'larda saklar fakat optimized INT4
kernel kullanmaz. Forward sırasında dequantized weight'lerle PyTorch matmul
yapıldığı için **hız kazancı iddia etmez** ve llama.cpp/GGUF performans
benchmark'ının yerini tutmaz. Storage ile compute kernel farkı hem kod çıktısında
hem ders notunda görünürdür.

## 12. Yaygın yanlış anlamalar

| Yanlış ifade | Derste kurulacak doğru ifade |
| --- | --- |
| “4-bit model tam 4× küçüktür.” | Scale, minimum, metadata ve mixed tensorler de yer kaplar. |
| “Q4_K_M'deki her şey 4-bit.” | Bu, hassas tensorleri daha yüksek precision'da tutabilen mixed recipe'dir. |
| “Q8_0 tam 8 bpw ve lossless'tir.” | FP16 scale metadata'sıyla 8.5 bpw'dir; rounding error vardır. |
| “Q8_0 ise activation/KV cache de 8-bit.” | Q8_0 tensor encoding'dir; compute ve cache dtype ayrı seçilir. |
| “Düşük bit kesin daha hızlıdır.” | Hız kernel, backend, dequantization ve memory bandwidth'e bağlıdır. |
| “GGUF bir quantization algoritmasıdır.” | GGUF container'dır; içinde farklı tensor encoding'leri olabilir. |
| “QLoRA ve Q4_K_M aynı şeydir.” | Biri fine-tuning yaklaşımı, diğeri GGUF inference recipe adıdır. |
| “Dosya açılıyorsa kalite yeterlidir.” | Göreve özel evaluation ve aynı promptlarla karşılaştırma gerekir. |
| “Weight quantization uzun context belleğini çözer.” | KV cache ayrı büyür ve ayrıca ele alınmalıdır. |

## 13. Eğitmen kontrol soruları

Ders boyunca kısa sorular kullanılacaktır:

- `scale` iki kat büyürse quantization adımları daha mı ince, daha mı kaba olur?
- Aralık dışındaki `1.6` neden `1.2` olarak geri geldi?
- `W4A16` içindeki `W` ve `A` neyi temsil eder?
- Static PTQ neden calibration verisine ihtiyaç duyar?
- `Q4_K_M` neden tam 4.0 bpw değildir?
- Daha küçük bir quant neden bazı cihazlarda daha yavaş olabilir?

Final exit ticket:

1. Quantization modelde neyi değiştirir?
2. `Q4_K_M` ne anlama gelir ve ne anlama gelmez?
3. Kendi cihazın için quantization seçerken hangi dört ölçümü yaparsın?

## 14. Kaynak ve doğruluk politikası

Teknik iddialar mümkün olduğunca primary source veya resmi proje
dokümantasyonundan doğrulanacaktır:

- [llama.cpp quantization guide](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)
- [llama.cpp tensor encoding schemes](https://github.com/ggml-org/llama.cpp/wiki/Tensor-Encoding-Schemes)
- [llama.cpp backend feature matrix](https://github.com/ggml-org/llama.cpp/wiki/Feature-matrix)
- [GPTQ paper](https://arxiv.org/abs/2210.17323)
- [AWQ paper](https://arxiv.org/abs/2306.00978)
- [SmoothQuant paper](https://proceedings.mlr.press/v202/xiao23c.html)
- [QLoRA paper](https://arxiv.org/abs/2305.14314)
- [PyTorch quantization documentation](https://docs.pytorch.org/docs/stable/quantization.html)

Özellikle llama.cpp recipe seçimi zamanla değişebileceği için `Q4_K_M`nin
tensor-tensor tam dağılımı sabit bir evrensel tanım gibi ezberletilmeyecektir.
İçeriğin hazırlandığı tarihte güncel kaynak kod ve dokümantasyon yeniden
kontrol edilecektir.

## 15. Rehberin kabul kriterleri

Hazırlanacak ders materyali:

- Her acronym'i ilk kullanımda açmalı.
- Elle yapılan örnekteki tüm ara değerleri göstermeli.
- Simülasyon ile gerçek packed inference arasındaki farkı açıkça belirtmeli.
- `Q4_K_M` için container, encoding ve recipe ayrımını korumalı.
- Kalite ve hız iddialarını ölçüm olmadan genellememeli.
- 90 dakika içinde uygulanabilir olmalı.
- Final üç sorunun cevabını ders içinde açıkça vermeli.
