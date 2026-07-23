# Quantization Öğrenme Modülü — Uygulama ve Doğrulama Planı

Durum: **uygulandı; final doğrulama bu dosyadaki test contract'ına göre
yapılır.**

Bu modül yalnız 90 dakikalık bir sunum değil; quantization matematiğini,
packing'i, PTQ/QAT'yi, yaygın LLM yöntemlerini ve gerçek model etkisini
pratik ederek öğrenmek için hazırlanmıştır. Mevcut model kaynakları ve
checkpoint'ler değiştirilmez.

## 1. Hedefler

Modülün sonunda okuyucu:

- affine quantization'ı elle ve kodla uygulayabilir;
- symmetric/asymmetric qparam hesaplayabilir;
- group size, metadata ve error trade-off'unu ölçebilir;
- gerçek signed/unsigned INT4 packing yapabilir;
- float weight kopyası olmayan bir quantized Linear kurabilir;
- PTQ ile QAT akışını çalıştırabilir;
- Q4_K storage yapısını ve Q4_K_M recipe farkını açıklayabilir;
- Q8_0'ın 32-code/FP16-scale block yapısını byte düzeyinde uygulayabilir;
- generic INT8, Q8_0 ve W8A8 arasındaki farkı açıklayabilir;
- NF4, GPTQ, AWQ ve SmoothQuant'ın ayırt edici fikirlerini deneyebilir;
- TinyQwen'de storage, loss, perplexity, logits ve generation farkını ölçebilir.

## 2. Dosyalar

| Dosya | Sorumluluk |
| --- | --- |
| `README.md` | Ana dokümantasyon, komutlar, ölçülmüş sonuçlar |
| `quantization_turkce_anlatim.md` | Uzun Türkçe teori ve implementation walkthrough |
| `CHEAT_SHEET.md` | Formüller, yöntemler, GGUF ve seçim akışı |
| `EXERCISES.md` | Matematik/kod/senaryo soruları ve cevap anahtarı |
| `core.py` | Qparams, affine quantization, pack/unpack, dequantization, STE |
| `linear.py` | `QuantizedLinear`, `QATLinear`, model-tree replacement |
| `q8_0.py` | Exact 32-value/34-byte Q8_0 structure ve `Q8_0Linear` |
| `q8_demo.py` | TinyQwen Q8_0 evaluation, INT8 equivalence, save/load |
| `q4_k.py` | 256-weight/144-byte Q4_K yapısal implementation |
| `nf4.py` | NF4 codebook, block scale, packed indices |
| `by_hand.py` | Sekiz elle kontrol edilebilir ve assert edilen örnek |
| `advanced_methods.py` | GPTQ/AWQ/SmoothQuant ayırt edici toy implementation'ları |
| `qat_demo.py` | Direct PTQ ve QAT→packed INT4 karşılaştırması |
| `demo.py` | TinyQwen FP32/INT8/INT4 model karşılaştırması |
| `GUIDELINE.md` | 90 dakikalık öğretim stratejisi |
| `PLAN.md` | Bu implementation ve validation contract'ı |

## 3. Implementation kararları

### Core affine quantization

- Desteklenen bitler: INT4 ve INT8.
- Symmetric aralıklar: `[-7,7]` ve `[-127,127]`.
- Asymmetric aralıklar: `[0,15]` ve `[0,255]`.
- Qparams FP32 hesaplanır; storage için scale dtype seçilebilir.
- Tamamen sıfır group division-by-zero üretmez.
- Padding asymmetric min/max istatistiğine katılmaz.
- Rounding half-away-from-zero olarak explicit uygulanır.

### Storage

- INT8 payload gerçek `int8`/`uint8` buffer'dır.
- INT4 payload iki code/`uint8` olacak şekilde gerçek nibble packing'dir.
- Symmetric quantization zero-point saklamaz.
- Asymmetric zero-point `int16` saklanır ve metadata hesabına katılır.
- `QuantizedTensor` original float weight'i saklamaz.

### Quantized Linear

- `QuantizedLinear` içinde float `weight` Parameter yoktur.
- Forward portable reference olarak unpack/dequantize + `F.linear` yapar.
- Bu path storage'ı gerçekten küçültür; optimized integer speedup iddia etmez.
- Packed buffers `state_dict` ile save/load edilebilir.

### QAT

- `QATLinear`, trainable FP master weight saklar.
- Forward'da group-wise fake quantization uygular.
- Backward için straight-through estimator kullanır.
- `to_quantized()` FP master'ı gerçek low-bit storage'a dönüştürür.

### TinyQwen

- Checkpoint: `qwen3/tiny_qwen.pt`.
- Device: CPU.
- Quantized: 14 attention/MLP Linear.
- Excluded: tied `lm_head`/embedding ve 1D norm weight'leri.
- Varsayılan group size: `32`.
- Scale storage: FP16.
- Evaluation: aynı 64 deterministic corpus window.
- Generation seed: `2026`.

### Q4_K

- 256-value super-block.
- 8 × 32 sub-block.
- 128-byte nibble payload.
- 12-byte packed 6-bit scale/min codes.
- FP16 `d` ve `dmin`.
- Exact size: 144 byte = 4.5 bpw.
- Parameter fitting eğitim amaçlı min/max yaklaşımıdır; llama.cpp byte
  compatibility veya GGUF writer iddia edilmez.

### Q8_0

- 32-value block.
- 32-byte signed INT8 payload.
- Bir FP16 scale (`d`) = 2-byte metadata.
- Exact size: 34 byte = 8.5 bpw.
- Symmetric range: `[-127,127]`; zero-point yoktur.
- Code FP32 scale ile seçilir, scale FP16 saklanır.
- Quantized row last dimension 32'ye bölünmelidir; sessiz padding yapılmaz.
- `Q8_0Linear` float weight kopyası saklamaz.
- Generic symmetric group-32 INT8 + FP16 scale ile code/scale equivalence
  executable olarak doğrulanır.
- GGUF writer veya optimized Q8_0 matmul kernel iddia edilmez.

### NF4

- Standard 16-level non-uniform codebook.
- 64-value block.
- FP16 absmax per block.
- Packed 4-bit codebook indices.
- Double quantization ve fused CUDA kernel kapsam dışıdır.

### Advanced yöntemler

- GPTQ: bir row üzerinde Hessian/inverse-Cholesky error compensation.
- AWQ: activation-aware channel scale grid search.
- SmoothQuant: `XW = (X/s)(W*s)` transform ve W8A8 error ölçümü.
- Bunlar distinguishing-math experiment'leridir, production converter değildir.

## 4. Ölçümler

TinyQwen için:

- quantize edilen layer ve weight sayısı;
- payload, metadata ve total weight bytes;
- effective bpw ve FP32 compression ratio;
- bütün model unique weight storage;
- cross-entropy loss;
- perplexity;
- mean/max absolute logit drift;
- aynı seed ile generation;
- packed state dict reload logit equality.

Toy örneklerde:

- exact hand values ve MSE;
- pack/unpack equality;
- group size error/metadata sweep;
- Q8_0 exact 34-byte block, code/scale dtype ve 8.5 bpw;
- Q8_0 block-wise scale ile global INT8 outlier karşılaştırması;
- TinyQwen Q8_0/generic INT8 code, scale, weight ve logit equality;
- Q4_K exact size;
- STE gradient;
- PTQ/QAT accuracy;
- uniform INT4/NF4 distribution comparison;
- naive/advanced calibration output MSE.

## 5. Sınırlar

- Büyük harici model indirilmez.
- llama.cpp build edilmez.
- GGUF reader/writer eklenmez.
- Fused CPU/GPU low-bit kernel yazılmaz.
- Python timing production tokens/s olarak sunulmaz.
- Exact architecture-specific Q4_K_M source logic kopyalanmaz.
- Full GPTQ/AWQ/QLoRA framework yerine öğretici distinguishing core uygulanır.
- Mevcut model/training/checkpoint dosyaları değiştirilmez.

## 6. Otomatik doğrulama

Repository root'undan:

```bash
.venv/bin/python quantization/by_hand.py
.venv/bin/python quantization/q8_0.py
.venv/bin/python quantization/q8_demo.py --show-layers
.venv/bin/python quantization/demo.py --show-layers
.venv/bin/python quantization/qat_demo.py
.venv/bin/python quantization/nf4.py
.venv/bin/python quantization/advanced_methods.py
```

Ek core testleri:

```bash
PYTHONPATH=quantization .venv/bin/python -m unittest discover \
  -s quantization/tests -v
```

Beklenen invariants:

- symmetric/asymmetric INT4 ve INT8 round-trip shape'i korur;
- signed/unsigned pack→unpack integer code'u birebir korur;
- zero tensor finite kalır;
- odd last dimension padding doğru kesilir;
- `QuantizedLinear` state dict içinde float weight yoktur;
- INT4 storage INT8'den, INT8 FP32'den küçüktür;
- TinyQwen INT8 mean logit drift INT4'ten küçüktür;
- packed state dict reload logits farkı sıfırdır;
- Q8_0 block tam 34 byte/8.5 bpw ve zero block exact zero'dır;
- `Q8_0Linear` float weight saklamaz ve state reload exact output verir;
- Q8_0 ile generic group-32 INT8 TinyQwen representation'ı identical'dır;
- Q4_K 256 weight için tam 144 byte/4.5 bpw'dir;
- NF4 normal sample'da uniform INT4 MSE'den düşük çıkar;
- advanced toy experiment'ler naive baseline'ı iyileştirir;
- bütün script assert'leri geçer.

## 7. Doküman doğrulama

- Her Python/Markdown link hedefi vardır.
- Her acronym ilk kullanımda açılır veya glossary'de tanımlıdır.
- `by_hand.py` çıktıları dokümandaki sayılarla eşleşir.
- TinyQwen README tablosu gerçek `demo.py` çıktısıyla eşleşir.
- Q8_0 tabloları `q8_0.py` ve `q8_demo.py` çıktılarıyla eşleşir.
- Q4_K, Q4_K_M ve GGUF birbirine karıştırılmaz.
- Real packed storage ile optimized compute kernel sınırı her ana belgede yazılıdır.
- Kaynaklar original paper veya resmi proje dokümantasyonudur.
- `GUIDELINE.md` ders süreleri toplamı 90 dakikadır.

## 8. Tamamlanma kriterleri

Çalışma ancak:

1. bütün planlanan dosyalar mevcut,
2. bütün otomatik komutlar exit code 0,
3. unit test suite geçiyor,
4. dokümante sayılar runtime çıktısıyla eşleşiyor,
5. checkpoint/model kaynakları değişmemiş,
6. `git diff --check` temiz

olduğunda tamamlandı kabul edilir.
