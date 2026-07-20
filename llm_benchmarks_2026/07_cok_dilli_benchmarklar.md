# 07. Çok Dilli Benchmarklar

> Bu bölüm, büyük dil modellerinin İngilizce dışındaki dillerde ne kadar iyi çalıştığını ölçen benchmarkları ele alır. Çoğu popüler LLM benchmarkı (MMLU, HellaSwag, ARC, GSM8K...) **İngilizce olarak tasarlanmış ve İngilizce veriyle eğitilen modellerin üstünlüğünü ölçmek üzere optimize edilmiştir**. Bu durum iki temel soruna yol açar: (1) diğer dillerdeki performans genelde makine çevirisiyle üretilen, düşük kaliteli test setleriyle ölçülüyor; (2) "kültürel olarak İngilizce/Batı-merkezli" sorular (ör. ABD hukuku, İngilizce deyimler) diğer dillere çevrildiğinde anlamsız veya adaletsiz hale geliyor. Bu bölümdeki benchmarklar tam olarak bu sorunları çözmeye çalışıyor — özellikle son kısımda **Türkçe için özel olarak geliştirilen** kaynaklara derinlemesine yer veriyoruz, çünkü bu doğrudan Türkçe/İngilizce tokenizer ve veri kümesi geliştiren projeler için kritik önemde.

---

## 1. Çok dilli değerlendirmenin iki temel sorunu

1. **Çeviri kalitesi problemi.** Bir MMLU sorusunu makine çevirisiyle Türkçeye çevirmek, çoğu zaman anlamı bozar, terminolojiyi yanlış aktarır veya dilbilgisel olarak tuhaf cümleler üretir. Model bu yüzden düşük puan alırsa, bu "modelin Türkçe bilmediği" anlamına mı gelir, yoksa "test setinin bozuk olduğu" anlamına mı gelir — ayırt etmek zordur.
2. **Kültürel adalet (cultural fairness) problemi.** "ABD Anayasası'nın kaçıncı maddesi..." tarzı bir soru Fransızcaya çevrilse bile özünde Amerikan-merkezlidir; bir Fransız modelin bu soruda düşük performans göstermesi dil yetkinliğiyle değil, kültürel bilgiyle ilgilidir.

Aşağıdaki benchmarkların çoğu, bu iki sorunu farklı stratejilerle (insan doğrulaması, anadilinde-yazma, kültürel etiketleme) çözmeye çalışır.

---

## 2. Global-MMLU — 42 dil, insan tarafından doğrulanmış çeviri

**Global-MMLU**, Cohere For AI liderliğinde geliştirilen ve MMLU'nun 42 dile (İngilizce dahil) genişletilmiş, kültürel önyargıları titizlikle etiketlenmiş versiyonudur.

### Metodoloji

- Klasik yaklaşımların aksine (tamamen otomatik makine çevirisi), Global-MMLU **insan tarafından doğrulanmış çeviriler** kullanır.
- 4 "altın standart" dil için (**Arapça, Fransızca, Hintçe, İspanyolca**) profesyonel, ücretli çevirmenler hem dilsel akıcılığı hem kültürel uygunluğu garanti edecek şekilde çalıştı.
- 11 ek dil için topluluk (community) katkılı çeviri + daha yüksek kaliteli makine çevirisi karışımı kullanıldı.

### Kültürel hassasiyet etiketlemesi

Her dil için 2.850 soruluk bir alt küme, **"Kültürel Olarak Hassas" (Culturally Sensitive)** ya da **"Kültürel Olarak Tarafsız" (Culturally Agnostic)** şeklinde etiketlendi. Bu etiketleme; sorunun kültürel bilgiye, bölgesel özgüllüğe ve lehçe anlayışına ne kadar bağlı olduğuna göre yapıldı.

**Yayımlanmış sonuç:** Modellerin tam MMLU üzerindeki sıralaması, sadece "kültürel olarak hassas" sorular alt kümesinde değerlendirildiğinde **değişiyor** — yani bazı modeller genel MMLU'da güçlü görünse de, kültürel bağlam gerektiren sorularda göreceli olarak zayıflıyor. Makalede bu hassas soruların veri kümesinin kayda değer bir bölümünü (araştırmacıların kendi raporlarına göre önemli bir yüzdesini) oluşturduğu belirtiliyor; kesin yüzde için orijinal makaleye bakılması önerilir.

- Kaynak: [Global MMLU: Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation (arXiv:2412.03304)](https://arxiv.org/abs/2412.03304), [IBM Research özeti](https://research.ibm.com/publications/global-mmlu-understanding-and-addressing-cultural-and-linguistic-biases-in-multilingual-evaluation)

**Beklenen cevap** (kültürel olarak hassas bir örnek soru — illüstratif):
> "Bayramda büyüklerin elini öpme geleneği hangi kültürel bağlamda yaygındır?" → Bu tür bir soru, kelime kelime çevrildiğinde başka bir dile/kültüre aktarıldığında anlam kaybına uğrar; Global-MMLU'nun etiketleme sistemi tam olarak bu tür soruları ayırt etmeyi hedefler.

**Model çıktısı (örnek/illustrative):** Ağırlıklı olarak İngilizce/Batı verisiyle eğitilmiş bir model, kültürel olarak tarafsız (matematik, fizik gibi) sorularda dile bakılmaksızın tutarlı performans gösterebilirken, kültürel olarak hassas sorularda düşük-kaynaklı dillerde performansı belirgin biçimde düşebilir.

---

## 3. MMLU-ProX — daha zor, daha geniş çok dilli MMLU

**MMLU-ProX**, MMLU-Pro'nun (MMLU'nun daha zor, daha az "tahmin edilebilir" versiyonu — 10 seçenekli, akıl yürütme ağırlıklı) çok dilli genişlemesidir.

### Yapı

| Özellik | Değer |
|---|---|
| Dil sayısı | Kaynağa göre 13-29 arası tipolojik olarak çeşitli dil (makalenin farklı versiyonlarında/güncellemelerinde genişletilmiş) |
| Soru sayısı / dil | ~11.829 (tam versiyon) |
| Lite versiyon | 658 soru/dil (verimli değerlendirme için) |
| Çeviri metodolojisi | Birden fazla güçlü LLM ile ön-çeviri + uzman incelemesi (terminoloji tutarlılığı ve kültürel uygunluk için) |

### Neden gerekliydi?

Mevcut çok dilli görevlerin çoğu, **paralel sorular** (tüm dillerde aynı soru, doğrudan karşılaştırılabilir) sunmuyordu — bu da diller arası akıl yürütme yeteneğinin doğrudan kıyaslanmasını zorlaştırıyordu. MMLU-ProX her dilde **birebir aynı 11.829 soruyu** sunarak doğrudan çapraz-dil (cross-lingual) karşılaştırma imkânı sağlıyor.

- Kaynak: [MMLU-ProX: A Multilingual Benchmark for Advanced Large Language Model Evaluation (arXiv:2503.10497)](https://arxiv.org/abs/2503.10497), [Proje sitesi](https://mmluprox.github.io/)

---

## 4. Belebele — 122 dil varyantı, FLORES-200 tabanlı okuma anlama

**Belebele**, Meta AI tarafından geliştirilen ve **122 dil varyantında** paralel çoktan seçmeli okuma anlama testleri sunan bir benchmarktır (ACL 2024).

### Yapı

| Özellik | Değer |
|---|---|
| Dil varyantı | 122 (115 farklı dil, 27 dil ailesi, 29 farklı yazı sistemi/alfabe) |
| Soru/dil | 900 |
| Toplam soru | 109.800 |
| Kaynak pasaj | FLORES-200'den alınan 488 farklı pasaj, her birinden 1-2 soru türetilmiş |
| Format | 4 seçenekli çoktan seçmeli |

Her soru **tam paralel**dir — yani aynı pasaj ve aynı soru seti tüm dil varyantlarına çevrilmiştir, bu da doğrudan çapraz-dil karşılaştırmayı mümkün kılar. Belebele; yüksek, orta ve düşük kaynaklı dillerdeki metin modellerinin değerlendirilmesini kapsamlı biçimde genişletmiştir.

- Kaynak: [The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants (arXiv:2308.16884)](https://arxiv.org/abs/2308.16884)

> Not: Belebele, Türkçeyi de (`tur_Latn`) 122 dil varyantından biri olarak içerir — FLORES-200 tabanlı olduğu için Türkçe metinler de profesyonel çeviri sürecinden geçmiştir; bu da onu düşük-kaliteli otomatik çeviri setlerine kıyasla daha güvenilir bir Türkçe okuma-anlama ölçütü yapar.

---

## 5. TyDi QA — 11 dil, ~204K soru-cevap, **çeviri değil, doğal yazım**

**TyDi QA** (Google Research), diğer birçok çok dilli QA veri kümesinden temel bir yöntem farkıyla ayrılır: sorular **çeviri yoluyla değil**, doğrudan o dilde, **cevabı bilmeyen** kişiler tarafından "gerçekten merak ederek" yazılmıştır.

### Yapı

| Özellik | Değer |
|---|---|
| Dil sayısı | 11 tipolojik olarak çeşitli dil |
| Soru-cevap çifti | ~204.000 (kimi kaynaklarda ~200K olarak yuvarlanır) |
| Toplama yöntemi | Doğal, çeviri kullanılmadan, doğrudan hedef dilde |

### Neden önemli?

Çoğu çeviri-tabanlı QA veri kümesinde soruyu yazan kişi zaten cevabı biliyordur (çünkü kaynak metinden türetilmiştir) — bu da "önyargı etkisi" (priming effect) yaratır: sorular genelde cevaba çok yakın kelimeler içerir, bu da görevi yapaylaştırır. TyDi QA bu önyargıyı ortadan kaldırarak gerçek bir "bilgi arama" (information-seeking) senaryosu simüle eder.

- Kaynak: [TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages (TACL 2020)](https://aclanthology.org/2020.tacl-1.30/), [GitHub - google-research-datasets/tydiqa](https://github.com/google-research-datasets/tydiqa)

**Beklenen cevap** (TyDi QA tarzı, doğal soru örneği — illüstratif):
> Soru (doğal, cevap bilinmeden yazılmış): "Bu şarkının bestecisi kimdir?"
> Bağlam (Wikipedia makalesi): "...şarkı 1962'de X tarafından bestelenmiştir..."
> Beklenen cevap: "X"

---

## 6. XTREME ve XTREME-R — CMU/DeepMind/Google'ın çok görevli çok dilli paketi

### XTREME (2020)

**XTREME**, önceden eğitilmiş çok dilli modellerin **çapraz-dil genelleme** (cross-lingual generalization) becerisini ölçmek için CMU, DeepMind ve Google tarafından geliştirilen bir benchmark paketidir.

| Özellik | Değer |
|---|---|
| Dil sayısı | 40 tipolojik olarak çeşitli dil |
| Dil ailesi sayısı | 12 |
| Görev sayısı | 9 (cümle sınıflandırma, yapılandırılmış tahmin, cümle retrieval, soru-cevaplama) |
| Öne çıkan az-kaynaklı diller | Tamil, Telugu, Malayalam (Dravidian), Swahili, Yoruba (Nijer-Kongo) |

### XTREME-R (2021)

XTREME'in daha zorlu ve ayrıntılı (fine-grained) versiyonudur:

- **50 dile** genişletildi.
- Yeni bir retrieval görevi eklendi: **Mewsli-X**.
- **MultiCheckList** adlı, hataları daha ayrıntılı teşhis eden (fine-grained diagnostic) çok dilli bir test paketi eklendi.
- İnteraktif, kamuya açık bir liderlik tablosu (leaderboard) ile çoklu-veri-kümesi değerlendirmesi sağlandı.

- Kaynak: [XTREME: A Massively Multilingual Multi-task Benchmark (arXiv:2003.11080)](https://arxiv.org/pdf/2003.11080), [XTREME-R (arXiv:2104.07412)](https://ar5iv.labs.arxiv.org/html/2104.07412), [GitHub - google-research/xtreme](https://github.com/google-research/xtreme)

---

## 7. FLORES-200 — makine çevirisi için altın standart

Çoğu benchmarkta yalnızca "kaynak metin" olarak geçen **FLORES-200**, aslında kendi başına bir **makine çevirisi (MT) değerlendirme benchmarkı**dır ve Meta'nın "No Language Left Behind" (NLLB) projesinin parçası olarak 2022'de yayımlanmıştır.

### Yapı

- **3.001 İngilizce cümle** (842 farklı Wikipedia makalesinden alınmış), profesyonel çevirmenlerce **201 diğer dile** (toplam 202 dil) çevrilmiştir — "many-to-many" (her dilden her dile) çeviri değerlendirmesine imkân tanıyan, tam hizalanmış (fully aligned) bir veri kümesidir.
- Tüm çeviriler profesyonel çevirmenler tarafından yapıldığı için referans çeviriler yüksek kalitelidir.

### Değerlendirme metrikleri

| Metrik | Açıklama |
|---|---|
| **BLEU** | Klasik n-gram örtüşme metriği (Papineni ve ark., 2002) |
| **chrF / chrF++** | Karakter-seviyesinde F-skoru; standart yazım biçimi olmayan veya yazım çeşitliliği yüksek diller için (Türkçe gibi eklemeli diller dahil) BLEU'dan daha güvenilir kabul edilir |
| **spBLEU** | BLEU'nun tokenizasyon sınırlamalarını aşmak için çok dilli bir SentencePiece tokenizer ile hesaplanan versiyonu |
| **COMET** | Nöral, öğrenilmiş (learned) bir kalite tahmin metriği; daha geniş görevlerde ek metrik olarak kullanılır |

Matematiksel olarak chrF, karakter n-gram'ları üzerinden hesaplanan bir F-skorudur:

\[
\text{chrF}_{\beta} = (1 + \beta^{2}) \cdot \frac{\text{chrP} \cdot \text{chrR}}{\beta^{2} \cdot \text{chrP} + \text{chrR}}
\]

burada \(\text{chrP}\) karakter n-gram kesinliği (precision), \(\text{chrR}\) karakter n-gram duyarlılığı (recall), \(\beta\) ise recall'a verilen ağırlıktır (chrF++'da genelde \(\beta = 2\) kullanılır).

**Yayımlanmış sonuç:** NLLB-200 modeli, FLORES-200 üzerinde önceki son-teknoloji (state-of-the-art) sistemlere kıyasla ortalama BLEU skorunda **%44** iyileşme sağladı; bazı Afrika ve Hint dilleri için bu iyileşme **%70'i aştı**.

- Kaynak: [No Language Left Behind: Scaling Human-Centered Machine Translation (arXiv:2207.04672)](https://arxiv.org/pdf/2207.04672)

> Not: FLORES-200, Türkçeyi (`tur_Latn`) de içerdiği için, bir Türkçe-İngilizce çeviri modelinin veya çok dilli bir LLM'in Türkçe çeviri kalitesini standart, karşılaştırılabilir biçimde ölçmek isteyenler için doğrudan kullanılabilir bir kaynaktır.

---

## 8. Karşılaştırma tablosu (genel çok dilli benchmarklar)

| Benchmark | Dil sayısı | Görev türü | Çeviri mi, anadilinde mi yazılmış? |
|---|---|---|---|
| Global-MMLU | 42 | Çoktan seçmeli genel bilgi/akıl yürütme (MMLU) | Karışık: 4 dil profesyonel çeviri, 11 dil topluluk çevirisi + üst kalite MT, gerisi MT (insan doğrulamalı) |
| MMLU-ProX | 13-29 (kaynağa göre değişir) | Zor çoktan seçmeli akıl yürütme (MMLU-Pro tabanlı) | LLM ön-çeviri + uzman incelemesi |
| Belebele | 122 (115 farklı dil) | Çoktan seçmeli okuma anlama | Profesyonel çeviri (FLORES-200 tabanlı) |
| TyDi QA | 11 | Açık uçlu bilgi-arama QA | **Anadilinde doğal yazım** (çeviri yok) |
| XTREME | 40 | 9 görev: sınıflandırma, yapılandırılmış tahmin, retrieval, QA | Karışık (görev veri kümesine göre değişir) |
| XTREME-R | 50 | XTREME + retrieval (Mewsli-X) + diagnostik (MultiCheckList) | Karışık |
| FLORES-200 | 201 (+İngilizce = 202) | Makine çevirisi kalite ölçümü (BLEU/chrF/COMET) | Profesyonel çeviri (kaynak metnin kendisi) |

---

## 9. Türkçe LLM Benchmarkları ve Değerlendirme Kaynakları

Bu bölüm, **Türkçe/İngilizce tokenizer ve veri kümesi geliştiren projeler için özellikle kritik** — çünkü "modelim Türkçe'de ne kadar iyi?" sorusuna makine-çevirisi kalitesiz test setleriyle değil, gerçek, güncel kaynaklarla cevap vermek gerekiyor. Aşağıda, 2026 ortası itibarıyla **fiilen kullanılabilir** Türkçe LLM değerlendirme kaynaklarının kapsamlı bir listesi var.

### 9.1 TurkishMMLU (akademik, EMNLP 2024 Findings)

Türkçe için **ilk çok görevli, çoktan seçmeli QA benchmarkı**. Makine çevirisine değil, **uzman yazarlığına** dayanır.

| Özellik | Değer |
|---|---|
| Soru sayısı | 10.000+ |
| Kapsam | Türk lise müfredatından 9 farklı konu (fen bilimleri, matematik, ayrıca Türk Edebiyatı ve Türkiye Cumhuriyeti tarihi gibi kültürel olarak temsili konular dahil) |
| Yazım yöntemi | Müfredat uzmanları tarafından **doğrudan Türkçe** yazılmış — makine çevirisi kullanılmamış |
| Değerlendirilen modeller | 20'den fazla LLM (Gemma, Llama, mT5 gibi çok dilli açık kaynak modeller; GPT-4o, Claude, Gemini gibi kapalı kaynak modeller; Türkçeye uyarlanmış modeller) |

TurkishMMLU'nun asıl vurgusu şu: otomatik çeviriye dayanan çok dilli benchmarklar hataya açıktır ve kültürel olarak yanlı sorular üretebilir; bu yüzden TurkishMMLU uzman yazarlığıyla bu riski baştan ortadan kaldırmayı hedefler.

- Kaynak: [TurkishMMLU: Measuring Massive Multitask Language Understanding in Turkish (arXiv:2407.12402, EMNLP 2024 Findings)](https://arxiv.org/abs/2407.12402)

### 9.2 TR-MMLU (Turkish NLP standardı önerisi, Ocak 2025)

Türk eğitim sistemindeki geniş bir soru havuzundan **titizlikle küçültülmüş** (curated-down) bir benchmark.

| Özellik | Değer |
|---|---|
| Nihai soru sayısı | 6.200 çoktan seçmeli soru |
| Bölüm sayısı | 62 |
| Kaynak havuzu | 280.000 soruluk daha büyük bir havuzdan seçilmiş, 67 disiplin ve 800+ konuyu kapsıyor |
| Kapsam alanları | Hukuk, sağlık, tarih, sanat ve daha fazlası — Türk eğitim sisteminden türetilmiş |

TR-MMLU, şeffaf, tekrarlanabilir (reproducible) ve kültürel açıdan ilgili bir Türkçe değerlendirme aracı sunmayı hedefliyor. Makalenin ikinci bir sürümü ("Büyük Dil Modelleri için TR-MMLU Benchmarkı: Performans Değerlendirmesi, Zorluklar ve İyileştirme Fırsatları", arXiv:2508.13044) 2025 ortasında yayımlanarak performans analizini derinleştirdi.

- Kaynak: [Setting Standards in Turkish NLP: TR-MMLU for Large Language Model Evaluation (arXiv:2501.00593)](https://arxiv.org/abs/2501.00593)

### 9.3 Cetvel — üretken ve kültürel kapasiteyi birleştiren benchmark (EACL 2026)

**Cetvel**, KUIS AI (Koç Üniversitesi) tarafından geliştirilen ve mevcut Türkçe benchmarkların (TurkishMMLU, TR-MMLU gibi) çoğunlukla çoktan seçmeli (MCQA) formatla sınırlı kalmasına karşı geliştirilen, hem **ayırt edici (discriminative)** hem **üretken (generative)** görevleri birleştiren kapsamlı bir pakettir.

| Özellik | Değer |
|---|---|
| Görev sayısı | 23 görev, 7 kategori altında |
| Görev örnekleri | Dilbilgisi hatası düzeltme, makine çevirisi, Türk tarihine dayalı soru-cevap, deyimsel dil anlama |
| Değerlendirilen model sayısı | 33 açık ağırlıklı (open-weight) model, 70B parametreye kadar |

**Yayımlanmış sonuç:** Cetvel değerlendirmesinde çarpıcı bir bulgu şu: **Türkçe-merkezli, talimat ince-ayarı yapılmış (instruction-tuned) modeller**, genellikle Llama 3 veya Mistral gibi genel amaçlı/çok dilli modellere kıyasla **daha düşük** performans gösteriyor — yani "Türkçeye özelleştirilmiş" etiketi tek başına daha iyi performans garantisi vermiyor.

- Kaynak: [Cetvel: A Unified Benchmark for Evaluating Language Understanding, Generation and Cultural Capacity of LLMs for Turkish (arXiv:2508.16431, EACL 2026)](https://arxiv.org/abs/2508.16431), [GitHub - KUIS-AI/cetvel](https://github.com/KUIS-AI/cetvel)

### 9.4 TurkBench — kurumsal ortaklıklarla üretilmiş, sentetik olmayan içerik (EACL 2026 SIGTURK)

**TurkBench**, üretken (generative) LLM'lerin Türkçe becerisini değerlendirmek için tasarlanmış ve verisi **sentetik olarak üretilmeden ya da mevcut literatürden uyarlanmadan**, doğrudan prestijli ulusal kurum ve üniversite bölümleriyle stratejik ortaklıklar yoluyla toplanmış bir benchmarktır.

| Özellik | Değer |
|---|---|
| Örnek sayısı | 8.151 |
| Alt görev sayısı | 21 |
| Ana kategori | 6: Bilgi (Knowledge), Dil Anlama, Akıl Yürütme, İçerik Denetimi (Content Moderation), Türkçe Dilbilgisi ve Kelime Hazinesi, Talimat Takibi (Instruction Following) |
| Değerlendirme | Çevrimiçi gönderim (online submission) sistemiyle huggingface.co/turkbench üzerinden |

- Kaynak: [TurkBench: A Benchmark for Evaluating Turkish Large Language Models (arXiv:2601.07020, EACL 2026 SIGTURK)](https://arxiv.org/abs/2601.07020)

### 9.5 TUMLU — Türkçenin de dahil olduğu Türk dilleri ailesi benchmarkı

**TUMLU (Turkic Massive Language Understanding)**, sadece Türkiye Türkçesini değil, **8 Türk dilini** birlikte, anadilinde (native, çeviri değil) değerlendiren bir benchmarktır.

| Özellik | Değer |
|---|---|
| Diller | Azerbaycan Türkçesi, Kırım Tatarcası, Karakalpakça, Kazakça, Tatarca, **Türkiye Türkçesi**, Uygurca, Özbekçe |
| Soru sayısı | 38.139 çoktan seçmeli soru |
| Konu sayısı | 11 akademik konu (matematik, fen bilimleri, edebiyat, sosyal bilgiler vb.), orta ve lise seviyesi |
| Alt versiyon | TUMLU-mini — daha kısa, dengelenmiş ve elle doğrulanmış alt küme |
| Değerlendirilen modeller | Claude, Gemini, GPT, LLaMA dahil açık ve kapalı kaynaklı geniş bir model yelpazesi |

TUMLU'nun özgün katkısı, MMLU tarzı benchmarkların genelde yüksek kaynaklı dillerden makine çevirisiyle üretilmesine karşı, **Türk dil ailesine özgü morfosentaktik ve kültürel özellikleri** hesaba katarak anadilinde geliştirilmiş olmasıdır — bu, Türkçe ile yakın akraba dillerdeki (Azerbaycan Türkçesi, Özbekçe vb.) transfer öğrenmeyi araştıranlar için de değerli bir kaynaktır.

- Kaynak: [TUMLU: A Unified and Native Language Understanding Benchmark for Turkic Languages (arXiv:2502.11020)](https://arxiv.org/abs/2502.11020)

### 9.6 MMLU-Pro-TR

MMLU-Pro'nun (10 seçenekli, daha zor MMLU versiyonu) Türkçeye uyarlanmış bir değerlendirme kaynağıdır; topluluk tarafından geliştirilen bir GitHub deposu (`bezir/MMLU-pro-TR`) olarak dağıtılmaktadır. Türkçe konuşan araştırmacılar için MMLU-Pro'nun daha zorlayıcı, akıl yürütme ağırlıklı formatını yerelleştirir.

- Kaynak: [GitHub - bezir/MMLU-pro-TR](https://github.com/bezir/MMLU-pro-TR)

### 9.7 lm-evaluation-harness Türkçe çevirileri (`_tr` görev takısı)

EleutherAI'nin standart **lm-evaluation-harness** çerçevesi, çoğu araştırmacının ve şirketin model karşılaştırması için fiilen kullandığı araçtır. Topluluk (malhajar17) bu çerçeveye Türkçe görev setleri ekleyen bir fork/repo yayımladı:

| Görev (İngilizce orijinali) | Türkçe görev adı (harness'te) |
|---|---|
| HellaSwag | `hellaswag_tr` |
| Winogrande | `winogrande_tr` |
| ARC | `arc_tr` |
| GSM8K | `gsm8k_tr` |
| TruthfulQA | `truthful_qa_tr` |
| MMLU | `mmlu_tr` |

Bu veri kümeleri, İngilizce orijinallerinden **çevrilerek ve uyumlulaştırılarak (harmonize)** üretilmiştir — yani TurkishMMLU/TUMLU gibi "anadilinde yazılmış" değil, çeviri-tabanlıdır. Bu nedenle hız/kapsam açısından pratik ama kültürel doğruluk açısından TurkishMMLU/TUMLU/Cetvel gibi kaynaklardan daha zayıf kabul edilmelidir.

```bash
# lm-evaluation-harness ile Türkçe HellaSwag ve ARC değerlendirmesi (örnek kullanım)
git clone https://github.com/malhajar17/lm-evaluation-harness_turkish
cd lm-evaluation-harness_turkish
pip install -e .

lm_eval --model hf \
  --model_args pretrained=my-turkish-model \
  --tasks hellaswag_tr,arc_tr,winogrande_tr,mmlu_tr \
  --device cuda:0 \
  --batch_size 8
```

- Kaynak: [GitHub - malhajar17/lm-evaluation-harness_turkish](https://github.com/malhajar17/lm-evaluation-harness_turkish)

### 9.8 OpenLLM Türkçe Liderlik Tablosu (Hugging Face Space)

Yukarıdaki `_tr` görev setlerini kullanarak modelleri otomatik değerlendirip sıralayan, Hugging Face üzerinde barındırılan bir liderlik tablosudur (malhajar tarafından geliştirilmiş, birden fazla versiyonu mevcut: v1 ve v0.2).

- [OpenLLM Turkish Leaderboard](https://huggingface.co/spaces/malhajar/OpenLLMTurkishLeaderboard)
- [OpenLLM Turkish Leaderboard v0.2](https://huggingface.co/spaces/malhajar/OpenLLMTurkishLeaderboard_v0.2)

Bu liderlik tablosu, modelleri gönderip (submit) topluluğa açık şekilde karşılaştırmalı sonuçlar görmeyi sağlayan pratik, "canlı" bir kaynaktır.

### 9.9 Türkçe MMLU Veri Kümesi ve Liderlik Tablosu (`alibayram`, Hugging Face)

Ayrıca dikkat çekici bir kaynak: Hugging Face üzerinde **`alibayram/turkish_mmlu`** adlı, tamamen özgün (makine çevirisi değil) Türkçe içerikten oluşan büyük ölçekli bir MMLU-tarzı veri kümesi ve buna bağlı bir liderlik tablosu (`alibayram/turkish_mmlu_leaderboard`) bulunuyor.

| Özellik | Değer |
|---|---|
| Soru sayısı | 293.468 |
| Bölüm sayısı | 67 |
| Konu sayısı | 800+ |
| İçerik kaynağı | Türkiye'deki önemli sınavlar dahil (TUS — Tıpta Uzmanlık Sınavı, KPSS — Kamu Personeli Seçme Sınavı gibi) tamamen özgün Türkçe içerik |
| Liderlik tablosu durumu | 50 modelden 51 giriş, 6.200 yanıt (ölçüm anına göre değişebilir) |

Bu kaynak, TR-MMLU ile benzer bir felsefeyi (geniş, özgün Türkçe sınav havuzundan seçim) paylaşıyor ama farklı bir kaynak/küratasyon sürecine dayanıyor ve canlı bir Hugging Face Space üzerinden model gönderimi/karşılaştırması sağlıyor.

- [Turkish MMLU Leaderboard (Hugging Face Space)](https://huggingface.co/spaces/alibayram/turkish_mmlu_leaderboard)
- [alibayram/turkish_mmlu (veri kümesi)](https://huggingface.co/datasets/alibayram/turkish_mmlu)

> Not: Türkçe/İngilizce tokenizer ve veri kümesi üzerinde çalışan projeler için bu kaynak özellikle ilgi çekici olabilir — hem büyük ölçekli özgün Türkçe soru havuzu, hem de doğrudan model karşılaştırması yapılabilecek canlı bir liderlik tablosu sunuyor.

### 9.10 TARA — RAG-odaklı Türkçe değerlendirme

**TARA (Turkish benchmark for RAG-related capabilities)**, modellerin sağlanan bağlamı ne kadar etkin kullandığını ve bağlama ne kadar sadık kaldığını (RAG prensiplerine uygunluk) Türkçe için ölçen bir veri kümesidir. `emre/TARA_Turkish_LLM_Benchmark` adıyla Hugging Face'te barındırılıyor.

- [emre/TARA_Turkish_LLM_Benchmark (Hugging Face)](https://huggingface.co/datasets/emre/TARA_Turkish_LLM_Benchmark)

### 9.11 Türkçe benchmark ekosisteminin özet karşılaştırması

| Kaynak | Yıl | Soru/örnek sayısı | Yazım yöntemi | Odak |
|---|---|---|---|---|
| TurkishMMLU | 2024 (EMNLP Findings) | 10.000+ | Uzman yazarlığı (anadilinde) | Lise müfredatı, MCQA |
| TR-MMLU | 2025 (Ocak) | 6.200 (280K havuzundan seçilmiş) | Türk eğitim sistemi kaynaklı, küratasyonlu | Genel MCQA, 67 disiplin |
| Cetvel | 2025/2026 (EACL 2026) | 23 görev, 7 kategori | Karışık (bazıları uzman yazarlığı) | MCQA + üretken görevler + kültür |
| TurkBench | 2026 (EACL SIGTURK) | 8.151, 21 alt görev | Kurumsal ortaklıklarla, sentetik olmayan | Üretken LLM değerlendirmesi, talimat takibi |
| TUMLU | 2025 | 38.139 (8 Türk dili) | Anadilinde, native | Türk dil ailesi, MCQA |
| MMLU-Pro-TR | topluluk | MMLU-Pro kapsamı | Çeviri | Zor akıl yürütme MCQA |
| `_tr` lm-eval-harness setleri | topluluk | HellaSwag/ARC/Winogrande/GSM8K/MMLU kapsamı | Çeviri | Hızlı otomatik karşılaştırma |
| `alibayram/turkish_mmlu` | topluluk | 293.468 | Özgün Türkçe (TUS, KPSS dahil) | Geniş kapsamlı MCQA + canlı liderlik tablosu |
| TARA | topluluk | değişken | Türkçe RAG-odaklı | Bağlama sadakat / RAG değerlendirmesi |

### 9.12 Pratik öneri: Türkçe/İngilizce tokenizer ve veri kümesi projeleri için

Bir Türkçe/İngilizce ikili modelin (bilingual model) veya tokenizer'ın kalitesini değerlendirirken:

1. **Hızlı, otomatik regresyon testi** için → lm-evaluation-harness'in `_tr` görev setleri (`hellaswag_tr`, `arc_tr`, `mmlu_tr` vb.) — CI/CD pipeline'a kolayca entegre edilebilir.
2. **Kültürel/dilbilgisel doğruluk odaklı, "gerçek Türkçe" testi** için → TurkishMMLU, TR-MMLU veya `alibayram/turkish_mmlu` gibi anadilinde/özgün içerikli kaynaklar — çünkü tokenizer'ın Türkçenin eklemeli (agglutinative) yapısını (ör. uzun ek zincirleri, ünlü uyumu) ne kadar iyi işlediği, ancak gerçek Türkçe metinlerle ortaya çıkar.
3. **Üretken (generative) yeterlilik** (sadece çoktan seçmeli değil, serbest metin üretimi, dilbilgisi hatası düzeltme, çeviri) için → Cetvel veya TurkBench.
4. **RAG / bağlama sadakat** odaklı bir sistem geliştiriliyorsa → TARA + `06_uzun_baglam_ve_rag.md` dosyasındaki RAGBench/TRACe çerçevesini Türkçe verilerle uyarlamak.
5. **Türk dil ailesi ile transfer/genelleme** araştırılıyorsa (ör. Azerbaycan Türkçesi, Özbekçe gibi yakın dillere aktarım) → TUMLU.
6. **Ham tokenizer verimliliği** (fertility, token/kelime oranı) için doğrudan bir "benchmark" olmasa da, FLORES-200'ün Türkçe (`tur_Latn`) bölümü hem çeviri kalitesi hem de tokenizasyon verimliliği ölçümü için hazır, standart bir paralel metin kaynağı sağlar.

**Beklenen cevap** (TurkishMMLU tarzı, Türk Edebiyatı sorusu — illüstratif):
> Soru: "Tanzimat Dönemi Türk edebiyatının ilk romanı kabul edilen eser aşağıdakilerden hangisidir?"
> Seçenekler: A) İntibah B) Taaşşuk-ı Talat ve Fitnat C) Araba Sevdası D) Cezmi
> Beklenen cevap: B) Taaşşuk-ı Talat ve Fitnat

**Model çıktısı (örnek/illustrative):** Ağırlıklı İngilizce veriyle eğitilmiş, Türkçe verisi sınırlı bir model, bu tür kültürel/edebi bilgi sorularında genel dilbilgisi sorularına kıyasla belirgin biçimde daha düşük doğruluk gösterme eğilimindedir — TurkishMMLU ve Cetvel'in tam olarak yakalamayı hedeflediği fark budur.

---

## 10. Çok dilli skorların hesaplanması: makro-ortalama ve "dil boşluğu" (language gap)

Çok dilli benchmarklarda tek bir toplam skor yerine, dil-bazlı kırılım ve bir **dil boşluğu** metriği raporlamak çok daha bilgilendiricidir.

### Makro-ortalama (macro-average)

\[
\text{Skor}_{\text{makro}} = \frac{1}{|\mathcal{L}|} \sum_{\ell \in \mathcal{L}} \text{Acc}_{\ell}
\]

burada \(\mathcal{L}\) değerlendirilen dillerin kümesi, \(\text{Acc}_{\ell}\) dil \(\ell\) için doğruluktur. Bu, her dile eşit ağırlık verir — düşük kaynaklı bir dilin skoru, yüksek kaynaklı bir dille aynı ağırlıkta toplam skora katkı sağlar (örnek sayısına göre ağırıklandırılan mikro-ortalamadan farklı olarak).

### Dil boşluğu (language gap)

\[
\Delta_{\text{gap}} = \text{Acc}_{\text{en}} - \frac{1}{|\mathcal{L}\setminus\{en\}|} \sum_{\ell \in \mathcal{L}\setminus\{en\}} \text{Acc}_{\ell}
\]

Bu metrik, bir modelin İngilizce performansı ile diğer dillerdeki ortalama performansı arasındaki farkı somutlaştırır. Global-MMLU, MMLU-ProX ve OneRuler gibi çalışmaların ortak bulgusu şu: \(\Delta_{\text{gap}}\) genelde **pozitif ve anlamlı büyüklüktedir** — yani çoğu model İngilizce'de sistematik olarak daha iyi performans gösterir — ama OneRuler'ın gösterdiği gibi bu her zaman kesin bir kural değildir (İngilizce'nin 26 dil arasında 6. sırada çıktığı durum gibi istisnalar var).

### Düşük-kaynak / yüksek-kaynak ayrımı

Çoğu çok dilli benchmark raporunda diller kabaca üç gruba ayrılır:

| Grup | Tipik örnekler | Özellik |
|---|---|---|
| Yüksek kaynaklı | İngilizce, Çince, İspanyolca, Fransızca, Almanca | Eğitim verisinde bol miktarda temsil |
| Orta kaynaklı | **Türkçe**, Lehçe, Vietnamca, Endonezce | Makul miktarda web verisi var, ama İngilizce'ye kıyasla çok daha az |
| Düşük kaynaklı | Swahili, Yoruba, Karakalpakça, Kırım Tatarcası | Eğitim verisinde çok kısıtlı temsil |

Türkçe, çoğu çok dilli LLM'in ön-eğitim karışımında (pretraining mixture) genelde "orta kaynaklı" kategoride yer alır — bu da Türkçe için değerlendirme sonuçlarının, en yüksek kaynaklı dillere kıyasla daha büyük varyans göstermesi ve model-aile bazında daha tutarsız olması anlamına gelir (bazı modeller Türkçede sürpriz derecede iyi, bazıları beklenenden kötü performans gösterebilir).

---

## 11. Tokenizer verimliliği ve çok dilli değerlendirme arasındaki bağlantı

Çok dilli benchmark sonuçlarını yorumlarken sıkça atlanan bir faktör: **tokenizer fertility** (bir kelimenin ortalama kaç token'a bölündüğü). Türkçe gibi eklemeli (agglutinative) bir dilde, İngilizce-merkezli bir tokenizer kelimeleri aşırı parçalayabilir — bu hem bağlam penceresinin etkin kapasitesini (aynı anlamsal içerik için daha fazla token harcanır) hem de model performansını dolaylı olarak etkiler.

\[
\text{Fertility}(\ell) = \frac{\text{toplam token sayısı}}{\text{toplam kelime sayısı}} \quad \text{(dil } \ell \text{ için, sabit bir referans metin kümesi üzerinde)}
\]

Bir modelin çok dilli benchmark performansını değerlendirirken, düşük skorun **dil anlama eksikliğinden** mi yoksa **tokenizer'ın o dili verimsiz parçalamasından** mı (ör. aynı bağlam penceresine daha az anlamsal içerik sığması) kaynaklandığını ayırt etmek önemlidir. Bu, doğrudan bir "benchmark" olmasa da, Türkçe/İngilizce tokenizer geliştiren projeler için FLORES-200'ün paralel metinleri (bkz. §7) fertility ölçümü için hazır ve standart bir kaynak sunar — aynı cümlenin İngilizce ve Türkçe token sayıları karşılaştırılarak tokenizer'ın diller arası dengesi nicel olarak ölçülebilir.

```python
# Basitleştirilmiş fertility karşılaştırma örneği (illüstratif)
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("my-tr-en-tokenizer")

def fertility(text: str) -> float:
    n_words = len(text.split())
    n_tokens = len(tokenizer.encode(text))
    return n_tokens / n_words

# FLORES-200'den paralel bir cümle çifti (örnek/illustrative)
en_sentence = "The committee approved the new regulations after months of debate."
tr_sentence = "Komite, aylar süren tartışmaların ardından yeni yönetmelikleri onayladı."

print(f"EN fertility: {fertility(en_sentence):.2f}")
print(f"TR fertility: {fertility(tr_sentence):.2f}")
# Yayımlanmamış/illüstratif beklenti: TR fertility genelde EN'den belirgin biçimde
# yüksek çıkar çünkü ekler (-lerin, -dükleri gibi) alt-kelime parçalarına bölünür;
# kesin oran kullanılan tokenizer'ın Türkçe eğitim verisi miktarına bağlıdır.
```

---

## 12. lm-evaluation-harness ile çok dilli/Türkçe toplu değerlendirme örneği

Aşağıdaki örnek, bu dosyada bahsedilen birden fazla kaynağı (küresel çok dilli + Türkçeye özel) tek bir komut dizisinde nasıl birleştirebileceğinizi gösterir (illüstratif kullanım — güncel görev adları için ilgili harness deposunun güncel dokümantasyonuna bakılmalıdır):

```bash
# 1) Küresel çok dilli karşılaştırma (Global-MMLU + Belebele, Türkçe alt kümesi dahil)
lm_eval --model hf \
  --model_args pretrained=my-tr-en-model \
  --tasks global_mmlu_tr,belebele_tur_Latn \
  --device cuda:0 --batch_size 8

# 2) Türkçeye özel çeviri-tabanlı hızlı regresyon seti
lm_eval --model hf \
  --model_args pretrained=my-tr-en-model \
  --tasks hellaswag_tr,arc_tr,winogrande_tr,mmlu_tr,gsm8k_tr \
  --device cuda:0 --batch_size 8

# 3) Sonuçları karşılaştırmalı bir tabloya dökmek için (illüstratif özet betiği)
python summarize_results.py --results_dir ./lm_eval_results --languages tr,en --format markdown
```

**Beklenen cevap:** Her görev için 0-1 aralığında doğruluk/EM (exact match) skoru içeren bir JSON çıktısı.

**Model çıktısı (örnek/illustrative):** Aynı model ailesinin farklı büyüklükteki versiyonları arasında, İngilizce görevlerdeki (`hellaswag`, `arc`) skor farkı genelde küçükken, Türkçe versiyonlarında (`hellaswag_tr`, `arc_tr`) fark daha belirgin olabilir — bu, küçük modellerin sınırlı Türkçe ön-eğitim verisiyle daha kırılgan bir dil temsili öğrendiğine işaret eder.

---

## 13. Sık yapılan yorumlama hataları (çok dilli değerlendirmede)

1. **"Model X dilinde iyi, o zaman genel olarak çok dillidir."** — Tek bir dildeki güçlü performans, diğer dillere genellemeyi garanti etmez; özellikle yazı sistemi (script) farklıysa (Latin vs. Kiril vs. Arap alfabesi) performans keskin biçimde değişebilir.
2. **"Çeviri-tabanlı bir test setinde düşük skor = model dili bilmiyor."** — Düşük skor, çeviri kalitesizliğinden de kaynaklanabilir (bkz. §1); TurkishMMLU/TUMLU gibi anadilinde yazılmış kaynaklarla çapraz doğrulama yapılmadan bu sonuca varılmamalı.
3. **"42 dilde ortalama %80 = her dilde ~%80."** — Makro-ortalama, dil-bazlı büyük varyansı gizleyebilir; bir model 10 dilde %95, 32 dilde %70 alarak da aynı makro-ortalamaya ulaşabilir.
4. **"Kültürel olarak hassas sorularda düşük skor = model önyargılı."** — Bu bazen doğru olsa da, bazen sadece "o kültüre özgü bilginin eğitim verisinde yeterince temsil edilmemesi" anlamına gelir; Global-MMLU'nun kültürel etiketleme sistemi bu ayrımı yapmayı kolaylaştırır ama otomatik olarak çözmez.
5. **"Bir Türkçe benchmarkı geçmek yeterli, hepsini kullanmaya gerek yok."** — §9.11'deki tabloda görüldüğü gibi, TurkishMMLU (uzman yazarlığı, MCQA), Cetvel (üretken görevler), TurkBench (talimat takibi) ve TUMLU (Türk dil ailesi transferi) farklı yetkinlikleri ölçer; tek bir kaynakla "Türkçe biliyor" sonucuna varmak eksik bir değerlendirmedir.

---

## 14. Kaynakça

- [Global MMLU (arXiv:2412.03304)](https://arxiv.org/abs/2412.03304)
- [MMLU-ProX (arXiv:2503.10497)](https://arxiv.org/abs/2503.10497)
- [Belebele (arXiv:2308.16884)](https://arxiv.org/abs/2308.16884)
- [TyDi QA (ACL Anthology, TACL 2020)](https://aclanthology.org/2020.tacl-1.30/)
- [XTREME (arXiv:2003.11080)](https://arxiv.org/pdf/2003.11080)
- [XTREME-R (arXiv:2104.07412)](https://ar5iv.labs.arxiv.org/html/2104.07412)
- [No Language Left Behind / FLORES-200 (arXiv:2207.04672)](https://arxiv.org/pdf/2207.04672)
- [TurkishMMLU (arXiv:2407.12402)](https://arxiv.org/abs/2407.12402)
- [TR-MMLU (arXiv:2501.00593)](https://arxiv.org/abs/2501.00593)
- [TR-MMLU v2 / Performans Değerlendirmesi (arXiv:2508.13044)](https://arxiv.org/abs/2508.13044)
- [Cetvel (arXiv:2508.16431)](https://arxiv.org/abs/2508.16431)
- [TurkBench (arXiv:2601.07020)](https://arxiv.org/abs/2601.07020)
- [TUMLU (arXiv:2502.11020)](https://arxiv.org/abs/2502.11020)
- [GitHub - bezir/MMLU-pro-TR](https://github.com/bezir/MMLU-pro-TR)
- [GitHub - malhajar17/lm-evaluation-harness_turkish](https://github.com/malhajar17/lm-evaluation-harness_turkish)
- [OpenLLM Turkish Leaderboard](https://huggingface.co/spaces/malhajar/OpenLLMTurkishLeaderboard)
- [Turkish MMLU Leaderboard (alibayram)](https://huggingface.co/spaces/alibayram/turkish_mmlu_leaderboard)
- [TARA Turkish LLM Benchmark](https://huggingface.co/datasets/emre/TARA_Turkish_LLM_Benchmark)

---

## 15. Ekosistemin genel görünümü ve 2026 itibarıyla açık boşluklar

Bu dosyada taranan kaynakları bir araya getirdiğimizde, Türkçe LLM değerlendirme ekosisteminin 2023-2026 arasında hızla olgunlaştığı görülüyor: 2023'te fiilen elde tek bir güvenilir, anadilinde yazılmış genel-amaçlı Türkçe MCQA kaynağı yokken, 2026 ortası itibarıyla TurkishMMLU, TR-MMLU, Cetvel, TurkBench ve TUMLU gibi birbirini tamamlayan beş akademik kaynak ve en az üç canlı Hugging Face liderlik tablosu (OpenLLM Turkish Leaderboard, Turkish MMLU Leaderboard, TurkBench'in çevrimiçi gönderim sistemi) mevcut.

Yine de göze çarpan bazı boşluklar var:

- **Uzun-bağlam × Türkçe kesişimi zayıf.** `06_uzun_baglam_ve_rag.md` dosyasında incelenen RULER/BABILong/Context-Rot tarzı testlerin Türkçeye özel, anadilinde yazılmış bir versiyonu (OneRuler'ın 26 dilinden biri olarak Türkçe dahil olsa da, ayrı ayrı raporlanmış detaylı bir Türkçe kırılım kamuya açık değil) henüz yok — bu, uzun-bağlamlı Türkçe RAG sistemleri geliştirenler için hâlâ bir araştırma boşluğu.
- **Konuşma/diyalog-tabanlı Türkçe değerlendirme sınırlı.** Mevcut kaynakların büyük kısmı MCQA (çoktan seçmeli) formatında; TurkBench ve Cetvel bu açığı üretken görevlerle kısmen kapatıyor ama çok turlu (multi-turn) diyalog değerlendirmesi hâlâ görece az kaynak içeriyor.
- **Güvenlik/zararlı içerik (safety) değerlendirmesi.** TurkBench'in "İçerik Denetimi" (Content Moderation) kategorisi bu yönde bir adım, ama İngilizce ekosistemdeki (ör. SafetyBench, ToxiGen tarzı) kapsamlı Türkçe karşılıkları henüz sınırlı.

Türkçe/İngilizce tokenizer ve veri kümesi geliştiren bir proje için pratik sonuç: **tek bir benchmarka güvenmek yerine**, §9.12'deki öneriler doğrultusunda hızlı otomatik regresyon (`_tr` görev setleri), kültürel doğruluk (TurkishMMLU/TR-MMLU/`alibayram/turkish_mmlu`) ve üretken yeterlilik (Cetvel/TurkBench) kaynaklarının bir kombinasyonunu kullanmak, günümüzün en sağlam değerlendirme stratejisidir.

---

*Önceki dosya: [`06_uzun_baglam_ve_rag.md`](./06_uzun_baglam_ve_rag.md) — uzun bağlam ve RAG benchmarkları.*
