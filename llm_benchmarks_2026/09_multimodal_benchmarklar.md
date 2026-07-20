# 9. Multimodal Benchmarklar (Görsel-Dil ve Video)

**Güncelleme tarihi:** 21 Temmuz 2026

Bu dosya, [`llm_benchmarks_guide_2026.md`](../llm_benchmarks_guide_2026.md) rehberinin multimodal (çok kipli) benchmark bölümünün genişletilmiş ve güncellenmiş sürümüdür. Multimodal benchmark'lar, bir modelin yalnızca metni değil; görüntüyü, belgeyi, grafiği, ekran görüntüsünü veya videoyu **doğru okuyup üzerine doğru muhakeme yapabildiğini** ölçer. Bu, salt dil benchmarklarından farklı bir hata kaynağı ekler: model matematiği doğru bilse bile, görüntüyü yanlış okursa madde yanlış sayılır.

Aynı kural burada da geçerlidir:

- **Beklenen cevap**, veri kümesindeki gold cevaptır.
- **Model çıktısı**, puanlama mekanizmasını göstermek için oluşturulmuş **açıklayıcı** bir örnektir.
- **"Yayımlanmış sonuç"**, yalnızca bir makale, teknik rapor veya resmî liderlik tablosunda gerçekten raporlanmış bir sayı için kullanılır.
- Bazı benchmark'ların görsel test verileri (özellikle telif hakkı içeren görüntüler) doğrudan yeniden üretilemediğinden, o bölümlerde gerçek görev biçimine sadık küçük örnekler kullanılmıştır.

---

## 9.0. Multimodal değerlendirmenin ek zorluğu

Bir metin benchmarkında hata iki yerden gelebilir: bilgi eksikliği veya muhakeme hatası. Multimodal bir benchmarkta buna üçüncü bir katman eklenir: **algı hatası** (görüntüyü yanlış okuma). Bu üç katmanı ayırt etmek zordur:

1. Model görüntüyü doğru okudu mu? (OCR / nesne tanıma / grafik okuma başarısı)
2. Model doğru okuduğu bilgiyi doğru birleştirdi mi? (muhakeme)
3. Model doğru muhakeme ettiği hâlde doğru biçimde cevap yazdı mı? (biçim/çıktı ayıklama sorunu)

İyi tasarlanmış bir multimodal benchmark, bu üç katmanı olabildiğince ayırmaya çalışır — örneğin metin-only bir modelin de çözebileceği soruları filtreleyerek (MMMU-Pro'nun yaptığı gibi) veya görüntü olmadan cevaplanamayacak sorular tasarlayarak.

---

## 9.1. MME — erken ve hâlâ referans alınan bir temel benchmark

**MME (Multimodal LLM Evaluation)**, multimodal büyük dil modellerinin **algı (perception)** ve **biliş (cognition)** yeteneklerini birlikte ölçen, erken dönemin en etkili benchmark'larından biridir.

### Yapısı

MME, **14 alt görevden** oluşur ve ikiye ayrılır:

- **Algı görevleri (10 alt görev):** varlık tespiti (existence), sayma (count), konum (position), renk (color), afiş (poster), ünlü tanıma (celebrity), sahne (scene), simge yapı/yer (landmark), sanat eseri (artwork) ve OCR.
- **Biliş görevleri (4 alt görev):** sağduyu muhakemesi (commonsense reasoning), sayısal hesaplama (numerical calculation), metin çevirisi (text translation) ve kod muhakemesi (code reasoning).

### Tasarımdaki önemli detay: veri sızıntısını önleme

MME ekibi, halka açık veri kümelerini doğrudan kullanmak yerine bütün soru-cevap çiftlerini **elle tasarlamıştır**. Ayrıca istemler kasıtlı olarak çok kısa ve standarttır (örn. "Görüntüde bir [nesne] var mı? Yalnızca Evet veya Hayır ile cevap ver."); amaç, modellerin istem mühendisliğiyle (prompt engineering) avantaj elde etmesini engelleyip adil bir karşılaştırma sağlamaktır.

### Örnek görev (existence)

Görüntüde bir masa, bir sandalye ve bir kedi bulunuyor.

Soru:

```text
Görüntüde bir köpek var mı? Yalnızca Evet veya Hayır ile cevapla.
```

**Beklenen cevap:**

```text
Hayır
```

**Model çıktısı:**

```text
Evet
```

Bu tür sorular basit görünse de, modelin görüntüde **olmayan** bir şeyi var zannetmesi (halüsinasyon) yaygın bir hata modudur ve MME'nin "existence" alt görevi doğrudan bunu ölçer.

### Puanlama

Her alt görevde hem "Evet/Hayır" doğruluğu hem de aynı görüntü için sorulan çift-yönlü soru çiftinin **ikisinin de** doğru cevaplanıp cevaplanmadığı (accuracy+) ayrı ayrı raporlanır. Bu, modelin şansla doğru "Evet" veya "Hayır" deme olasılığını azaltmaya yöneliktir.

### Neden hâlâ önemli?

MME, sonradan gelen MMMU, MathVista gibi daha "akademik" benchmark'ların aksine, çok temel ve ucuza tekrarlanabilir algı görevlerine odaklanır. Bu yüzden yeni bir modelin "temel görüş yeteneklerinde bir gerileme olup olmadığını" hızlıca kontrol etmek için hâlâ model kartlarında sıkça raporlanır.

---

## 9.2. SEED-Bench — üretici anlama (generative comprehension) odaklı geniş kapsam

**SEED-Bench**, çoktan seçmeli sorularla multimodal modellerin **üretici anlama** yeteneğini ölçen, MME'den daha geniş kapsamlı bir başka erken/orta dönem benchmark'tır.

### Kapsamı

- **19.000 çoktan seçmeli soru**, insan tarafından doğrulanmış.
- **12 değerlendirme boyutu**, hem görüntü hem video kipini kapsar (örn. sahne anlama, örnek/varlık düzeyinde tanıma, örnek nitelikleri, örnek sayma, uzamsal ilişki, örnek etkileşimi, görsel akıl yürütme; video tarafında ise eylem tanıma, eylem tahmini, prosedür anlama gibi zamana bağlı boyutlar).
- Sorular, görüntü/video bilgisinden GPT-4 benzeri bir modele özel istemlerle ürettirilip ardından otomatik filtreleme ve insan denetiminden geçirilerek oluşturulmuştur.

### Neden çoktan seçmeli tasarım?

SEED-Bench'in tasarım felsefesi, açık uçlu üretimi (ki bu bir hakem LLM'e ihtiyaç duyar ve öznelliğe açıktır) devre dışı bırakıp, **objektif ve tekrarlanabilir** bir puanlama sunmaktır: doğru seçenek insan tarafından belirlenmiştir, ekstra bir hakem modele gerek yoktur.

### Örnek görev (uzamsal ilişki boyutu)

Görüntüde bir kitap masanın üstünde, bir çanta ise sandalyenin altında duruyor.

Soru:

```text
Kitapla çanta arasındaki konumsal ilişki hangisidir?
A) Kitap çantanın üstünde
B) Kitap çantanın solunda
C) Kitap çantadan daha yüksekte
D) Kitap çantanın içinde
```

**Beklenen cevap:** C

**Model çıktısı:**

```text
A
```

Madde puanı: 0. Model muhtemelen "üstünde" ile "daha yüksekte" ifadelerini karıştırmış, yani konumsal ilişkiyi doğru okumuş ama yanlış seçeneğe eşlemiş olabilir — bu da çoktan seçmeli tasarımın bazen ince nüansları kaybettiğini gösterir.

---

## 9.3. MMMU ve MMMU-Pro

**MMMU (Massive Multi-discipline Multimodal Understanding)**, üniversite düzeyinde uzmanlık gerektiren, gerçek ders materyallerinden derlenmiş bir benchmark'tır:

- **11.500 soru**
- **6 ana disiplin** (Sanat & Tasarım, İş, Sağlık & Tıp, Bilim, Beşerî Bilimler & Sosyal Bilim, Teknoloji & Mühendislik)
- **30 ders, 183 alt alan**
- **30 farklı görsel türü** (grafik, şema, harita, tıbbi görüntü, nota, teknik çizim, tablo vb.)

### MMMU-Pro: üç katmanlı sağlamlaştırma

MMMU'nun doygunlaşmaya başlaması ve bazı soruların görüntüye hiç bakılmadan (yalnızca metinden) çözülebildiğinin fark edilmesi üzerine, MMMU-Pro üç değişiklik uygulamıştır:

1. **Metin-only filtreleme:** Yalnızca metinle (görüntü olmadan) doğru cevaplanabilen sorular veri kümesinden çıkarılmıştır.
2. **Seçenek sayısının artırılması:** Klasik 4 seçenek yerine **10 seçenek** kullanılır; bu, rastgele tahmin başarısını %25'ten %10'a düşürür.
3. **Vision-only (yalnızca görsel) girdi ayarı:** Soru ve seçenekler ayrı bir metin olarak değil, doğrudan **bir ekran görüntüsü/fotoğraf içine gömülü** olarak sunulur. Model hem "görmek" hem "okumak" zorunda kalır.

### Yayımlanmış sonuç: zorluk artışının etkisi

MMMU-Pro makalesinde raporlanan sonuçlara göre:

- Seçenek sayısının 4'ten 10'a çıkarılması tek başına GPT-4o'nun doğruluğunu **%10,7 puan** düşürmüştür.
- Vision-only girdi ayarı ek bir düşüşe yol açmıştır: GPT-4o için **%4,3 puan** ek düşüş, LLaVA-OneVision-72B için **%14,0 puan** ek düşüş.
- Genel olarak modellerin MMMU-Pro'daki doğruluğu, klasik MMMU'ya kıyasla **%16,8 ile %26,9 puan arasında** düşmüştür.

Bu, MMLU → MMLU-Pro geçişinde görülen düşüş örüntüsüne çok benzer: zorluk artırıldıkça, "yüzeysel şans + ezber" payı azalır ve gerçek yetenek farkları ortaya çıkar.

### Görev örneği

Bir devre şeması gösterilir; soru ve 10 seçenek görüntünün içine gömülüdür (vision-only ayar).

Soru (görüntü içinde):

```text
Anahtar kapatıldığında R2 direncinden geçen akım kaç amperdir?
A) 0,5 A   B) 1 A   C) 1,5 A   D) 2 A   E) 2,5 A
F) 3 A     G) 3,5 A H) 4 A     I) 4,5 A J) 5 A
```

**Beklenen cevap:** D (2 A)

**Model çıktısı:**

```text
H
```

Madde puanı: 0. Model doğru Kirchhoff denklemini bilse bile devredeki paralel bağlantıyı görüntüden yanlış okumuşsa hata burada oluşur — yani hata "biliş" katmanında değil "algı" katmanındadır.

---

## 9.4. MathVista

**MathVista**, görsel matematik problemlerini bir araya getiren bir benchmark'tır:

- **6.141 görsel matematik problemi**
- **28 mevcut veri kaynağından** derlenmiş ve **3 yeni veri kaynağı** ile genişletilmiş
- Görsel türleri: fonksiyon grafiği, geometri şekli, tablo, diyagram, bilimsel çizim, istatistiksel grafik

### Örnek

Grafikte:

```text
2023 satış: 120 birim
2024 satış: 150 birim
```

Soru:

```text
Satışlar yüzde kaç artmıştır?
```

Hesap:

\[
(150-120)/120 \times 100=\%25
\]

**Beklenen cevap:** %25

**Model çıktısı** (grafikten 2023 değerini yanlış okuyan bir model):

```text
Grafikte 2023 değeri yaklaşık 100 görünüyor, bu yüzden artış
(150-100)/100 = %50'dir.
```

Madde puanı: 0. Hata matematik adımından değil, **önce görsel okuma aşamasından** kaynaklanmıştır — MathVista'nın asıl ölçmeye çalıştığı da tam olarak bu ayrımdır: model matematiği biliyor mu, yoksa görseli mi yanlış okuyor?

---

## 9.5. OCRBench ve OCRBench v2

**OCRBench**, görseldeki metni okuma ve bu metni **kullanma** (yalnızca tanıma değil) yeteneğini ölçen bir benchmark'tır. İlk sürümü **1.000 insan doğrulamalı** soru-cevap çifti ve 5 OCR bileşenini kapsar.

### OCRBench v2: kapsam genişlemesi

OCRBench v2, öncekine göre 4 kat daha fazla görev türü içerir ve şu şekilde genişler:

- **10.000 insan doğrulamalı** soru-cevap çifti (öncekinin 10 katı)
- **31 farklı senaryo** (sokak tabelası, fatura/makbuz, formül, diyagram, form, el yazısı vb.)
- **8 temel yetenek** ölçülür: metin tanıma, metne referans verme (text referring), metin bulma/işaretleme (text spotting), ilişki çıkarımı (relation extraction), eleman ayrıştırma (element parsing), matematiksel hesaplama, görsel metin anlama ve bilgiye dayalı akıl yürütme.
- **6 sıkı değerlendirme metriği** kullanılır.
- Veri, 81 metin-yoğun akademik veri kümesinden ve ek özel (private) verilerden derlenmiştir.
- Genelleme yeteneğini test etmek için ayrıca **1.500 görüntülük gizli (private) bir test kümesi** bulunur — bu kümenin cevapları hiçbir zaman kamuya açıklanmaz, böylece modellerin ona özel eğitilmesi engellenir.

### Örnek

Bir faturada:

```text
Ara toplam: 1.250 TL
KDV: 250 TL
Toplam: 1.500 TL
```

Soru:

```text
Ödenecek toplam tutar nedir?
```

**Beklenen cevap:**

```text
1.500 TL
```

**Model çıktısı:**

```text
1.250 TL
```

Madde puanı: 0. Model metni doğru okumuş (karakterleri doğru tanımış) fakat "ara toplam" ile "toplam" alanlarını **yapısal olarak ayıramamıştır**. Bu, OCRBench'in salt karakter tanımadan (OCR) daha ileri gidip **belge yapısını anlama** (element parsing, relation extraction) yeteneğini neden ayrıca ölçtüğünü gösterir.

---

## 9.6. ChartQA ve DocVQA — grafik ve belge anlamaya özel benchmark'lar

MMMU ve MathVista genel amaçlı olsa da, grafik (chart) ve belge (document) anlama o kadar sık karşılaşılan ve o kadar farklı beceriler gerektiren iki alandır ki, kendi özel benchmark'larını doğurmuşlardır.

### ChartQA

ChartQA, grafik görüntüleri üzerinde soru-cevap yeteneğini ölçer:

- **9,6 bin insan tarafından yazılmış soru**
- **23,1 bin**, insan yazımı grafik özetlerinden otomatik türetilmiş ek soru
- Grafikler Statista, Pew Research, Our World in Data (OWID) ve OECD gibi gerçek kaynaklardan toplanmıştır; çubuk, çizgi ve pasta grafiklerini kapsar.
- Sorular iki türdedir: **compositional** (mantıksal/matematiksel işlem gerektiren, örn. "İki değer arasındaki fark yüzde kaçtır?") ve **visual** (grafiğin görsel özelliğine dair, örn. "En yüksek çubuğun rengi nedir?").

**Örnek:**

Çubuk grafikte 2020: 45, 2021: 60, 2022: 55 değerleri gösteriliyor.

Soru:

```text
2021 ile 2022 arasındaki değer farkı kaçtır?
```

**Beklenen cevap:**

```text
5
```

**Model çıktısı:**

```text
15
```

Madde puanı: 0. Model muhtemelen 2020 ile 2022'yi karıştırmıştır — grafik okuma hatasının doğrudan sonucu.

### DocVQA

DocVQA, taranmış/dijital belge görüntüleri üzerinde soru-cevap yeteneğini ölçer:

- **50.000 soru**
- **12.767 farklı endüstriyel belge** (form, mektup, rapor, fatura vb.), UCSF kütüphanesi kaynaklı bir külliyattan derlenmiştir.

**Örnek:**

Bir form görüntüsünde "Başvuru Tarihi: 14/03/2025" alanı var.

Soru:

```text
Başvuru hangi tarihte yapılmıştır?
```

**Beklenen cevap:**

```text
14/03/2025
```

**Model çıktısı:**

```text
2025
```

Kısmi doğru olsa da katı Exact Match ölçütüyle bu genellikle 0 puan alır (gün ve ay eksik) — bu, ana rehberin 1.2 bölümünde anlatılan Exact Match normalizasyon tartışmasının belge anlama alanındaki karşılığıdır.

### ChartQA/DocVQA'nın MMMU'dan farkı

MMMU ve MathVista genel akademik/bilimsel görselleri kapsarken, ChartQA ve DocVQA özellikle **kurumsal/iş dünyası kullanım senaryolarını** (finansal rapor okuma, form doldurma, fatura işleme) hedefler. Bu yüzden bir "belge işleme asistanı" ürünü değerlendirilirken MMMU'dan çok bu ikisi daha belirleyicidir.

---

## 9.7. Video-MME ve Video-MME-v2

### Video-MME

**Video-MME**, video anlamaya adanmış ilk kapsamlı multimodal LLM benchmark'larından biridir:

- **900 video**
- **2.700 soru-cevap çifti**
- Video süreleri **11 saniyeden yaklaşık 1 saate** kadar değişir; kısa/orta/uzun olarak üç alt kümeye ayrılır.
- Video görüntüsü yanında altyazı (subtitle) ve ses kullanımını da ayrı ayrı değerlendirebilir (yalnızca görüntü / görüntü + altyazı / görüntü + ses koşulları).

### Örnek

Videoda: (1) kişi kırmızı kutuyu masaya koyar, (2) mavi kutuyu dolaba kaldırır, (3) kırmızı kutuyu pencerenin yanına taşır.

Soru:

```text
Videonun sonunda kırmızı kutu nerededir?
```

**Beklenen cevap:**

```text
Pencerenin yanında.
```

**Model çıktısı** (yalnızca videonun ortasını dikkate alan bir model):

```text
Masanın üzerinde.
```

Madde puanı: 0.

### Video-MME-v2: 2026'nın yeni nesil video benchmark'ı

Nisan 2026'da yayımlanan **Video-MME-v2**, orijinal Video-MME'nin en üst düzey modellerde doygunlaşmaya başlamasına (leaderboard skorlarının yükselmesine rağmen gerçek kullanıcı deneyimiyle skor arasında büyüyen bir boşluk gözlenmesine) tepki olarak geliştirilmiştir.

Öne çıkan tasarım farkları:

- **3.300'den fazla insan-saati** annotasyon ile, 12 anotatör ve 50 bağımsız denetçiden oluşan sıkı bir kalite güvence sürecinde (5 tura kadar denetim) oluşturulmuştur.
- **Üç kademeli (tri-level) aşamalı zorluk hiyerarşisi:** çoklu-nokta görsel bilgi birleştirme → zamansal dinamik modelleme → karmaşık multimodal muhakeme.
- **Grup-tabanlı doğrusal olmayan değerlendirme:** klasik soru-başına-doğruluk yerine, birbiriyle ilişkili soru gruplarında **tutarlılık** ve çok adımlı muhakemede **bütünlük** aranır; parçalı veya şans eseri doğru cevaplar cezalandırılır.

**Yayımlanmış sonuç:** Video-MME-v2 makalesinde raporlanan en dikkat çekici bulgulardan biri, değerlendirme sırasında en iyi performans gösteren modelin (Gemini-3-Pro) ile insan uzmanları arasındaki uçurumdur:

\[
\text{Gemini-3-Pro: } 49{,}4 \quad \text{vs.} \quad \text{İnsan uzmanlar: } 90{,}7
\]

Bu, orijinal Video-MME'de en iyi modellerin insan performansına çok daha yakın göründüğü döneme kıyasla önemli bir farktır ve şunu gösterir: **zorluk artırıldığında (daha uzun videolar, daha tutarlı çok adımlı sorular), video anlama alanındaki insan-model açığı sanıldığından çok daha büyük çıkabilir.**

---

## 9.8. Video-MME'nin ötesinde: uzun video anlama benchmark'ları

Video-MME'nin en uzun videoları ~1 saate kadar çıksa da, gerçek dünyadaki birçok video (ders kaydı, spor maçı, güvenlik kamerası, film) çok daha uzundur. Bu ihtiyaçla birlikte 2024-2025 döneminde birkaç "saat ölçekli" (hour-scale) video benchmark'ı ortaya çıkmıştır:

| Benchmark | Video uzunluğu | Öne çıkan özellik |
|---|---|---|
| **LongVideoBench** | ~1 saate kadar, 3.763 video | 6.678 çoktan seçmeli soru, 17 kategori; zamansal bilgiyi geri çağırma ve analiz etmeye odaklanır |
| **MLVU** | 3 dakika – 2 saat | 9 değerlendirme görevi: konu muhakemesi, anormallik tespiti, video özetleme, olay örgüsü (plot) soru-cevabı |
| **LVBench** | Aşırı uzun videolar (saatler) | 2025'te yayımlanmış, "extreme long video understanding" için tasarlanmış |
| **HLV-1K** | Saat ölçekli (hour-long) | Zamana-özgü (time-specific) sorular; örn. LLaVA-Video 72B modeli genel skor olarak ~78,93 almıştır (Ocak 2025, yayımlanmış sonuç) |

### Neden bu ayrım önemli?

Bir video modeli, kısa klipleri (Video-MME'nin "short" alt kümesi) çok iyi anlarken, saatlik bir videonun ortasındaki bir olayı hatırlamakta ciddi şekilde zorlanabilir — tıpkı metin modellerinde "uzun bağlam iddiası" ile "uzun bağlamda gerçekten doğru muhakeme" arasındaki farkın (bkz. RULER, Needle-in-a-Haystack) video karşılığı gibi. Video anlamada da "kaç saatlik video destekliyorum" iddiası ile "o videonun 40. dakikasındaki detayı doğru hatırlıyorum" arasında ciddi bir fark olabilir.

---

## 9.9. ScreenSpot / ScreenSpot-Pro — multimodal ajanlar için GUI benchmark'ı

2025-2026 döneminde multimodal modellerin yalnızca soru cevaplamak için değil, **bilgisayar/telefon arayüzlerini kullanan ajanlar** olarak kullanılması yaygınlaştı (bkz. ana rehberin WebArena/OSWorld bölümü). Bu ajanların temel bir alt-yeteneği, bir ekran görüntüsünde **doğru arayüz elemanını bulmaktır (GUI grounding)** — bu, ScreenSpot ailesinin ölçtüğü şeydir.

### ScreenSpot

- **1.272 test örneği**, mobil, masaüstü ve web arayüzlerini kapsar (geliştirme araçları, yaratıcı yazılımlar, ofis uygulamaları, işletim sistemi menüleri gibi çeşitli uygulama alanlarından).
- Her örnek bir ekran görüntüsü ve hedef arayüz elemanının doğal dil açıklamasından oluşur; model elemanın **hassas 2 boyutlu koordinatını** tahmin etmelidir.

### ScreenSpot-v2 ve ScreenSpot-Pro

- **ScreenSpot-v2**, orijinal veri kümesindeki anotasyon hatalarının elle düzeltildiği, daha güvenilir bir sürümdür.
- **ScreenSpot-Pro**, 1.581 ek ve çok daha zor test örneğiyle kapsamı genişletir: karmaşık arayüz düzenleri, belirsiz eleman açıklamaları ve yüksek çözünürlüklü **profesyonel** yazılım arayüzleri (CAD programları, IDE'ler, profesyonel video/ses düzenleme araçları gibi) üzerinde test yapılır — genel tüketici uygulamalarından çok daha zorlayıcıdır.

### Örnek görev

Ekran görüntüsünde bir e-tablo uygulaması açık; sağ üstte küçük bir "Paylaş" butonu var.

Talimat:

```text
"Paylaş" düğmesine tıkla.
```

**Beklenen çıktı:** buton merkezinin piksel koordinatı, örn. `(1042, 58)`

**Model çıktısı** (açıklayıcı):

```text
(1042, 210)
```

Model doğru butonu (adını) tanımış olsa da, koordinatı butonun altında kalan başka bir elemana denk gelecek şekilde yanlış tahmin etmiştir — bu, klasik metin/görsel soru-cevap benchmark'larında hiç karşılaşılmayan, **piksel-düzeyi hassasiyet gerektiren** bir hata türüdür.

### Ne ölçer, ne ölçmez?

ScreenSpot ailesi, bir ajanın "nereye tıklayacağını bulma" becerisini izole olarak ölçer; ancak bir ekran görüntüsünde doğru elemanı bulmak, o elemana tıklamanın **doğru bir sonraki adım** olduğunu bilmekle aynı şey değildir (bu, WebArena/OSWorld gibi tam görev tamamlama benchmark'larının işidir). Bu yüzden ScreenSpot, GUI ajanı değerlendirme zincirinde genellikle **ilk aşama** (grounding) olarak, WebArena/OSWorld ise **son aşama** (uçtan uca görev başarısı) olarak kullanılır.

---

## 9.10. Karşılaştırma tablosu

| Benchmark | Modalite | Soru/örnek sayısı | Neyi hedefler |
|---|---|---:|---|
| MME | Görüntü | 14 alt görev (elle tasarlanmış) | Temel algı (existence/count/OCR) + biliş (muhakeme, hesaplama) |
| SEED-Bench | Görüntü + video | 19.000 çoktan seçmeli | Geniş kapsamlı üretici anlama, 12 boyut |
| MMMU / MMMU-Pro | Görüntü (akademik) | 11.500 soru | Üniversite düzeyi uzmanlık, disiplinler arası muhakeme |
| MathVista | Görüntü (matematik) | 6.141 problem | Görsel matematik: grafik/geometri/diyagram okuma + hesap |
| OCRBench v2 | Görüntü (metin-yoğun) | 10.000 QA, 31 senaryo | OCR + belge yapısını anlama, 8 yetenek |
| ChartQA | Grafik (chart) | ~32,7 bin soru | Grafik okuma + mantıksal/matematiksel işlem |
| DocVQA | Belge (document) | 50.000 soru, 12.767 belge | Taranmış/dijital belgelerde alan bulma |
| Video-MME | Video (11 sn – 1 sa) | 900 video, 2.700 QA | Kısa/orta/uzun video anlama, ses/altyazı katkısı |
| Video-MME-v2 | Video (uzun, çok kademeli) | 3.300+ insan-saati annotasyon | Tutarlılık odaklı, çok adımlı video muhakemesi |
| LongVideoBench / MLVU / LVBench / HLV-1K | Video (saat ölçekli) | Değişken (bin+ soru) | Saatlik videolarda zamana-özgü hatırlama ve olay örgüsü takibi |
| ScreenSpot / ScreenSpot-Pro | Ekran görüntüsü (GUI) | 1.272 + 1.581 örnek | Arayüz elemanını piksel hassasiyetiyle bulma (grounding) |

---

## 9.11. Multimodal benchmarkların ölçemedikleri

- **Gerçek zamanlı video akışı (streaming):** Çoğu video benchmark'ı önceden kaydedilmiş, tam videoyla çalışır. Bir modelin canlı bir video akışını izleyip **anında** tepki vermesi (örn. bir güvenlik kamerasını izlerken uyarı üretmesi) farklı bir yetenektir ve ayrı, henüz olgunlaşmakta olan bir değerlendirme alanıdır.
- **Çoklu görüntü/video birleşik muhakeme:** Birçok benchmark tek bir görüntü veya tek bir video üzerine kuruludur. Birden fazla görseli (örn. bir ürünün farklı açılardan 5 fotoğrafı) birlikte karşılaştırarak muhakeme yapma çok daha az test edilmiştir.
- **Kültürel ve bölgesel görsel bağlam:** MMMU, MathVista gibi büyük benchmark'lar büyük ölçüde Batı/İngilizce kaynaklı görsellerle inşa edilmiştir. Türkçe bağlamda yerel tabela, resmî belge formatı veya kültürel görsel referansları içeren sorular bu benchmark'larda neredeyse hiç yoktur.
- **Ajan bağlamında uzun-vadeli tutarlılık:** ScreenSpot gibi benchmark'lar tek bir "doğru elemanı bulma" anını ölçer; bir ajanın 20 adımlık bir GUI görevinde tutarlı kalıp kalmadığı (bkz. WebArena/OSWorld) ayrı bir sorudur ve GUI-grounding skorunun yüksek olması, uçtan uca görev başarısını garanti etmez.
- **Güvenlik ve kötüye kullanım riski:** Bir modelin ekran görüntüsünü doğru okuyup doğru koordinatı bulması (ScreenSpot başarısı), aynı zamanda bu yeteneğin kötüye kullanılabileceği (örn. istenmeyen otomasyon, CAPTCHA aşma) anlamına da gelebilir; mevcut benchmark'lar bu riski ölçmez.

---

## Kaynaklar / ek okuma

- MME makalesi: arXiv 2306.13394 — *"MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models"*
- SEED-Bench makalesi: arXiv 2307.16125 — *"SEED-Bench: Benchmarking Multimodal LLMs with Generative Comprehension"*
- MMMU-Pro makalesi: arXiv 2409.02813 — *"MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark"*
- ChartQA makalesi: arXiv 2203.10244 — *"ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning"*
- OCRBench v2 makalesi: arXiv 2501.00321 — *"OCRBench v2: An Improved Benchmark for Evaluating Large Multimodal Models on Visual Text Localization and Reasoning"*
- Video-MME resmî sayfası ve GitHub deposu (MME-Benchmarks/Video-MME)
- Video-MME-v2 makalesi: arXiv 2604.05015 — *"Video-MME-v2: Towards the Next Stage in Benchmarks for Comprehensive Video Understanding"*
- ScreenSpot-Pro makalesi: arXiv 2504.07981 — *"ScreenSpot-Pro: GUI Grounding for Professional High-Resolution Computer Use"*
- HLV-1K deposu: GitHub Vincent-ZHQ/HLV-1K-Long-Video-Understanding-Benchmark
