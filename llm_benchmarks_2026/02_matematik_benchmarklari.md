# 2. Matematik Benchmarkları — Kapsamlı Rehber

**Tarih:** 21 Temmuz 2026

Bu belge, [Güncel LLM Benchmarkları: 2026 İçin Kapsamlı Rehber](../llm_benchmarks_guide_2026.md) ana dokümanının matematik bölümünü genişleten, bağımsız okunabilir bir alt dosyadır. Aşağıdaki kurallar geçerlidir:

- **Beklenen cevap**, veri kümesindeki doğru cevap, yani *gold answer*'dır.
- **Model çıktısı**, puanlama mekanizmasını göstermek için verilmiş somut (gerçek veya gerçekçi) bir model cevabıdır.
- **"Yayımlanmış sonuç"** denildiğinde, bir makalede, teknik raporda veya resmî bir liderlik tablosunda gerçekten raporlanmış bir sayıdan söz edilir; kaynağı dipnot olarak belirtilir.
- Bazı benchmarkların soruları veri sızıntısını önlemek amacıyla tamamen paylaşılmadığından, o bölümlerde gerçek görev biçimine sadık **temsilî** örnekler kullanılmıştır. Bu örnekler açıkça "temsilî örnek" olarak işaretlenmiştir.
- Uydurma istatistik yoktur: Bir sayı buradaysa ya arama yoluyla doğrulanmış bir kaynağa dayanır ya da açıkça "örnek/temsilî" olarak etiketlenmiştir.

---

## İçindekiler

1. GSM8K
2. GSM1K ve kirlenme sorunu
3. MATH, MATH-500 ve doygunluk
4. AIME (2025-2026 güncel skorlarla)
5. HARP
6. Putnam-AXIOM
7. UGMathBench ve U-MATH
8. OlympiadBench
9. MathArena
10. FrontierMath
11. Matematik olimpiyatlarında AI: IMO 2025 vakası
12. Puanlayıcı tuzakları: sembolik eşdeğerlik
13. Karşılaştırma tablosu
14. Genel çıkarımlar

---

# 1. GSM8K

GSM8K (Grade School Math 8K), OpenAI tarafından 2021'de yayımlanmış, yaklaşık 8.500 ilkokul ve ortaokul düzeyinde fakat birden fazla aritmetik işlem gerektiren sözel matematik problemi içeren bir veri kümesidir. Eğitim kümesi ~7.500, test kümesi **1.319** sorudan oluşur.

Sorular kasıtlı olarak "kolay matematik, zor okuma ve çok adımlı akıl yürütme" felsefesiyle tasarlanmıştır: hiçbir soru lise düzeyinde matematik bilgisi gerektirmez, ama doğru sırayla 2-8 aritmetik adım uygulanmalıdır.

## 1.1. Gerçek örnek

Veri kümesinin en sık alıntılanan ilk örneklerinden biri:

> Natalia nisan ayında 48 arkadaşına klips satıyor. Mayısta bunun yarısı kadar satıyor. İki ayda toplam kaç klips satmıştır?

Hesap:

\[
48/2=24
\]

\[
48+24=72
\]

**Beklenen cevap:** 72.

## 1.2. Model çıktısı

```text
Mayısta 24 klips satmıştır.
Toplam 48 + 24 = 72.
#### 72
```

Cevap ayıklayıcı (answer extractor) `####` işaretinden sonraki son sayısal değeri alır. Madde puanı: **1**.

Model şöyle derse:

```text
48 + 48/2 = 96
```

sonuç yanlış olduğundan puan **0**'dır. Ara açıklamanın uzun veya ikna edici olması puanı değiştirmez — GSM8K yalnızca nihai sayısal cevaba bakar, akıl yürütme adımlarının kendisini ayrıca puanlamaz (bu, aşağıda göreceğimiz "Teacher-Forced Accuracy" gibi daha yeni metriklerin neden ortaya çıktığının bir nedenidir).

## 1.3. Toplam skor

1.319 test sorusunun 1.100'ü doğruysa:

\[
1100/1319\approx\%83{,}4
\]

## 1.4. GSM8K'nin sınırları

GSM8K 2021'de yayımlandığı için, 2023 sonrası eğitilen hemen hemen bütün büyük modellerin ön-eğitim verisinde (web taramaları, GitHub kopyaları, akademik alıntılar üzerinden) bir şekilde yer alma ihtimali yüksektir. Bu, "**eğitim-test kirlenmesi**" (train-test contamination) sorununu doğurur: model soruyu çözmüyor, ezberliyor olabilir. Skorun yüksek olması tek başına genellenebilir aritmetik akıl yürütme kanıtı sayılamaz.

---

# 2. GSM1K ve kirlenme sorunu

## 2.1. Neden çıktı?

GSM8K sorularının eğitim verilerine karıştığı şüphesi somut bir akademik çalışmayla test edildi: **"A Careful Examination of Large Language Model Performance on Grade School Arithmetic"** (Zhang, Da ve ark., arXiv:2405.00332, Mayıs 2024, NeurIPS 2024 Datasets & Benchmarks Track).

Yazarlar, GSM8K ile:

- Zorluk düzeyi,
- İnsan çözme oranı,
- Adım sayısı,
- Cevap büyüklüğü

bakımından eşleştirilmiş, fakat **hiçbir yerde yayımlanmamış** yaklaşık **1.000 yeni soru** (GSM1K) hazırladılar. Amaç, GSM8K'daki yüksek skorun gerçekten genellenebilir aritmetik beceri mi, yoksa ezber/aşırı uyum mu olduğunu ölçmekti.

## 2.2. Yayımlanmış sonuç

Makalenin temel bulguları:

- **Phi ailesi ve Mistral gibi bazı modellerde GSM8K → GSM1K geçişinde skor düşüşü %13'e kadar çıkmıştır** — bu, sistematik aşırı uyumun güçlü bir işaretidir.
- **GPT ve Claude ailesindeki frontier modeller çok daha küçük düşüş göstermiştir** (bazı sürümlerde neredeyse fark yok).
- Modelin GSM8K örneklerini **birebir üretme olasılığı** ile GSM8K-GSM1K arasındaki skor farkı arasında **Spearman r² ≈ 0,32** düzeyinde pozitif bir korelasyon bulunmuştur — yani bir modelin eğitim verisinde GSM8K'yı "ezberlediğine" dair kanıt ne kadar güçlüyse, iki test arasındaki fark da o kadar büyümektedir.
- Önemli bir nüans: Bir modelin aşırı uyumlu (overfit) çıkması, o modelin akıl yürütmede kötü olduğu anlamına gelmez — yalnızca GSM8K skorunun göstergesinden **daha zayıf** olabileceği anlamına gelir. Çoğu "aşırı uyumlu" model yine de yeni sorularda makul ölçüde akıl yürütebilmektedir.

## 2.3. Model çıktısı (temsilî karşılaştırma)

| Model ailesi (temsilî) | GSM8K | GSM1K | Fark |
|---|---|---|---|
| Küçük/orta ölçekli, agresif ince ayarlı model | %95 | %65-82 | Büyük düşüş → ezber şüphesi |
| Frontier model (GPT/Claude sınıfı) | %95 | %92-94 | Küçük düşüş → gerçek beceriye daha yakın |

Bu tablo GSM1K makalesinin bulgu **eğilimini** yansıtan temsilî bir özetlemedir; kesin yüzdeler modelden modele ve sürümden sürüme değişir, makalenin kendisine bakılmalıdır.

### Yorum kuralı

Bir model GSM8K'da %95, GSM1K'da %65 alıyorsa, eski testte ezber veya aşırı uyum ihtimali ciddi biçimde değerlendirilmelidir. Tersine, iki skor birbirine yakınsa, bu modelin genellenebilir aritmetik akıl yürütmeye daha çok sahip olduğuna dair (kesin olmayan ama) destekleyici bir sinyaldir.

---

# 3. MATH, MATH-500 ve doygunluk

## 3.1. MATH veri kümesi

MATH (Hendrycks ve ark., 2021), yaklaşık **12.500** yarışma düzeyi matematik problemi ve adım adım çözümler içerir. Cebir, geometri, sayılar teorisi, olasılık, kombinatorik ve kalkülüs gibi alanları kapsar; zorluk 1 (kolay) ile 5 (çok zor, AIME/olimpiyat sınırında) arasında derecelendirilmiştir.

### Gerçek MATH örneği

> \(2x^2-12x+3\) ifadesini en küçük yapan \(x\) değeri nedir?

Parabolün tepe noktası formülüyle:

\[
x=-b/(2a)=12/4=3
\]

**Beklenen cevap:**

```text
\boxed{3}
```

### Model çıktıları

```text
\boxed{3}
```

Puan: **1**.

```text
\boxed{-3}
```

Puan: **0**.

```text
\boxed{\frac{6}{2}}
```

İyi bir sembolik cevap denetleyicisi (ör. SymPy tabanlı bir eşdeğerlik kontrolcüsü) bunu 3 ile eşdeğer sayabilir. Katı metin karşılaştırması ise yanlışlıkla **0** verebilir — bu, MATH benzeri açık-uçlu sayısal/simgesel testlerin en sık karşılaşılan puanlama tuzağıdır.

## 3.2. MATH-500

Tam MATH testinden OpenAI tarafından seçilmiş **500** soruluk, daha hızlı ve yaygın kullanılan alt kümedir. Küçük olması nedeniyle birkaç soruluk fark skoru hızlı değiştirir:

\[
425/500=\%85
\]

Bir soru toplam skorda **0,2 puana** karşılık gelir — bu, MATH-500'ün büyük modeller arasında ince farkları ayırt etmekte neden yetersiz kaldığını açıklar.

## 3.3. Doygunluk (saturation) sorunu ve HARP'ın doğuşu

2024 sonuna gelindiğinde MATH artık frontier modeller için neredeyse doygun hale gelmişti: **o1-mini %90,0**, **Gemini 1.5 Pro %86,5** gibi skorlar rapor edildi. Bu, testin artık modeller arasında ayrım gücü kalmadığı anlamına gelir — bu doygunluk endişesi, aşağıda anlatılan **HARP** gibi daha zor testlerin ortaya çıkmasının doğrudan nedenlerinden biridir.

---

# 4. AIME

AIME (American Invitational Mathematics Examination), ABD Matematik Olimpiyatı seçme sürecinin ikinci basamağıdır ve AMC'de yüksek puan alan öğrencilere açıktır. Her yıl **Şubat** ayında iki oturum (AIME I ve AIME II) halinde yapılır, her oturumda **15 soru** vardır. Cevap her zaman `000`–`999` arasında bir **tam sayıdır** — çoktan seçmeli değildir, bu da şans başarısını neredeyse sıfıra indirir.

## 4.1. Puanlama örneği

Beklenen:

```text
042
```

Model:

```text
42
```

Normalizasyon `042` ile `42`'yi eşdeğer sayabilir. Fakat model `41` derse doğrudan yanlıştır — kısmi puan yoktur.

30 soruda 18 doğru:

\[
18/30=\%60
\]

## 4.2. Yayımlanmış sonuçlar: AIME 2025 (Temmuz 2026 itibarıyla liderlik tablosu)

Kaggle'ın AIME 2025 liderlik tablosunda (8 Temmuz 2026 güncellemesi) rapor edilen skorlar:[^aime2025]

| Model | AIME 2025 skoru |
|---|---|
| GPT-5.5 | %100,0 |
| Claude Opus 4.8 | %100,0 |
| Gemini 3.5 Flash | %96,7 |
| Gemini 3 Pro Preview | %95,8 |
| Grok 4 | %93,3 |
| GPT-5 | %90,8 |
| o3 | %87,5 |
| Claude Sonnet 5 | %86,7 |
| DeepSeek-R1 | %84,2 |

**Önemli uyarı:** Bu modellerin çoğu, AIME 2025 sınavının yapıldığı Şubat 2025'ten **sonra** eğitilmiş veya güncellenmiştir. Bu da AIME 2025 sorularının ve resmî çözümlerinin bu modellerin ön-eğitim/ince ayar verisinde bulunma ihtimalinin yüksek olduğu anlamına gelir. Dolayısıyla yukarıdaki tablo, ham yetenek kıyaslamasından çok, **"bu modeller AIME 2025'i muhtemelen görmüş olabilir"** uyarısıyla birlikte okunmalıdır — tam da MathArena'nın (bkz. Bölüm 9) çözmeye çalıştığı sorun budur.

Ayrıca kaynaklar arasında küçük tutarsızlıklar var: llm-stats.com gibi bazı toplayıcılar Gemini 3 Pro'yu %100 (1.000) olarak, GPT-5'i AIME 2026'da %100 olarak raporluyor. Üçüncü taraf liderlik tabloları arasındaki bu farklar, protokol (kaç deneme, hangi istem şablonu, pass@1 mi consensus@k mi) farklılıklarından kaynaklanır.

## 4.3. AIME'nin sınırı

AIME yalnızca 30 (yılda iki oturum × 15) soru içerdiğinden, tek bir sınavın sonucu istatistiksel olarak gürültülüdür. Bir modelin bir AIME'de %90, bir sonrakinde %73 alması normaldir. Bu nedenle ciddi değerlendirmeler birden fazla yıl/oturum ortalaması veya güven aralığı raporlar (bkz. MathArena, Bölüm 9).

---

# 5. HARP

**HARP** (Human Annotated Reasoning Problems for math), Aralık 2024'te yayımlanmış (arXiv:2412.08819), ABD ulusal matematik yarışmalarından — **A(J)HSME, AMC, AIME, USA(J)MO** — 1950-2024 arası AoPS Wiki'den derlenmiş **5.409** problemden oluşur.

## 5.1. Neden gerekliydi?

MATH gibi önceden en zor kabul edilen testler doygunlaşınca (o1-mini %90,0, Gemini 1.5 Pro %86,5), HARP hem daha geniş bir zorluk yelpazesi hem de altı ayrı zorluk seviyesine bölünmüş problemler sunarak ayrım gücünü geri getirmeyi hedefledi.

- 5.409 problemin **4.780**'inin otomatik olarak doğrulanabilir (checkable) cevabı vardır.
- Problemler **altı zorluk düzeyine** ayrılmıştır.
- En zor dilimde (**197 problem**) frontier modeller bile zorlanır: **o1-mini ortalama %41,1**, **Gemini 1.5 Pro yalnızca %9,6** doğruluk göstermiştir.[^harp]

## 5.2. Model çıktısı (temsilî)

```text
Görev: 2019 AIME I, Problem 15 zorluğunda bir kombinatorik soru.
Model cevabı: 217
Beklenen cevap: 384
```

Puan: **0**. HARP'ın en zor katmanında bu tür yanlış cevaplar frontier modellerde bile sık görülür; bu da testin ayrım gücünün MATH veya AIME'den daha yüksek olduğunu gösterir.

---

# 6. Putnam-AXIOM

**Putnam-AXIOM** (arXiv:2508.08292, ICML 2025), William Lowell Putnam Matematik Yarışması'ndan **522** üniversite düzeyi yarışma problemi içerir. Putnam, ABD ve Kanada'daki üniversite öğrencilerine yönelik, dünyanın en zor matematik yarışmalarından biridir (medyan insan puanı genellikle 12 üzerinden 0-2'dir).

## 6.1. Kirlenmeye dirençli tasarım: "Fonksiyonel varyasyonlar"

Putnam-AXIOM'un asıl yeniliği, orijinal 522 soruya ek olarak **100 "fonksiyonel varyant"** üretmesidir: değişkenler ve sabitler programatik olarak değiştirilerek aynı zorlukta, fakat metin olarak farklı sonsuz bir soru akışı üretilebilir. Bu, statik bir veri kümesinin aksine **kirlenmeye dayanıklı bir test yatağı** sağlar.

## 6.2. Yayımlanmış sonuç

Makalenin kendi raporuna göre:

- Orijinal (Original) sette en güçlü değerlendirilen model olan **OpenAI o1-preview %41,9** almıştır.
- Aynı modelin **Varyasyonlar (Variations)** setindeki doğruluğu **%19,6 mutlak** (yaklaşık **%46,8 göreli**) düşmüştür.
- Değerlendirilen diğer on sekiz model de aynı azalan eğilimi göstermiştir.

Bu sonuç, orijinal Putnam sorularının bir kısmının modellerin eğitim verisinde ezberlenmiş olabileceğine dair güçlü bir işarettir.

## 6.3. Teacher-Forced Accuracy (TFA)

Putnam-AXIOM, yalnızca "kutulu" (`\boxed{}`) nihai cevaba bakmak yerine, akıl yürütme izinin kendisini doğrudan puanlayan hafif bir metrik olan **Teacher-Forced Accuracy**'yi de tanıtır ve doğal dil ispat değerlendirmesini kısmen otomatikleştirir.

### Model çıktısı örneği

```text
Soru (orijinal): f(n) fonksiyonu için ... n=7 olduğunda değeri nedir?
Beklenen cevap: 34

Soru (fonksiyonel varyant, aynı yapı farklı sabitler):
f(n) fonksiyonu için ... n=11 olduğunda değeri nedir?
Beklenen cevap: 58 (varyant için yeniden hesaplanmış)
```

Model orijinali doğru çözüp varyantı yanlış çözerse, bu ezber sinyali olarak işaretlenir.

---

# 7. UGMathBench ve U-MATH

## 7.1. UGMathBench

**UGMathBench** (arXiv:2501.13766, ICLR 2025), lisans (undergraduate) düzeyi matematik akıl yürütmesi için tasarlanmış, **16 ders**, **111 konu**, **6 zorluk düzeyi** ve **10 farklı cevap türü** kapsayan dinamik bir benchmarktır.

- Toplam **5.062** problem içerir; her soru için **3 farklı sürüm** (farklı rastgele tohumla değişken değerleri değiştirilmiş) yayımlanır — bu da onu hem dinamik hem de kısmen kirlenmeye dirençli kılar.
- **Effective Accuracy (EAcc)** adlı katı bir metrik kullanır: bir sorunun üç sürümünün de doğru çözülmesi gerekir.
- Yayımlanmış sonuç: **OpenAI o1-mini** EAcc bakımından yalnızca **%56,3** almıştır; çoğu açık kaynaklı model, uzmanlaşmış matematik modelleri dahil, **%30'un altında** kalmıştır.

### Model çıktısı örneği

```text
Soru (sürüm 1): a = 4 için f(a) integralinin değeri?
Model: 12  → Doğru

Soru (sürüm 2, aynı soru a = 7 ile): f(a) integralinin değeri?
Model: 19  → Yanlış (beklenen 21)
```

Üç sürümden biri yanlış olduğu için bu soru EAcc hesabında **0** sayılır — model yalnızca ilk versiyonu "tanıdığı" için doğru cevaplamış olabilir.

## 7.2. U-MATH ve μ-MATH

**U-MATH** (arXiv:2412.03205, Toloka AI), **1.100** yayımlanmamış, öğretim materyallerinden derlenmiş üniversite düzeyi açık uçlu problem içerir; altı ana konuya dengeli dağıtılmıştır ve **%20'si görsel** (diyagram/şekil gerektiren) içerik barındırır.

Yayımlanmış sonuç:

- Metin tabanlı görevlerde en iyi model **%93,1** doğruluğa ulaşırken,
- Görsel (multimodal) görevlerde aynı en iyi model yalnızca **%58,5**'e ulaşabilmiştir — bu, çoklu-modal matematik akıl yürütmesinin metin-tabanlıdan belirgin biçimde geride olduğunu gösterir.

**μ-MATH**, U-MATH'ten türetilmiş bir "meta-değerlendirme" veri kümesidir: LLM'lerin serbest biçimli matematik çözümlerini **hakemlik yaparak** (judge olarak) ne kadar iyi değerlendirebildiğini ölçer (271 U-MATH görevinden üretilmiş 1.084 etiketli örnek). En güçlü modeller bile bu hakemlik görevinde **F1 ≈ %90,1** ile tavan yapmaktadır.

---

# 8. OlympiadBench

**OlympiadBench** (arXiv:2402.14008, ACL 2024), olimpiyat düzeyi matematik ve fizik yarışmalarından derlenmiş, iki dilli (İngilizce-Çince) ve çoklu-modal bir benchmarktır.

- Toplam **8.952** problem: **6.524 matematik** (%73), **2.428 fizik** (%27).
- Kaynaklar arasında **IMO, RMM, ARML, EGMO, IPhO, APhO** ve Çin üniversite giriş sınavı (gaokao) benzeri sorular bulunur.
- Her problem, adım adım akıl yürütme için uzman düzeyinde açıklamalarla notlandırılmıştır; sayısal, sembolik ve ispat-tabanlı cevapları ayırt eden katı bir değerlendirme protokolü vardır.

## 8.1. Yayımlanmış sonuç (2024 lansmanı)

Yayımlandığı dönemde en güçlü değerlendirilen model **GPT-4V**, ortalama **%17,23** skor almıştır; fizik alt kümesinde bu **%11,28**'e kadar düşmüştür. Bu düşük skorlar, OlympiadBench'in lansman anında ne kadar zorlayıcı olduğunu gösterir (2026 itibarıyla frontier reasoning modelleri bu sayıların çok üzerindedir, fakat güncel toplu skorlar için MathArena/epoch.ai gibi canlı liderlik tablolarına bakılmalıdır).

---

# 9. MathArena: kirlenmeye karşı "canlı" bir çözüm

**MathArena** (arXiv:2505.23281, NeurIPS 2025 Datasets & Benchmarks Track), statik bir veri kümesi olmak yerine **tekrarlayan matematik yarışmalarını gerçek zamanlı olarak** değerlendiren bir platformdur.

## 9.1. Çalışma prensibi

Yeni bir yarışma (ör. güncel yılın AIME'si) sonuçlandığı **anda**, sorular MathArena'ya eklenir ve modeller hemen değerlendirilir. Bir model, yarışmanın yapıldığı tarihten **sonra** eğitildiyse, liderlik tablosunda açık bir **kirlenme uyarı işareti** ile gösterilir.

## 9.2. Yayımlanmış sonuç: AIME 2024 kirlenmiş çıktı

MathArena'nın kendi analizine göre, popüler **AIME 2024** veri kümesi çoğu önde gelen LLM tarafından **önemli ölçüde kirletilmiştir** — bu da onu, modellerin gerçek yeteneğini ölçmek için **artık güvenilmez** kılmaktadır. Araştırmacılar bu kirlenmeye dair güçlü işaretler bulmuştur (ör. modelin soruyu, resmî çözüm metnine has ifadelerle "hatırlaması").

## 9.3. İstatistiksel titizlik

MathArena, ham doğruluk yüzdesinin ötesine geçerek:

- Standart hata (standard error),
- **%95 güven aralığı**,
- Modeller arası sıralamanın anlamlılığını test eden **eşleştirilmiş permütasyon testleri (paired permutation tests)**

raporlar. Bu, "Model A %72, Model B %70 aldı, A daha iyi" gibi yüzeysel yorumların önüne geçmeyi amaçlar — aradaki 2 puanlık fark istatistiksel olarak anlamsız olabilir.

### Model çıktısı (temsilî karşılaştırma)

| Model | AIME (canlı, yeni oturum) | Güven aralığı (temsilî) |
|---|---|---|
| Model A | %72,0 | [%64, %80] |
| Model B | %70,0 | [%62, %78] |

Aralıklar örtüştüğü için A'nın B'den "gerçekten" daha iyi olduğu istatistiksel olarak iddia edilemez.

---

# 10. FrontierMath

**FrontierMath** (Epoch AI, ilk duyuru 8 Kasım 2024), matematik benchmarklarının en zorlarından biridir; profesyonel matematikçiler tarafından özel olarak hazırlanmış, **hiç yayımlanmamış** orijinal problemlerden oluşur.

## 10.1. Yapı

- Toplam **338 problem**: **295 problemlik temel set** (Tier 1-3) + **43 problemlik genişletme seti** (Tier 4).
- **Tier 1-3:** Lisans düzeyinden başlayıp ileri lisansüstü öğrenci düzeyinde "keşif" (exploratory) problemlere kadar uzanır.
- **Tier 4:** Araştırma düzeyi matematik — konunun uzmanı matematikçilerin bile saatler/günler harcayabileceği problemler.
- Problemler, doğrulaması otomatik yapılabilecek şekilde (ör. büyük tam sayı veya kesin sembolik ifade) tasarlanmıştır; bu da kaçamak/yaklaşık cevaplarla puan almayı engeller.
- **Önemli çıkar çatışması notu:** Epoch AI, FrontierMath'in "About" sayfasında OpenAI'nin projeye finansal destek sağladığını açıkça beyan etmektedir — bu, OpenAI modellerinin sonuçlarını yorumlarken akılda tutulması gereken bir husustur.

## 10.2. Skor tarihçesi — dramatik bir ilerleme

| Tarih | Yayımlanmış sonuç |
|---|---|
| Kasım 2024 (lansman) | Claude 3.5 Sonnet, o1-preview, GPT-4o, Gemini 1.5 Pro dahil altı önde gelen model, problemlerin **%2'sinden azını** çözebildi. |
| 20 Aralık 2024 | OpenAI'nin o3 duyurusunda, FrontierMath'in (11-26-24 sürümü üzerinde) **%25,2** skor aldığı iddia edildi; OpenAI'nin daha sonraki iç değerlendirmelerinde bu rakam **~%30**'a çıkarıldı. |
| Erken 2026 | Claude Opus 4.5, Tier 4'te **%10'un altında** kaldı. |
| 12 Haziran 2026 | Epoch AI, problemlerin **%42'sindeki hataları düzelttiği** **v2 sürümünü** yayımladı — bu, v2 öncesi bütün skorların bir miktar dikkatle yorumlanması gerektiği anlamına gelir. |
| 2026 ortası | Kaynaklar arasında belirgin farklılık var: bazı özetler en iyi reasoning modellerin **Tier 1-3'te %50'nin üzerine, Tier 4'te %25-40 aralığına** ulaştığını; başka bir kaynak ise v2 sonrası "bugünün en iyi skorlarının Tier 4'te %88 civarına" ulaştığını aktarıyor. |

**Yorum notu:** Bu son satırdaki büyük fark (Tier 4 için %25-40 mı, %88 mi), üçüncü taraf özetleyicilerin farklı zaman dilimlerini, farklı pass@1/best-of-N protokollerini veya v1/v2 sürüm karışıklığını yansıtıyor olabilir. Kesin, tarihli, tek bir modele atfedilmiş bir sayı isteniyorsa **epoch.ai/frontiermath** sayfasındaki canlı liderlik tablosuna doğrudan bakılmalıdır; bu belgede söz konusu üst sınır rakamı bir model adına bağlanmadan, mevcut belirsizliğiyle birlikte aktarılmaktadır.

## 10.2. Model çıktısı (temsilî — Tier 4 tarzı bir problem)

```text
Görev (temsilî, gerçek FrontierMath sorusu yayımlanmamıştır):
Belirli bir cebirsel çeşitliliğin (algebraic variety) belirli bir
sonlu cisim üzerindeki nokta sayısını modülo büyük bir asal sayı hesapla.

Model çıktısı:
"... [birkaç sayfalık cebirsel geometri akıl yürütmesi] ...
Sonuç: 384947228"

Beklenen cevap: 217038451
```

Puan: **0**. FrontierMath'te kısmi puan yoktur; cevap ya tam doğrudur ya da yanlıştır, ve yanlış cevaba genellikle uzun, tutarlı görünen ama hatalı bir akıl yürütme eşlik eder — bu da testin insan gözüyle "kontrol edilmesini" özellikle zorlaştırır.

---

# 11. Matematik olimpiyatlarında AI: IMO 2025 vakası

2025 Uluslararası Matematik Olimpiyatı (IMO), AI'nin matematik akıl yürütmesindeki ilerlemesi açısından bir dönüm noktası oldu ve bu konudaki iki büyük laboratuvarın (Google DeepMind ve OpenAI) farklı iddialarını karşılaştırmak, "yayımlanmış sonuç" kavramının ne kadar dikkatli okunması gerektiğini gösteren iyi bir örnektir.

## 11.1. Yayımlanmış sonuç: Google DeepMind

Google DeepMind, "Deep Think" özellikli gelişmiş bir Gemini sürümünün IMO 2025'in 6 probleminden **5'ini çözerek 42 üzerinden 35 puan** aldığını duyurdu. Model, resmî problem metinlerinden doğrudan, ön işleme yapılmadan, doğal dilde biçimsel ispatlar üretti ve yarışmanın **4,5 saatlik** süre sınırı içinde çalıştı. Kritik nokta: bu sonuç, **IMO'nun kendi koordinatörleri tarafından, insan yarışmacılarla aynı kriterlerle resmî olarak notlandırıldı** — Google, sonuçların resmî olarak değerlendirildiği ilk kohorta katıldı.

## 11.2. Yayımlanmış sonuç: OpenAI

OpenAI araştırmacısı Alexander Wei, kendi deneysel akıl yürütme modellerinin aynı IMO 2025 sorularında (1'den 5'e kadar, 6. hariç) **altın madalya düzeyinde** sonuç aldığını duyurdu — yine 35/42 puanla, Gemini ile aynı skor. Fakat OpenAI'nin sonucu **resmî IMO değerlendirmesine girmedi**; model resmî yarışmaya kaydolmadı.

## 11.3. Tartışma: altın mı gümüş mü?

Google DeepMind'ın Superhuman Reasoning ekibi liderlerinden Thang Luong, OpenAI'nin kendi puanlama yönteminde bir puan kesintisi olduğunu belirterek bunun aslında **altın değil gümüş madalya** eşiğine denk geldiğini öne sürdü. Bu tartışma, aynı ham sayının (35/42) bile puanlama protokolüne ve resmî doğrulamaya bağlı olarak farklı yorumlanabileceğini gösteren somut bir örnektir.

### Çıkarım

Bu vaka, bu belgenin başındaki uyarıyı doğrudan doğruluyor: bir "yayımlanmış sonuç" okurken şu sorular sorulmalıdır — *Kim doğruladı? Hangi protokolle? Resmî bir üçüncü taraf mı, yoksa şirketin kendisi mi notlandırdı?*

---

# 12. Puanlayıcı tuzakları: matematik benchmarklarında sembolik eşdeğerlik

Matematik benchmarklarının kod benchmarklarından farklı, kendine has bir puanlama zorluğu vardır: **doğru cevabın birden fazla geçerli yazım biçimi olabilir.** İyi tasarlanmamış bir puanlayıcı, doğru bir cevabı yanlışlıkla sıfır puanla cezalandırabilir (yanlış-negatif) veya bazen yanlış bir cevabı doğru sanabilir (yanlış-pozitif).

## 12.1. Sık görülen eşdeğerlik türleri

| Beklenen cevap | Model çıktısı | Matematiksel olarak eşdeğer mi? | Katı metin eşleşmesi ne der? |
|---|---|---|---|
| `3` | `\frac{6}{2}` | Evet | Yanlış (yanlış-negatif riski) |
| `1/2` | `0.5` | Evet | Yanlış (biçim normalizasyonu gerekir) |
| `2\sqrt{3}` | `\sqrt{12}` | Evet | Yanlış (sembolik sadeleştirme gerekir) |
| `x=2, x=-2` | `x = \pm 2` | Evet | Yanlış (küme gösterimi farklı) |
| `042` (AIME) | `42` | Evet | Bağlama göre değişir (sıfır dolgusu) |
| `\boxed{5}` | `\boxed{-5}` | Hayır | Doğru biçimde yanlış |

## 12.2. Neden önemli?

MATH, MATH-500 ve UGMathBench gibi testlerde kullanılan sembolik denetleyiciler (ör. SymPy tabanlı), yukarıdaki ilk dört satırı genellikle doğru biçimde eşdeğer sayacak şekilde tasarlanmıştır. Fakat daha basit, saf metin karşılaştırmasına dayanan eski değerlendirme betikleri bu tür durumları **yanlışlıkla sıfır** puanlayabilir. Bir modelin gerçek yeteneğini raporlarken hangi puanlayıcının kullanıldığı (katı metin mi, sembolik eşdeğerlik denetleyicisi mi, yoksa bir LLM hakem mi) mutlaka belirtilmelidir — aksi hâlde iki farklı çalışmanın aynı benchmark üzerinde bildirdiği sayılar doğrudan karşılaştırılamaz.

## 12.3. Model çıktısı örneği (yanlış-negatif riski)

```text
Soru: 12 sayısının kare kökünü en sade radikal biçimde yazın.
Beklenen cevap: 2\sqrt{3}
Model çıktısı: \sqrt{12}
```

Sayısal olarak iki ifade de \( 3{,}464\ldots \) değerine eşittir, fakat "en sade radikal biçim" isteyen bir soruda katı bir metin eşleştirici bunu yanlış sayabilir; sembolik bir denetleyici ise sayısal değeri karşılaştırıp doğru kabul edebilir. Hangi yaklaşımın "doğru" olduğu, sorunun kendisinin sadeleştirilmiş biçim isteyip istemediğine bağlıdır — bu da otomatik puanlamanın neden hâlâ tam çözülmüş bir problem olmadığını gösterir.

---

# 13. Karşılaştırma tablosu

| Benchmark | Zorluk düzeyi | Soru sayısı | Cevap biçimi | Kirlenmeye dayanıklılık |
|---|---|---|---|---|
| **GSM8K** | İlkokul/ortaokul, çok adımlı aritmetik | ~8.500 (1.319 test) | Serbest metin → sayısal cevap ayıklama | Düşük (2021'den beri açık, yaygın kirlenme şüphesi) |
| **GSM1K** | GSM8K ile eşleştirilmiş | ~1.000 | Sayısal | Yüksek (hiç yayımlanmadı, özellikle kirlenme testi için tasarlandı) |
| **MATH / MATH-500** | Lise yarışma düzeyi (5 seviye) | 12.500 / 500 | `\boxed{}` sembolik/sayısal | Düşük-orta (2021'den beri açık, doygun) |
| **AIME** | ABD olimpiyat seçmesi | 30/yıl (15×2 oturum) | Tam sayı 000-999 | Düşük (statik geçmiş yıllar); MathArena ile canlı kullanılırsa yüksek |
| **HARP** | AMC/AIME/USA(J)MO karışık, 6 zorluk düzeyi | 5.409 (4.780 kontrol edilebilir) | Karışık (çoğunlukla sayısal) | Orta (2024 sonu itibarıyla yeni, geniş zorluk yelpazesi) |
| **Putnam-AXIOM** | Üniversite yarışma düzeyi (çok zor) | 522 orijinal + 100 varyant | Sayısal/sembolik + TFA ile akıl yürütme izi | Yüksek (fonksiyonel varyantlar sayesinde sonsuz benzer soru üretilebilir) |
| **UGMathBench** | Lisans düzeyi | 5.062 (×3 sürüm) | 10 farklı cevap türü, EAcc ile katı puanlama | Orta-yüksek (her soru 3 varyantlı) |
| **U-MATH** | Üniversite düzeyi, %20 görsel | 1.100 | Açık uçlu, LLM-hakem ile puanlama | Yüksek (yayımlanmamış, öğretim materyalinden) |
| **OlympiadBench** | Olimpiyat matematik + fizik | 8.952 | Sayısal/sembolik/ispat | Düşük-orta (2024'te açık yayımlandı) |
| **MathArena** | Değişken (kullanılan yarışmaya bağlı) | Yarışma başına ~30-100 | Yarışmanın kendi formatı | Çok yüksek (canlı, yarışma biter bitmez değerlendirme) |
| **FrontierMath** | Lisans → araştırma düzeyi (4 kademe) | 338 (295 + 43) | Kesin sayısal/sembolik, otomatik doğrulama | Çok yüksek (hiç yayımlanmadı, kapalı tutuluyor) |

---

# 14. Genel çıkarımlar

1. **Statik benchmarklar zamanla "ölür".** GSM8K, MATH gibi 2021 civarı yayımlanmış testler, frontier modeller tarafından ya doygunlaşmış ya da kirlenme riski taşır hâle gelmiştir. HARP, Putnam-AXIOM, FrontierMath gibi daha yeni testler bunun doğrudan cevabıdır.

2. **Kirlenmeye direnç iki yoldan sağlanır:** (a) soruları hiç yayımlamamak (FrontierMath, U-MATH), (b) sürekli yeni soru üretmek — ya programatik varyasyonla (Putnam-AXIOM, UGMathBench) ya da gerçek zamanlı yeni yarışmalarla (MathArena).

3. **"GSM8K'de %95" tek başına bir şey söylemez.** GSM1K gibi bir kontrol testiyle karşılaştırılmadığı sürece, bu skorun ne kadarının gerçek akıl yürütme, ne kadarının ezber olduğu bilinemez.

4. **Tek bir yarışma sonucu (AIME gibi) istatistiksel olarak gürültülüdür.** MathArena'nın güven aralığı raporlama yaklaşımı, iki modelin puanları arasındaki küçük farkların anlamlı olup olmadığını ayırt etmeye yardımcı olur.

5. **"Yayımlanmış sonuç" ile "resmî olarak doğrulanmış sonuç" farklı şeylerdir.** IMO 2025 vakasında görüldüğü gibi, aynı ham puan (35/42), bağımsız/resmî doğrulama olup olmamasına göre çok farklı yorumlanabilir.

6. **Tier'li/kademeli testler (FrontierMath gibi) tek bir yüzdeye indirgenmemelidir.** "FrontierMath'te %52 aldı" cümlesi, bunun Tier 1-3 mü Tier 4 mü olduğu belirtilmeden eksik kalır — iki kademe arasındaki zorluk farkı çok büyüktür.

---

[^aime2025]: Kaggle AIME 2025 Leaderboard (8 Temmuz 2026 güncellemesi) ve llm-stats.com AIME 2025 liderlik tablosu; ayrıca bkz. [MathArena AIME 2024/2025/2026 analizleri](https://matharena.ai) kirlenme uyarıları için.
[^harp]: HARP, arXiv:2412.08819, "HARP: A challenging human-annotated math reasoning benchmark".
