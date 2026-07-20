# Genel Bilgi ve Muhakeme Benchmarkları

**Bu belge**, `llm_benchmarks_2026/` klasöründeki çok dosyalı bilgi tabanının ikinci parçasıdır. Puanlama yöntemlerinin nasıl işlediği için önce `00_puanlama_yontemleri.md` dosyasına bakınız.

Aşağıdaki örneklerde aynı kural geçerlidir:

- **Beklenen cevap**, veri kümesindeki doğru cevaptır (*gold answer*).
- **Model çıktısı**, puanlama mekanizmasını göstermek için verilmiş somut bir örnek cevaptır.
- **"Yayımlanmış sonuç"**, gerçek bir makalede veya resmî liderlik tablosunda raporlanmış, kaynağı belirtilen bir sayıdır.
- Test sorularının bir kısmı veri sızıntısını önlemek için paylaşılmadığından (GPQA, HLE gibi), o bölümlerde gerçek görev biçimine sadık **temsilî** örnekler kullanılmıştır ve böyle işaretlenmiştir.
- Liderlik tablosu sayıları **canlıdır**; bu belgedeki anlık görüntüler Temmuz 2026 itibarıyladır ve okuduğunuz an değişmiş olabilir. Her sayının yanında erişim tarihi ve kaynağı belirtilmiştir.

---

## Karşılaştırma tablosu

| Benchmark | Soru sayısı | Format | Şans tabanı | Neyi hedefler |
|---|---:|---|---:|---|
| MMLU | ~14.000 | 4 şıklı çoktan seçmeli, 57 konu | %25 | Geniş akademik/mesleki bilgi |
| MMLU-Pro | ~12.000 | 10 şıklı çoktan seçmeli | %10 | Daha zor, muhakeme ağırlıklı bilgi |
| MMLU-Redux | 5.700 (yeniden etiketlenmiş) | 4 şıklı, 57 konu | %25 | MMLU'nun hata payından arındırılmış hâli |
| GPQA (ana küme) | 448 | 4 şıklı çoktan seçmeli | %25 | Lisansüstü düzey, "Google-proof" bilim |
| GPQA Diamond | 198 | 4 şıklı çoktan seçmeli | %25 | GPQA'nın en yüksek kaliteli alt kümesi |
| Humanity's Last Exam (HLE) | ~2.500 | Çoktan seçmeli + kısa cevap + görsel | Değişken (çoğu kapalı uçlu) | Frontier düzey uzmanlık sınırı |
| BBH | 23 görev, ~6.500 soru | Çeşitli (serbest üretim) | Göreve bağlı | Büyük modellerin zorlandığı çok adımlı muhakeme |
| BBEH | 23 görev (BBH'nin zorlaştırılmış hâli) | Çeşitli | Göreve bağlı | BBH doygunlaştıktan sonraki zorluk katmanı |
| ARC-AGI-2 | 1.000 eğitim + 360 değerlendirme | Izgara (grid) dönüşümü | ~%0 (rastgele tahmin anlamsız) | Az örnekten yeni kural keşfi |
| ARC-AGI-3 | Yüzlerce etkileşimli ortam, binlerce seviye | Etkileşimli oyun ortamı | ~%0 | Ajan düzeyinde çevrimiçi öğrenme |
| LiveBench | Aylık güncellenen, 18 görev / 6 kategori | Nesnel, otomatik puanlanan | Göreve bağlı | Veri sızıntısına dirençli güncel yetenek |
| HELM (Capabilities/Lite/Safety) | Senaryoya bağlı (çok sayıda alt test) | Karma | Göreve bağlı | Bütüncül (accuracy + kalibrasyon + adalet + maliyet) profil |
| AGIEval | ~8.000+ | Çoktan seçmeli + kısa cevap, gerçek sınav soruları | Göreve bağlı | İnsan sınavlarına (SAT, hukuk, giriş sınavları) dayalı muhakeme |
| SuperGPQA | 26.529 (GitHub'a göre) | Çoktan seçmeli, 285 disiplin | %10–25 arası (şıklara bağlı) | GPQA'yı 285 lisansüstü disipline genişletme |
| KOR-Bench | 125 kural × 10 örnek = 1.250 | Serbest üretim, kural uygulama | Göreve bağlı | "Bilgiden bağımsız" saf kural-tabanlı muhakeme |

---

## MMLU

**MMLU — Massive Multitask Language Understanding**, 57 akademik ve profesyonel alandan dört seçenekli sorular içerir. Matematik, hukuk, tıp, tarih, bilgisayar bilimi, etik ve işletme gibi geniş bir alanı kapsar.

### Nasıl çalışır?

Modele soru ve dört seçenek gösterilir. Değerlendirme iki biçimde yapılabilir:

1. Modelden `A`, `B`, `C` veya `D` üretmesi istenir.
2. Modelin her seçeneğe verdiği olasılık (log-probability) hesaplanır ve en yüksek olasılıklı seçenek seçilir.

İkinci yöntem özellikle talimatla ince ayarlanmamış (base) modellerde daha yaygındır; talimatla ayarlanmış (instruction-tuned) sohbet modellerinde genellikle ilk yöntem — modele doğrudan harf ürettirme — tercih edilir, çünkü bu modeller genellikle serbest metin üretimine daha uygundur.

### Gerçek MMLU örneği

MMLU'nun `abstract_algebra` bölümündeki gerçek bir soru, veri kümesinde (`cais/mmlu`, Hugging Face) şu şekilde yer alır:

> Find the degree for the given field extension \(\mathbb{Q}(\sqrt2,\sqrt3,\sqrt{18})\) over \(\mathbb{Q}\).

Seçenekler:

- A: 0
- B: 4
- C: 2
- D: 6

\(\sqrt{18}=3\sqrt2\) olduğu için yeni bir bağımsız kök eklemez. \(\sqrt2\) ve \(\sqrt3\) birlikte dereceyi 4 yapar.

**Beklenen cevap:** B, yani 4.

### Model çıktısı ve puan

Model:

```text
C
```

çıktısını verirse:

```text
Beklenen: B
Üretilen: C
Madde puanı: 0
```

100 cebir sorusunun 68'i doğruysa bölüm puanı:

\[
68/100=\%68
\]

### Ne ölçer / ne ölçmez?

**Ölçer:** Ezberlenmiş akademik bilgi, bazı sorularda temel muhakeme, İngilizce soru anlama, seçenekler arasında ayrım yapma.

**Ölçmez:** İyi sohbet etme, uzun metin üretme, gerçek yazılım geliştirme, araç kullanma, güncel internet bilgisi, güvenlik, Türkçe başarısı.

### Güncel durumu ve sınırlaması

MMLU hâlâ karşılaştırma için kullanılır, fakat üst modeller açısından önemli ölçüde doygunlaşmıştır (birçok frontier model %85–92 bandındadır). Ayrıca soruların eğitim verilerine karışmış olma ihtimali yüksektir. "Are We Done with MMLU?" (Gema vd., NAACL 2025, arXiv:2406.04127) çalışması, incelenen MMLU sorularının **%6,49'unda** hata bulunduğunu, en kötü alt kümede (Virology) bu oranın **%57'ye** çıktığını göstermiştir — bu da tek başına yüksek MMLU skorunun artık güçlü bir frontier model göstergesi olmadığını doğrular (ayrıntı bir sonraki bölümde).

---

## MMLU-Pro

MMLU-Pro (Wang vd., NeurIPS 2024, arXiv:2406.01574), klasik MMLU'nun daha zor ve daha güvenilir hâle getirilmiş sürümüdür.

### Başlıca farkları

- Çoğu soruda 4 yerine **10 seçenek** bulunur.
- Gürültülü ve çok kolay sorular temizlenmiştir.
- Bilgi ezberinden çok muhakeme gerektiren sorular eklenmiştir.
- Farklı istem biçimlerinden (prompt sensitivity) MMLU'ya göre daha az etkilenir.
- Chain-of-Thought kullanımı MMLU-Pro'da klasik MMLU'nun aksine belirgin fayda sağlar — bu, sorularının gerçekten çok adımlı muhakeme gerektirdiğinin bir kanıtıdır.

### Yayımlanmış gerçek sonuç

Makalede modellerin MMLU-Pro puanlarının, klasik MMLU'ya göre **16–33 puan** düşebildiği raporlanmıştır. Somut örnek: dönemin en güçlü modeli GPT-4o, klasik MMLU'da **%88,7** alırken MMLU-Pro'da **%72,6**'ya düşmüştür (yaklaşık 16 puanlık kayıp); daha zayıf modellerde (ör. Mixtral-8x7B) bu kayıp 30 puanı aşabilmektedir.

### Gerçek örnek

Yayımlanan doğrulama örneklerinden biri özetle:

> \(2\mathbb{Z}\) halkasının karakteristiği nedir?

Seçenekler arasında `0`, `30`, `3`, `10`, `12`, `50`, `2`, `100`, `20`, `5` bulunur. Doğru cevap `0`dır. Veri kümesindeki etiketi **A**'dır.

### Model çıktısı

```text
A
```

**Madde puanı:** 1. Model `G`, yani 2 derse madde puanı 0 olur.

12.000 sorunun 6.300'ü doğruysa:

\[
6300/12000=\%52{,}5
\]

### Neden MMLU'dan daha değerlidir?

Dört seçenekli MMLU'da şans başarısı %25 iken, on seçenekli soruda yaklaşık %10'dur. Ayrıca yanlış seçenekler birbirine daha yakın olduğundan modelin gerçekten konuya hâkim olması gerekir.

### Sınırlaması

Yine statik bir veri kümesidir; zamanla eğitim verilerine karışabilir. Çoktan seçmeli başarı, modelin aynı bilgiyi açık uçlu biçimde doğru açıklayacağı anlamına gelmez.

---

## MMLU-Redux

MMLU-Redux, "Are We Done with MMLU?" makalesiyle (Gema vd., NAACL 2025, arXiv:2406.04127) birlikte tanıtılan, MMLU'nun **elle yeniden denetlenmiş** bir alt kümesidir.

### Nasıl çalışır?

Araştırmacılar, 57 MMLU konusunun tamamından toplam **5.700 soruluk** bir örneklem alıp uzmanlara yeniden inceletmiştir. Her soru şu kategorilerden birine etiketlenmiştir:

- Doğru (hatasız),
- Yanlış etiketlenmiş cevap (ground truth hatalı),
- Kötü tanımlı soru (birden fazla makul cevap mümkün),
- Seçenek eksikliği (doğru cevap şıklar arasında yok),
- Yanlış/belirsiz soru metni.

### Bulgular

Genel hata oranı **%6,49** olarak tahmin edilmiştir; bu oran konudan konuya çok değişir — Virology alt kümesinde incelenen soruların **%57'si** hatalı bulunmuştur. Bu bulgular, modellerin gerçek raporlanan MMLU skorlarıyla, hatalardan arındırılmış MMLU-Redux skorları arasında **belirgin farklar** olduğunu göstermiştir; bazı modellerin "gerçek" bilgisi, orijinal (hatalı) veri kümesindeki skordan hem daha yüksek hem daha düşük çıkabilmiştir (çünkü bazı "yanlış" sayılan cevaplar aslında doğruydu, bazı "doğru" sayılan cevaplar ise şans eseri hatalı bir altın cevaba denk gelmişti).

İlginç bir meta-bulgu: araştırmacılar MMLU-Redux'un kendi yeniden etiketleme sürecinde bile bir insan uzmanın **doğru bir çözümü yanlışlıkla hatalı olarak yeniden etiketlediğini** tespit etmiştir — bu da "altın standart" oluşturmanın bile hatasız olmadığını gösteren çarpıcı bir örnektir.

### Ne ölçer?

MMLU-Redux, modelin yeteneğini değil, **MMLU'nun kendisinin ne kadar güvenilir olduğunu** ölçmeye yarayan bir "meta-benchmark"tır. Model karşılaştırmalarında MMLU-Redux üzerindeki skor, orijinal MMLU skoruna göre daha az gürültülü kabul edilir.

### Sınırlaması

5.700 sorulu alt küme, orijinal ~14.000 sorunun tamamını kapsamaz; dolayısıyla tüm 57 konuda aynı istatistiksel güvenilirlikte değildir.

---

## GPQA ve GPQA Diamond

GPQA (Rein vd., 2023, arXiv:2311.12022), biyoloji, fizik ve kimya alanlarında uzmanlar tarafından yazılmış **448 lisansüstü düzey soru** içerir. Sorular, uzman olmayan ancak güçlü araştırma becerisine sahip kişilerin internet kullanarak bile kolayca bulamayacağı şekilde ("Google-proof") hazırlanmıştır.

GPQA Diamond, en yüksek kalite ve zorlukta kabul edilen **198 sorudan** oluşan alt kümedir ve günümüzde model kartlarında daha sık raporlanır.

### Nasıl çalışır?

- Soru ve seçenekler modele verilir.
- Genellikle dört seçenekten biri seçilir.
- Puan basit accuracy'dir.

### Test sorularını neden burada göstermiyorum?

GPQA ekibi, test maddelerinin internette çoğaltılmasının veri sızıntısını hızlandıracağını özellikle belirtmektedir. Bu nedenle gerçek test sorusunu kopyalamak benchmarkın güvenilirliğine zarar verir. Görev biçimi şöyledir:

> İleri düzey bir kimya/fizik/biyoloji problemi verilir. Dört cevap birbirine çok yakındır ve yalnızca terim ezberiyle doğru cevap bulunamaz.

### Puanlama örneği (temsilî)

Beklenen seçenek:

```text
D
```

Model:

```text
B
```

Madde puanı: `0`

198 maddelik bir alt kümede 113 doğru:

\[
113/198\approx\%57{,}1
\]

### Yayımlanmış gerçek sonuç

GPQA makalesine (Rein vd., 2023) göre:

- İlgili alandaki **doktora uzmanları** ortalama **%65** doğruluk elde etmiştir (kendi belirttikleri açık hataları hariç tutunca bu oran %74'e çıkar);
- **Uzman olmayan ama güçlü araştırmacılar** (ortalama 30 dakikadan fazla süre harcayıp internete sınırsız erişimle) yalnızca **%34** doğruluk elde etmiştir;
- Veri kümesi Kasım 2023'te yayımlandığında, dönemin en güçlü **GPT-4 tabanlı** temel (baseline) sistemi **%39** doğruluk elde etmiştir.

Bu sonuç, sorunun yalnızca Google'da aramanın yeterli olmadığını göstermiştir — uzman olmayan güçlü araştırmacılar bile şans seviyesinin (%25) biraz üzerinde kalmıştır.

### Ne ölçer / sınırlaması

**Ölçer:** İleri bilimsel bilgi, birden fazla bilgiyi birleştirme, zor seçenekleri eleme, uzun muhakeme.

**Sınırlaması:** GPQA başarısı, modelin yeni bilim ürettiğini göstermez. Model hâlâ eğitiminde karşılaştığı teori ve örüntüleri kullanıyor olabilir. Ayrıca 448 (ya da Diamond için 198) soruluk küçük örneklem büyüklüğü, `00_puanlama_yontemleri.md`'de anlatılan bootstrap güven aralığı mantığıyla ele alınmalıdır — birkaç soruluk fark toplam skoru hızla değiştirebilir.

---

## Humanity's Last Exam (HLE)

Humanity's Last Exam (Center for AI Safety ve Scale AI ortak girişimi), klasik akademik benchmarkların doygunlaşmasına karşı geliştirilmiş, çok zor bir testtir. Yaklaşık **2.500–3.000 arası** (kaynağa göre değişen, çünkü bazı sorular yayımlandıktan sonra veri kalitesi nedeniyle çıkarılmıştır) uzman yazımı, çoğunlukla kapalı uçlu ve bir kısmı multimodal sorudan oluşur. Matematik, doğa bilimleri, beşerî bilimler ve uzmanlık alanlarını kapsar.

### Nasıl çalışır?

Sorular çoktan seçmeli, kısa cevaplı, görsel içeren veya ileri uzmanlık gerektiren biçimlerde olabilir. Modelden yalnızca cevap değil, çoğu değerlendirme düzeninde cevabına ne kadar güvendiği de istenir — bu, kalibrasyon ölçümüne izin verir (bkz. `00_puanlama_yontemleri.md`, §1.11 Brier/ECE).

### Gerçek görev biçimine sadık örnek (temsilî)

Soru:

> Belirli bir cebirsel geometri yapısının verilen koşullar altındaki kohomoloji boyutunu hesaplayın.

Beklenen kısa cevap:

```text
3
```

Model çıktısı:

```text
5
Güven: %90
```

Doğruluk puanı: `0`. Kalibrasyon bakımından bu çok kötü bir hatadır; çünkü model yanlış cevabına çok yüksek güven vermiştir. Brier türü basit bir güven cezası:

\[
(0{,}9-0)^2=0{,}81
\]

Burada düşük değer daha iyidir.

### Yayımlanmış / güncel liderlik tablosu görüntüsü

Scale AI'nin resmî HLE liderlik tablosundan (labs.scale.com, Temmuz 2026 itibarıyla erişilen anlık görüntü) alınan sıralama:

| Sıra | Model | Skor |
|---|---|---:|
| 1 | Gemini 3.1 Pro Preview (yüksek düşünme) | %46,44 ± 1,96 |
| 2 | GPT-5.4 Pro (2026-03-05) | %44,32 ± 1,95 |
| 3 | Muse Spark | %40,56 ± 1,92 |
| 4 | Gemini 3 Pro Preview | %37,52 ± 1,90 |
| 5 | GPT-5.4 (xhigh düşünme) | %36,24 ± 1,88 |
| 6 | Claude Opus 4.7 | %36,20 ± 1,88 |
| 7 | Claude Opus 4.6 (thinking max) | %34,44 ± 1,86 |
| 8 | GPT-5 Pro (2025-10-06) | %31,64 ± 1,82 |

**Önemli not:** Bu, canlı bir liderlik tablosunun anlık görüntüsüdür; sıralama haftalar içinde değişebilir ve raporlanan hata payları (±) aslında istatistiksel belirsizliği göstermektedir — iki modelin skorları hata payı içinde örtüşüyorsa (ör. sıra 4–5–6 birbirine çok yakın), aralarında "kesin" bir üstünlük iddia etmek `00_puanlama_yontemleri.md` §1.13'te anlatılan istatistiksel anlamlılık mantığına aykırı olur.

### Ne ölçer / önemli yorum

**Ölçer:** Frontier düzey uzmanlık, çok zor kapalı uçlu muhakeme, bazı sorularda görsel yorumlama, modelin bilmediğini bilip bilmediği.

**Önemli yorum:** HLE'de %40–50 bandı görünüşte düşük gelse de, soruların zorluk düzeyi nedeniyle MMLU'daki %85–90 ile doğrudan karşılaştırılamaz. HLE, tanımı gereği "insanlığın önündeki en zor sorulardan" oluşacak biçimde tasarlanmıştır; skorların zamanla yükselmesi, testin "aşınması" (yani kısmi veri sızıntısı veya modellerin genel muhakeme yeteneğinin gerçekten artması) anlamına gelebilir — ikisini ayırt etmek kolay değildir.

---

## BIG-Bench, BBH ve BBEH

BIG-Bench, çok sayıda farklı görevi tek çatı altında toplamaya çalışan geniş bir benchmark girişimidir (200'den fazla görev).

**BBH — BIG-Bench Hard**, büyük modellerin özellikle zorlandığı **23 görevi** seçer. Bunlar arasında mantıksal sıralama, nesne takibi, tarih anlama, çok adımlı aritmetik, nedensellik, Boole ifadeleri ve kural çıkarımı bulunur.

**BBEH — BIG-Bench Extra Hard**, BBH doygunlaşmaya başlayınca görevleri daha zor benzerleriyle değiştirmiştir; amaç, BBH'nin artık üst modelleri ayırt edemez hâle gelmesine (ceiling effect) karşı yeni bir zorluk katmanı sağlamaktır.

### Gerçek görev türü örneği

Bilgiler:

```text
Ayşe, Bora'nın solundadır.
Bora, Cem'in solundadır.
Deniz, Ayşe'nin sağında fakat Cem'in solundadır.
```

Soru:

```text
En soldaki kişi kimdir?
```

Beklenen: `Ayşe`. Model: `Bora`. Madde puanı: `0`.

### Görevlerin ortalanması ve makro/mikro tartışması

Üç görevde model: Mantıksal çıkarım %80, Nesne takibi %55, Tarih anlama %65 almışsa **makro ortalama**:

\[
(80+55+65)/3=\%66{,}7
\]

Makro ortalamada her görev aynı ağırlıktadır — bu, BBH/BBEH gibi görev-çeşitliliği yüksek testlerde standart yaklaşımdır, çünkü bir görevde 1.000, diğerinde 100 soru olması sonucu "adaletsizce" değiştirmemelidir (bkz. `00_puanlama_yontemleri.md` §1.12 için mikro alternatifin nasıl farklı sonuç verebileceği).

### Ne ölçer / sınırlaması

Özellikle kısa ama çok adımlı muhakemeyi ölçer. Görevlerin bir kısmı yapaydır (sentetik); gerçek hayattaki karmaşık problem çözmeyi tam olarak temsil etmez.

---

## ARC-AGI-2 ve ARC-AGI-3

ARC-AGI (Chollet, "On the Measure of Intelligence" fikrinden doğan seri), dil bilgisi veya ansiklopedik bilgi yerine **yeni bir kuralı birkaç örnekten keşfetme** becerisini ölçmeye çalışır. ARC Prize Vakfı bu benchmarkı yönetir ve her yıl büyük nakit ödüllü bir yarışma düzenler.

### ARC-AGI-2

ARC-AGI-2 (Chollet vd., 2025, arXiv:2505.11831), resmî ARC Prize sitesine göre **1.000 eğitim görevi** ve **360 değerlendirme görevinden** (120'şer görevlik genel/yarı-özel/özel setlere bölünmüş) oluşur. Modele renkli hücrelerden oluşan birkaç giriş-çıkış ızgarası gösterilir; model dönüşüm kuralını anlamalı ve yeni test ızgarasına uygulamalıdır.

Görevler; sembollere bağlamsal anlam verme, birden fazla kuralı birleştirme, yüzeysel benzerlik yerine temel yapıyı bulma gibi becerileri sınar. ARC Prize'ın resmî sitesine göre, **400'den fazla insan katılımcıyla** kontrollü testler yapılarak zorluk kalibre edilmiştir ve kabul edilen her görev **en az iki insan tarafından, en fazla iki denemede (pass@2)** çözülebilecek şekilde seçilmiştir — yani insan temel çizgisi (baseline) tanım gereği %100'dür. Lansman döneminde temel (reasoning içermeyen) LLM'ler %0, akıl yürütme (reasoning) sistemleri ise **%4'ün altında** skor almıştır.

### Basitleştirilmiş ARC örneği (temsilî görev biçimi)

Eğitim girdisi:

```text
0 0 0
0 2 0
0 0 0
```

Eğitim çıktısı:

```text
2 2 2
2 2 2
2 2 2
```

Buradan kuralın, ortadaki rengin bütün ızgaraya yayılması olduğu çıkarılabilir. Test girdisi:

```text
0 0 0
0 4 0
0 0 0
```

Beklenen:

```text
4 4 4
4 4 4
4 4 4
```

Model çıktısı:

```text
4 4 4
4 0 4
4 4 4
```

Yalnızca bir hücre yanlış olsa bile görev tam çözülememiş sayılır: `Görev puanı: 0`. 100 görevin 7'si tamamen doğruysa skor **%7**'dir.

### Ödül yapısı ve maliyet-verimlilik sınırı

ARC Prize 2026'da ARC-AGI-2 için **büyük ödül (grand prize)**, yalnızca "%85 başarı" eşiğine değil, aynı zamanda belirli bir hesaplama verimliliği içinde kalma şartına bağlıdır — pahalı, aşırı hesaplama harcayan bir sistem eşiği geçse bile "verimli" sayılmayabilir. Bu, `00_puanlama_yontemleri.md` §1.9'da anlatılan "maliyet-verimlilik sınırı" kavramının somut bir uygulamasıdır.

### ARC-AGI-3

ARC-AGI-3, **25 Mart 2026'da** François Chollet ve OpenAI CEO'su Sam Altman'ın katıldığı bir etkinlikte (Y Combinator merkezi, San Francisco) resmen tanıtılmıştır. Statik ızgaralardan **etkileşimli, insan tasarımcılar tarafından elle hazırlanmış yüzlerce oyun ortamına ve binlerce oyun-tarzı seviyeye** geçer. Bu ortamlarda ajana **hiçbir talimat, kural veya hedef verilmez** — ajan:

- Ortamı bağımsız biçimde araştırmalı,
- Ortamın nasıl işlediğini keşfetmeli,
- Başarı koşulunu kendi kendine bulmalı,
- Giderek zorlaşan seviyelerde bu bilgiyi uygulamalı,
- Geri bildirime göre stratejisini değiştirmelidir.

### Yayımlanmış gerçek sonuç

ARC Prize'ın resmî lansman duyurusuna göre, ARC-AGI-3 tanıtıldığında **insanlar %100** başarı oranına ulaşırken, test edilen **frontier yapay zekâ sistemlerinin toplu (agregate) başarı oranı yalnızca %0,51**'dir. Duyuru, Anthropic, Google DeepMind, OpenAI ve xAI'nin bireysel model isimlerini veya ayrı skorlarını paylaşmamış, yalnızca bu toplu "frontier AI" rakamını vermiştir.

### Neden önemli?

ARC, modelin daha önce ezberlediği akademik bilgiden ziyade yeni örüntüleri ne kadar verimli öğrendiğini ölçmeye çalışır. ARC-AGI-3'teki %0,51 ile %100 arasındaki uçurum, dil modellerinin "statik bilgi" testlerinde (MMLU gibi) insan seviyesine yaklaşırken, "çevrimiçi keşif ve adaptasyon" gerektiren görevlerde hâlâ çok gerilerde olduğunu gösteren en çarpıcı 2026 bulgularından biridir.

---

## LiveBench

LiveBench, veri sızıntısı sorununa karşı düzenli olarak yeni sorular eklenen, mümkün olduğunca nesnel (otomatik, hakem-LLM'siz) puanlayıcı kullanan canlı bir benchmarktır.

### Kategoriler ve güncelleme sıklığı

Güncel yapısına göre LiveBench, **6 kategoride toplam 18 görevden** oluşur: matematik, kod, muhakeme, dil, talimat izleme ve veri analizi. Liderlik tablosu **aylık** olarak güncellenir; Temmuz 2026 itibarıyla 38 modelin sonuçlarını içermektedir. Yeni yarışma soruları, güncel veri kaynakları ve testin ilk yayımlanma tarihinden sonra oluşturulan görevler düzenli biçimde eklenir.

### Örnek görev (temsilî)

Haziran 2026'da yayımlanan yeni bir programlama sorusu, Temmuz 2026 model değerlendirmesine eklenebilir. 2025'te eğitilmiş bir modelin bu test cevabını eğitim verisinden ezberlemesi mümkün değildir.

Beklenen program çıktısı: `42`. Model programının çıktısı: `41`. Madde puanı: `0`.

### Skor

Her alt görev kendi nesnel metriğiyle değerlendirilir; sonra görev ve kategori skorları ortak bir ölçeğe getirilerek ortalanır.

Örnek: Matematik 70, Kod 55, Muhakeme 65, Dil 80, Talimat 75, Veri analizi 60 olsun:

\[
(70+55+65+80+75+60)/6=67{,}5
\]

### Avantajı / sınırlaması

**Avantajı:** Daha az veri sızıntısı, güncel soru dağılımı, LLM hakemine daha az bağımlılık, zaman içindeki gerçek ilerlemeyi daha iyi gösterme.

**Sınırlaması:** Canlı bir benchmark olduğundan farklı tarihlerde ölçülen iki skor tam olarak aynı test sürümünü temsil etmeyebilir. Tarih ve sürüm mutlaka belirtilmelidir.

---

## HELM

HELM tek bir soru kümesi değildir. Stanford CRFM (Center for Research on Foundation Models) tarafından geliştirilen **bütüncül değerlendirme çerçevesidir**. Bir modeli yalnızca doğrulukla değil, kalibrasyon, sağlamlık, adalet, önyargı, toksisite, verimlilik, maliyet ve şeffaflık boyutlarıyla değerlendirmeyi amaçlar.

### Alt izler (tracks)

- **HELM Capabilities:** Modellerin temel yeteneklerini ölçen, seçilmiş senaryolardan oluşan görece yeni bir liderlik tablosudur; 2025 itibarıyla 22 model, 5 yetenek odaklı senaryo üzerinden karşılaştırılmıştır.
- **HELM Lite:** Daha hafif, daha hızlı çalıştırılabilen, geniş yetenek kapsamına odaklanan bir değerlendirme paketidir (2023'te tanıtıldı).
- **HELM Safety:** Güvenlik odaklı senaryoları içerir; MLCommons AI safety çalışma grubuyla bağlantılı biçimde standartlaştırılmış güvenlik değerlendirmeleri raporlar.
- Ayrıca AIR-Bench gibi özel amaçlı liderlik tabloları da HELM şemsiyesi altında yayımlanır.

### Örnek

İki model aynı soru-cevap testinde:

| Ölçüm | Model A | Model B |
|---|---:|---:|
| Doğruluk | %82 | %79 |
| Toksik çıktı | %8 | %1 |
| Ortalama gecikme | 4,1 sn | 1,3 sn |
| 1.000 istek maliyeti | 20 $ | 4 $ |

Yalnızca accuracy'ye bakılırsa A daha iyidir. Fakat gerçek ürün için B daha uygun olabilir.

### Puanlama felsefesi

HELM'in temel felsefesi, bütün özellikleri tek bir sayıya ezmemektir. Sonuçlar bir **performans profili** olarak raporlanır. Bu nedenle "HELM skoru 75" ifadesi, hangi senaryo (Capabilities mi, Lite mi, Safety mi) ve hangi metrik olduğu belirtilmeden anlamlı değildir.

---

## AGIEval

AGIEval, gerçek insan sınavlarından (SAT, Çin gaokao/üniversite giriş sınavı, hukuk fakültesi giriş sınavı — LSAT, çeşitli profesyonel sertifika sınavları) derlenmiş bir benchmarktır. Amacı, yapay olarak üretilmiş akademik sorular yerine, **insanların gerçekten girdiği** standart sınavlardaki performansı ölçmektir.

### Nasıl çalışır?

Çoğu soru çoktan seçmelidir, bazıları kısa cevap gerektirir. Sorular İngilizce ve Çince karışıktır — bu, AGIEval'i tek dilli MMLU'dan ayıran önemli bir özelliktir.

### C-Eval ile karşılaştırma

AGIEval'e yakın, sık karıştırılan bir başka benchmark **C-Eval**'dir (2023). C-Eval, Çince bağlamda gelişmiş bilgi ve muhakeme yeteneklerini değerlendirmek üzere tasarlanmış ilk kapsamlı Çince değerlendirme paketidir; ortaokul, lise, üniversite ve profesyonel düzey olmak üzere dört zorluk seviyesinde, **52 farklı disiplinde** çoktan seçmeli sorulardan oluşur ve soruların büyük kısmı deneme sınavı PDF/Word belgelerinden derlenmiştir. AGIEval ise SAT, gaokao, hukuk sınavı gibi **gerçek ulusal sınavlardan** birebir alınmış sorulara odaklanır ve hem Çince hem İngilizce sınavları kapsar. Yani C-Eval "Çince bağlamda geniş disiplin kapsamı", AGIEval ise "gerçek, resmî insan sınavlarına sadakat" önceliğiyle ayrışır.

### Ne ölçer / sınırlaması

**Ölçer:** Modelin, insanların girdiği gerçek yüksek riskli (high-stakes) sınavlarda ne kadar başarılı olacağı — bu, "bir model üniversiteye girebilir mi" veya "bir model barolar sınavını geçebilir mi" gibi doğrudan yorumlanabilir sonuçlar üretir.

**Sınırlaması:** Sınav soruları belirli bir müfredata ve kültürel bağlama (özellikle Çin eğitim sistemine) bağlıdır; genel muhakeme yeteneğinin evrensel bir ölçüsü olarak görülmemelidir. Ayrıca resmî sınav soruları zamanla halka açık kaynaklarda (geçmiş sınav arşivleri) yaygın olduğundan veri sızıntısı riski MMLU'dakine benzer biçimde yüksektir.

---

## SuperGPQA

SuperGPQA (2025, arXiv:2502.14739, NeurIPS 2025 Datasets and Benchmarks Track), GPQA'nın kapsamını devasa biçimde genişleten bir benchmarktır.

### Nasıl çalışır?

**285 lisansüstü disiplini** kapsayan, mühendislik, tıp, bilim ve hukuk gibi **13 geniş alanın** yanı sıra hafif sanayi, tarım ve hizmet odaklı disiplinler gibi geleneksel benchmarklarda neredeyse hiç yer almayan alanları da içeren **25.957 çoktan seçmeli sorudan** oluşur. Soru havuzu, "insan-LLM iş birlikli filtreleme" adı verilen bir yöntemle (LLM cevapları ve uzman geri bildirimiyle yinelemeli arıtma) sıradan/belirsiz soruların ayıklanmasıyla, **80'den fazla uzman gözden geçiren** katkısıyla oluşturulmuştur.

### Ne ölçer?

GPQA'nın "az sayıda, çok yüksek kaliteli soru" felsefesini, "çok sayıda disiplinde geniş kapsam" felsefesiyle birleştirir. Matematiksel hesaplama ve biçimsel muhakemeye özel önem verilir; zorluk sıkı biçimde lisansüstü düzeyde tutulur.

### Yayımlanmış gerçek sonuç

Makaleye göre, en gelişmiş güncel akıl yürütme modelleri bile SuperGPQA'da **%60–65 doğruluk eşiğini belirgin biçimde aşamamaktadır** — bu, GPQA Diamond'daki skorların (bazı frontier modellerde %80'in üzerine çıkabildiği) SuperGPQA'nın çok daha geniş disiplin yelpazesinde tekrarlanamadığını gösterir.

### Sınırlaması

285 disiplin arasında soru sayısı dengesiz dağılmış olabilir; bu da `00_puanlama_yontemleri.md` §1.12'de anlatılan makro/mikro ortalama tuzağının SuperGPQA'da özellikle dikkat gerektirdiği anlamına gelir.

---

## KOR-Bench

KOR-Bench — Knowledge-Orthogonal Reasoning Benchmark (Ma vd., ICLR 2025, arXiv:2410.06526) — "bilgiden bağımsız muhakeme" fikrini test eden farklı bir yaklaşım sunar.

### Nasıl çalışır?

Beş görev kategorisi vardır: **İşlem (Operation), Mantık (Logic), Şifre (Cipher), Bulmaca (Puzzle), Karşıolgusal (Counterfactual)**. Her kategoride, **önceden eğitim verisinde görülmemiş olacak biçimde özel olarak tasarlanmış 25 kural** bulunur ve her kural için 10 problem örneği hazırlanmıştır (toplam 5×25×10 = 1.250 problem). Modele önce yeni bir kural açıklanır (ör. uydurma bir şifreleme yöntemi veya alışılmadık bir aritmetik işlem tanımı), sonra bu kuralı yeni örneklere uygulaması istenir.

### Örnek (temsilî görev biçimi)

Kural tanımı:

```text
"flip-toplama" işlemi: iki sayıyı topla, sonra basamaklarını ters çevir.
```

Soru:

```text
flip-toplama(23, 45) = ?
```

Hesap: \(23+45=68\), ters çevrilince `86`.

Beklenen: `86`. Model `68` derse (ters çevirmeyi unutursa) madde puanı `0` olur.

### Yayımlanmış gerçek sonuç

Makaleye göre OpenAI'nin o1-preview ve o1-mini modelleri sırasıyla **%72,88** ve **%70,16** doğruluk elde ederek, Claude 3.5 Sonnet (**%58,96**) ve GPT-4o'yu (**%58,00**) belirgin farkla geride bırakmıştır — bu sonuç, uzun düşünme zinciri (chain-of-thought) kullanan modellerin, kuralı ilk kez gören ve saf mantıkla uygulaması gereken görevlerde daha güçlü olduğunu göstermektedir.

### Ne ölçer / sınırlaması

**Ölçer:** Modelin, ansiklopedik bilgiye değil, **anlık verilen yeni bir kurala** ne kadar sadık ve tutarlı biçimde uyabildiğini — bu, ARC-AGI'nin ızgara tabanlı yaklaşımına dilsel/sembolik bir alternatif sunar.

**Sınırlaması:** Kurallar yapay olarak tasarlandığından, gerçek dünyadaki açık uçlu problem çözmeyi birebir yansıtmayabilir; ayrıca 25 kural × 5 kategori görece küçük bir örneklemdir, tek bir kuralda yaşanan başarısızlık kategori skorunu orantısız etkileyebilir.

---

## Kaynaklar

- Rein vd., "GPQA: A Graduate-Level Google-Proof Q&A Benchmark", 2023 — https://arxiv.org/abs/2311.12022
- Wang vd., "MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark", NeurIPS 2024 — https://arxiv.org/abs/2406.01574
- Gema vd., "Are We Done with MMLU?" (MMLU-Redux), NAACL 2025 — https://arxiv.org/abs/2406.04127
- Hugging Face, `cais/mmlu` veri kümesi (abstract_algebra örneği doğrulaması) — https://huggingface.co/datasets/cais/mmlu
- Humanity's Last Exam, Wikipedia özeti ve kaynakça — https://en.wikipedia.org/wiki/Humanity's_Last_Exam
- Scale AI, Humanity's Last Exam Leaderboard (canlı, Temmuz 2026 anlık görüntüsü) — https://labs.scale.com/leaderboard/humanitys_last_exam
- Chollet vd., "ARC-AGI-2: A New Challenge for Frontier AI Reasoning Systems", 2025 — https://arxiv.org/abs/2505.11831
- ARC Prize, ARC-AGI-2 resmî sayfası — https://arcprize.org/arc-agi/2
- ARC Prize, ARC-AGI-3 resmî sayfası — https://arcprize.org/arc-agi/3
- ARC Prize, "Announcing ARC-AGI-3" lansman duyurusu (25 Mart 2026) — https://arcprize.org/blog/arc-agi-3-launch
- ARC Prize, Leaderboard — https://arcprize.org/leaderboard
- LiveBench resmî makale/PDF — https://livebench.ai/livebench.pdf
- LiveBench liderlik tablosu (llm-stats.com üzerinden, Temmuz 2026) — https://llm-stats.com/benchmarks/livebench
- Stanford CRFM, HELM Capabilities — https://crfm.stanford.edu/2025/03/20/helm-capabilities.html
- Stanford CRFM, HELM Lite — https://crfm.stanford.edu/2023/12/19/helm-lite.html
- Stanford CRFM, HELM Safety — https://crfm.stanford.edu/2024/11/08/helm-safety.html
- Zhong vd., "AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models" — https://arxiv.org/abs/2304.06364
- Huang vd., "C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models", NeurIPS 2023 — https://arxiv.org/abs/2305.08322
- Du vd., "SuperGPQA: Scaling LLM Evaluation across 285 Graduate Disciplines", 2025 — https://arxiv.org/abs/2502.14739
- SuperGPQA GitHub deposu — https://github.com/SuperGPQA/SuperGPQA
- Ma vd., "KOR-Bench: Benchmarking Language Models on Knowledge-Orthogonal Reasoning Tasks", ICLR 2025 — https://arxiv.org/abs/2410.06526
- KOR-Bench proje sayfası — https://kor-bench.github.io/
