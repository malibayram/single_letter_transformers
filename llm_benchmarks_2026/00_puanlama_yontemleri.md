# Puanlama Yöntemleri: LLM Benchmarklarının Motoru

**Bu belge**, `llm_benchmarks_2026/` klasöründeki çok dosyalı bilgi tabanının ilk parçasıdır ve bir sayının ("Model X, Y benchmarkında %73 aldı") arkasında gerçekte ne olduğunu derinlemesine açıklar.

Aşağıdaki örneklerde köklü bir kural izlenir:

- **Beklenen cevap**, veri kümesindeki doğru cevaptır (*gold answer*).
- **Model çıktısı**, puanlama mekanizmasını göstermek için verilmiş somut bir örnek cevaptır.
- **"Yayımlanmış sonuç"** denildiğinde, gerçek bir makalede veya model kartında raporlanmış bir sayıdan söz edilir ve kaynağı belirtilir.
- Kaynağı doğrulanamayan sayılar **"temsilî örnek"** olarak işaretlenir; bunlar gerçek bir yayından alınmamıştır, yalnızca yöntemi göstermek içindir.

---

## 0. Bir benchmark neden tek bir sayı değildir?

Bir benchmark dört parçalı bir sistemdir:

1. **Veri:** Sorular, belgeler, resimler, kod depoları veya görev ortamları.
2. **Protokol:** Modele ne kadar bağlam verildiği, araç kullanıp kullanamayacağı, kaç örnek (few-shot) gösterildiği, sıcaklık (temperature) ve örnekleme ayarları.
3. **Puanlayıcı:** Cevabın nasıl doğru veya yanlış sayıldığı — string eşleştirme, birim test, başka bir LLM, insan değerlendirici.
4. **Toplulaştırma:** Binlerce tekil puanın tek bir genel skora nasıl çevrildiği — ortalama, ağırlıklı ortalama, medyan, en iyi-of-k.

Aynı model, aynı veri kümesinde farklı istem şablonu, farklı cevap ayıklayıcı (answer extractor), farklı sıcaklık veya farklı sayıda deneme (trial) kullanılınca gözle görülür biçimde farklı puanlar alabilir. "Are We Done with MMLU?" (Gema vd., 2024) çalışması bunun ne kadar ciddi olabileceğini göstermiştir: incelenen MMLU sorularının yaklaşık **%6,49'unda** hata bulunmuş, Virology alt kümesinde bu oran **%57'ye** kadar çıkmıştır — yani "model MMLU'da %80 aldı" cümlesi tek başına eksik, hatta bazen yanıltıcıdır. Bu yüzden ciddi laboratuvarlar artık model kartlarında sadece nihai sayıyı değil, örnekleme sayısını, sıcaklığı ve bazen güven aralığını da yayımlıyor (bkz. §1.15).

---

## 1.1. Accuracy — doğruluk

En basit yöntemdir:

\[
\text{Accuracy}=\frac{\text{Doğru cevap sayısı}}{\text{Toplam soru sayısı}}
\]

### Örnek

Model 100 sorunun 73'ünü doğru, 27'sini yanlış cevapladıysa:

\[
73/100=0{,}73=\%73
\]

Çoktan seçmeli MMLU, GPQA, MedQA ve benzeri benchmarkların temel metriği budur.

### Şans başarısı (random-chance baseline) tablosu

| Seçenek sayısı | Rastgele başarı |
|---|---:|
| 2 (doğru/yanlış) | %50 |
| 4 (klasik MMLU) | %25 |
| 5 | %20 |
| 10 (MMLU-Pro) | %10 |

Aynı **%30** skoru dört seçenekli bir testte rastgeleye yakınken, on seçenekli bir testte anlamlı bir sinyal olabilir. Bu yüzden ham accuracy'yi karşılaştırırken şans tabanını mutlaka hesaba katmak gerekir.

### Sık görülen tuzaklar

1. **Reddetme (refusal) belirsizliği:** Model "bilmiyorum" derse bu genellikle yanlış sayılır, ama bazı protokollerde ayrı bir "attempted/not attempted" sınıfı açılır (bkz. SimpleQA yaklaşımı). Hangi kuralın kullanıldığı raporda belirtilmezse iki modelin accuracy'si karşılaştırılamaz olur.
2. **Sınıf dengesizliği:** Bir alt konudan (subject) çok az soru varsa, o konudaki tek bir hata toplam skoru orantısız etkiler. Bu, §1.12'de ele alınan makro/mikro ortalama tartışmasının köküdür.
3. **Cevap ayıklama (answer extraction) hataları:** Model doğru düşünse bile son cevabı istenen biçimde yazmazsa (`"Cevap: C"` yerine `"C şıkkı doğrudur"`), zayıf bir regex bunu yanlış sayabilir — bu, gerçek yetenek değil, **protokol hatası** kaynaklı bir puan kaybıdır.

---

## 1.2. Exact Match — birebir eşleşme

Modelin cevabı, beklenen cevapla aynı olmalıdır.

Beklenen:

```text
72
```

Model:

```text
72
```

Puan: **1**

Model:

```text
Cevap yaklaşık 72 adettir.
```

Katı bir Exact Match sistemi yalnızca `72` bekliyorsa puan **0** olabilir.

Bu nedenle modern değerlendiriciler genellikle:

- Büyük-küçük harfi,
- Noktalama işaretlerini,
- Başlangıç ve sondaki boşlukları,
- `72.0` ile `72` gibi eşdeğer sayı biçimlerini

normalize eder. SQuAD gibi kaynak-cevap (extractive QA) testlerinde her soru için birden fazla (genelde 3) referans cevap toplanır ve modelin cevabı bunlardan **en iyi eşleşenle** karşılaştırılır — tek bir "doğru" cevap dayatmak, geçerli eşanlamlı ifadeleri haksız yere yanlış sayabilir.

### Matematik cevaplarında özel tuzak

`\boxed{3}` ile `\boxed{6/2}` matematiksel olarak eşdeğerdir ama saf metin karşılaştırması bunları farklı sayar. Bu yüzden MATH, AIME gibi testlerde genellikle bir **sembolik eşdeğerlik denetleyicisi** (ör. SymPy tabanlı) kullanılır; salt string Exact Match kullanmak, doğru cevap veren modelleri haksız yere cezalandırabilir.

---

## 1.3. Token F1

Özellikle kısa cevaplı soru-cevap testlerinde (SQuAD, TriviaQA türevleri) kullanılır. Model cevabındaki kelimelerle doğru cevaptaki kelimelerin ne kadar örtüştüğüne bakar.

Beklenen cevap:

```text
Ankara Türkiye'nin başkentidir.
```

Model cevabı:

```text
Başkent Ankara'dır.
```

Basitleştirilmiş token kümeleri:

- Beklenen: `Ankara`, `Türkiye`, `başkent`
- Model: `başkent`, `Ankara`

İki token doğru bulunmuştur.

\[
Precision=2/2=1
\]

\[
Recall=2/3=0{,}667
\]

\[
F1=\frac{2 \times 1 \times 0{,}667}{1+0{,}667}\approx0{,}80
\]

Yani F1 yaklaşık **%80** olur.

### Sınırlaması

Token F1, yüzeysel kelime örtüşmesine bakar; **anlamsal** eşdeğerliği anlamaz. "Başkent Ankara'dır" ile "Ankara, Türkiye'nin yönetim merkezidir" cümleleri anlamca aynıdır ama ortak token az olduğu için F1 düşük çıkabilir. Bu sınırlama, açık uçlu değerlendirmede LLM-as-a-Judge yaklaşımının (§1.8) neden geliştirildiğini kısmen açıklar.

---

## 1.4. pass@k

Kod benchmarklarında (HumanEval, MBPP, LiveCodeBench, SWE-bench) kullanılır. Modele aynı soru için birden fazla kod üretme hakkı verilir.

- `pass@1`: İlk üretilen kod geçiyor mu?
- `pass@5`: Üretilen ilk beş koddan en az biri geçiyor mu?
- `pass@10`: İlk on koddan en az biri geçiyor mu?

Formül, OpenAI'nin Codex makalesinde (Chen vd., "Evaluating Large Language Models Trained on Code", 2021, arXiv:2107.03374) tanıtılan **yansız (unbiased) tahmincidir**:

\[
pass@k=1-\frac{\binom{n-c}{k}}{\binom{n}{k}}
\]

Burada:

- \(n\): Üretilen toplam çözüm sayısı,
- \(c\): Testleri geçen çözüm sayısı,
- \(k\): Kaç denemeye bakıldığı.

### Neden doğrudan k deneme çalıştırıp bakmıyoruz?

Sadece \(k\) örnek üretip "en az biri geçti mi" diye bakmak yüksek varyanslıdır — aynı modelden aynı \(k\) ile tekrar örnekleme yapsanız farklı sonuç alabilirsiniz. Bunun yerine **daha büyük bir \(n\)** (ör. \(n=200\)) örnekleyip yukarıdaki kombinatorik formülle \(pass@k\)'yı **analitik olarak** tahmin etmek, aynı bütçeyle çok daha kararlı (düşük varyanslı) bir sonuç verir.

### Örnek

Model 10 kod üretmiş ve bunların 2'si doğru olsun.

\[
pass@1=2/10=\%20
\]

Fakat rastgele seçilen beş çözüm arasında en az bir doğru bulunma ihtimali:

\[
pass@5=1-\frac{\binom{8}{5}}{\binom{10}{5}}
=1-\frac{56}{252}
\approx\%77{,}8
\]

Bu nedenle `pass@1` ile `pass@10` kesinlikle aynı şey değildir.

### Uç durumlar

- \(c=0\) ise (hiç doğru çözüm yoksa) her \(k\) için \(pass@k=0\)'dır.
- \(c=n\) ise (tüm çözümler doğruysa) her \(k\) için \(pass@k=1\)'dir.
- \(k>n\) matematiksel olarak tanımsızdır; bu yüzden raporlarda her zaman \(n \geq k\) olacak biçimde örnekleme yapılır (ör. AIME gibi az sorulu testlerde \(n\) büyük tutulur).

### pass@k ile pass^k karıştırılmamalı

Ajan (agent) değerlendirmelerinde bazen **tüm \(k\) denemenin de** geçmesi istenir (\(pass\hat{}k\), "tutarlılık" ölçer) — bu, "en az biri geçsin" anlamına gelen `pass@k`'dan tamamen farklı, çok daha katı bir metriktir. İkisini karıştırmak yaygın bir raporlama hatasıdır.

### Güncel raporlama pratiği

OpenAI'nin GPT-5 sistem kartında bazı değerlendirmeler için birincil metrik olarak `pass@1`'in çoklu deneme (bazı testlerde 30 deneme) üzerinden ortalaması ve **uzunluğa göre ayarlanmış (length-adjusted) skorlar** kullanılmaktadır; bu, daha uzun cevapların yapay olarak skor şişirmesini engellemeyi amaçlar. Google DeepMind ise Gemini model değerlendirme raporlarında "tek deneme" (single attempt, çoğunluk oylaması yok) ile "çoklu deneme" (majority voting, ör. n=64) skorlarını **ayrı ayrı** tablo hâlinde sunar — ikisini karıştırmamak için.

---

## 1.5. Win rate — kazanma oranı

Açık uçlu cevaplarda tek bir doğru cevap bulunmayabilir. İki modelin cevapları insanlara veya hakem bir LLM'ye gösterilir.

100 karşılaştırmada:

- Model A: 55 kez kazanmış,
- Model B: 35 kez kazanmış,
- 10 karşılaştırma berabere kalmış olsun.

Beraberliklerin yarısını iki modele dağıtırsak:

\[
A=(55+5)/100=\%60
\]

\[
B=(35+5)/100=\%40
\]

AlpacaEval gibi testler doğrudan kazanma oranı raporlayabilir.

### Bilinen bir istismar: "null model" saldırısı

"Cheating Automatic LLM Benchmarks: Null Models Achieve High Win Rates" (2024) çalışması, girdiyle hiç ilgisi olmayan **sabit, uzun ve biçimli bir metnin** bile otomatik hakemler karşısında yapay olarak yüksek win rate alabildiğini göstermiştir. Bu, ham win rate'in tek başına güvenilir olmadığını, uzunluk ve biçim kontrolü olmadan kolayca "oyuna getirilebildiğini" (gamed) kanıtlar — bu yüzden §1.8'de anlatılan length-controlled win rate geliştirilmiştir.

---

## 1.6. Bradley–Terry puanı ve Chatbot Arena

Chatbot Arena'da her model her modelle eşit sayıda karşılaşmaz. Bu nedenle yalnızca ham kazanma oranı yeterli değildir.

Bradley–Terry modeli, karşılaşma sonuçlarından her modele gizli bir "güç puanı" \(\theta\) tahmin eder. İki model \(A\) ve \(B\) karşılaştığında \(A\)'nın kazanma olasılığı:

\[
P(A \succ B)=\frac{e^{\theta_A}}{e^{\theta_A}+e^{\theta_B}}
\]

A modeli B'yi, B modeli C'yi sık sık yeniyorsa, A'nın C'den de güçlü olma ihtimali yükselir — geçişli bir sıralama tahmin edilir.

"Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference" (Chiang vd., 2024, arXiv:2403.04132) makalesine göre platform, 2024 başına kadar **100'den fazla modelde milyonlarca çiftli insan tercihi** toplamıştır ve klasik satranç Elo'sundan ziyade çiftli tercih verilerine dayalı Bradley–Terry tarzı bir sıralama kullanır.

### Style control

Arena ekibi daha sonra bir **"style control"** (biçim kontrolü) sürümü eklemiştir: bu, cevabın uzunluğu, madde işaretleri, kalın yazı gibi yüzeysel biçim unsurlarının modelin gerçek "kazanma gücüne" katkısını istatistiksel olarak ayırmaya çalışır — tıpkı AlpacaEval'daki length-controlled win rate gibi bir mantık.

### Puanın anlamı

Puanın mutlak bir anlamı yoktur; **diğer modeller karşısındaki göreli konumu** gösterir ve genellikle bir **güven aralığıyla** (ör. "sıralama 3–5 arası, %95 güven") birlikte raporlanır — tek bir sabit sayı değil, bir aralık.

---

## 1.7. Retrieval ve embedding metrikleri

### Recall@k

Aranan 10 ilgili belge var ve sistem ilk 5 sonuç içinde bunlardan 4'ünü buluyorsa:

\[
Recall@5=4/10=\%40
\]

### MRR — Mean Reciprocal Rank

İlk doğru sonuç üçüncü sıradaysa:

\[
RR=1/3=0{,}333
\]

Birçok sorgunun değeri ortalanarak MRR elde edilir.

### nDCG@10

Doğru belgelerin yalnızca bulunup bulunmadığına değil, **ne kadar üst sıraya yerleştirildiğine** bakar. Sıralı bir listenin **Discounted Cumulative Gain**'i:

\[
DCG@k=\sum_{i=1}^{k}\frac{rel_i}{\log_2(i+1)}
\]

Burada \(rel_i\), \(i\)'inci sıradaki belgenin ilgililik derecesidir (0, 1, 2, 3...). Bu değer, mümkün olan en iyi sıralamanın DCG'sine (\(IDCG\)) bölünerek normalize edilir:

\[
nDCG@k=\frac{DCG@k}{IDCG@k}
\]

Çok ilgili bir belgeyi birinci sıraya koymak, onuncu sıraya koymaktan **logaritmik olarak** daha değerlidir; bu yüzden nDCG, Recall@k'nın veremediği "sıralama kalitesi" bilgisini taşır.

### Spearman ve Pearson korelasyonu

İnsanlar üç cümle çiftinin benzerliğini sırasıyla `1, 2, 3` olarak sıralamış; embedding modeli `1, 3, 2` sıralamasını üretmiş olsun. Spearman korelasyonu bu iki **sıralamanın** ne kadar uyumlu olduğunu ölçer. Bu küçük örnekte değer **0,5** olur. Tam uyum `1`, ters sıralama `-1`'dir.

Spearman, ham **sıra**ları karşılaştırırken Pearson korelasyonu ham **sayısal değerleri** karşılaştırır. Embedding benzerlik skorlarının insan puanlarıyla "doğrusal" ilişkisi değil, "aynı sırayı verip vermediği" önemliyse (STS görevlerinde olduğu gibi) Spearman tercih edilir. MTEB (Massive Text Embedding Benchmark) gibi geniş embedding değerlendirme paketleri, Recall@k, MRR, nDCG@10 ve Spearman'ı farklı alt görevlerde (retrieval, clustering, STS, reranking) birlikte raporlar.

---

## 1.8. LLM-as-a-Judge

Bir başka LLM'ye şu görev verilir:

> "İki cevabı doğruluk, ilgililik, açıklık ve talimata uyma bakımından karşılaştır."

Hakem A daha iyi / B daha iyi / berabere kararı verebilir veya 1–10 arası puan verebilir.

### Bilinen önyargılar (biases)

"Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (Zheng vd., NeurIPS 2023, arXiv:2306.05685) makalesi bu yaklaşımı sistematik olarak incelemiş ve GPT-4 düzeyindeki güçlü hakemlerin insan kararlarıyla **%80'in üzerinde** uyum gösterdiğini, ama şu önyargıların da var olduğunu bulmuştur:

- **Position bias (sıra önyargısı):** Hakem, iki cevaptan hangisi önce gösterilirse ona meyledebilir. Mitigasyon: her çifti iki sırada da (A-B ve B-A) değerlendirip sonuçları birleştirmek.
- **Verbosity/length bias (uzunluk önyargısı):** Daha uzun cevaplar, içerik kalitesi aynı olsa bile daha yüksek puan alma eğilimindedir.
- **Self-enhancement bias (kendini kayırma):** Bir model, kendi ürettiği veya kendi yazım biçimine benzeyen cevapları kayırabilir.
- **Sınırlı muhakeme yeteneği:** Hakem model, kendisinden daha güçlü bir modelin ince hatasını (özellikle matematik/kod gibi doğrulanabilir alanlarda) fark edemeyebilir.

### AlpacaEval Length-Controlled Win Rate

"Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators" (Dubois vd., COLM 2024, arXiv:2404.04475), uzunluk önyargısını **nedensel çıkarım (causal inference)** çerçevesiyle ayıklamayı önerir: "Bütün modellerin cevapları referans modelle aynı uzunlukta olsaydı skor ne olurdu?" sorusuna regresyon tabanlı bir cevap üretir. Bu düzeltme, AlpacaEval'in Chatbot Arena ile Spearman korelasyonunu **0,94'ten 0,98'e** yükseltmiştir — yani uzunluk kontrolü, otomatik testi gerçek insan tercihine daha da yaklaştırmıştır.

Bir modelin:

```text
Ham win rate: %65
Uzunluk kontrollü win rate: %54
```

alması, avantajının bir kısmının yalnızca daha uzun yazmasından geldiğini gösterebilir.

### Hakemler arası tutarlılık: Cohen's kappa (bkz. §1.14)

Bir laboratuvar, tek bir hakem yerine "hakem paneli" (ör. 3 farklı LLM) kullanıyorsa, bu hakemlerin birbiriyle ne kadar uyuştuğu Cohen's kappa ile ölçülebilir; düşük kappa, hakem sonuçlarının güvenilmez olduğuna işaret eder (ayrıntı için §1.14'e bakınız).

---

## 1.9. Ajan başarı oranı

Bir ajan benchmarkında (SWE-bench, OSWorld, WebArena, τ-bench gibi) model yalnızca metin üretmez; tarayıcıyı, terminali veya API'leri kullanır.

\[
Success\ Rate=\frac{\text{Tamamlanan görev}}{\text{Toplam görev}}
\]

20 görevin 13'ü tamamlandıysa başarı **%65**'tir.

Buna ek olarak şu ölçümler yapılabilir:

- Harcanan token,
- API çağrısı sayısı,
- Süre,
- Para maliyeti,
- Gereksiz işlem sayısı,
- Yanlış veya tehlikeli eylem sayısı.

### Maliyet-verimlilik sınırı (cost-efficiency frontier)

Modern ajan benchmarkları artık salt başarı oranını değil, **başarı/maliyet oranını** da öne çıkarıyor. Örneğin ARC Prize'ın ARC-AGI-2 yarışması, ödülü yalnızca "%85 başarı" şartına değil, aynı zamanda görev başına belirli bir hesaplama bütçesi içinde kalma şartına bağlar: %200 maliyetle %85 alan bir sistem, %10 maliyetle %81 alan bir sistemden "daha iyi" sayılmaz. Bu, ajan değerlendirmesinde salt başarı yüzdesinin yeterli olmadığını gösteren güncel bir eğilimdir.

---

## 1.10. Güvenlik puanları

### Attack Success Rate — ASR

100 zararlı saldırı isteminin 28'inde model yasaklanan çıktıyı ürettiyse:

\[
ASR=\%28
\]

Burada **düşük skor daha iyidir**. ASR'nin nasıl hesaplandığı (anahtar kelime eşleştirme mi, yoksa başka bir LLM hakem mi kullanıldığı) sonucu ciddi biçimde değiştirir; iki farklı ASR metodolojisiyle üretilmiş sayılar doğrudan karşılaştırılamaz.

### Aşırı reddetme (overrefusal)

100 güvenli sorunun 20'sini model gereksiz yere reddediyorsa:

\[
Overrefusal=\%20
\]

Güvenli model yalnızca saldırıları reddetmemeli; normal soruları da cevaplayabilmelidir.

Bu alanda iki tamamlayıcı benchmark öne çıkar:

- **XSTest** (Röttger vd.): 250 elle hazırlanmış, güvenli görünen ama gerçekte tehlikeli olmayan istemden oluşur. Yeni nesil modeller için artık **doygunlaşmıştır** — bazı modeller neredeyse tüm 250 soruyu doğru cevaplayabiliyor.
- **OR-Bench** (2024): ~80.000 aşırı-reddetme istemi, bunların içinde ~1.000 sorudan oluşan ve en güçlü modeller için bile zor olan bir alt küme, artı 600 gerçekten zararlı istemden oluşan bir kontrol grubu içerir. OR-Bench, XSTest'in yapısal olarak eşleştirilmiş (paired) güvenli/zararlı istemlerinden farklı olarak **anlamsal belirsizliğe** odaklanır; aynı model iki testte çok farklı overrefusal oranı verebilir (yüzeysel kelime tetikleyicileri XSTest'te daha fazla reddetmeye yol açabilir).

Güvenli bir model, ASR'yi düşürürken overrefusal'ı da düşük tutmalıdır — bu iki metrik arasında bir **ödünleşim (tradeoff)** vardır ve "Refusal–Compliance Tradeoff" araştırmaları bu dengeyi büyük ölçekte denetlemeye çalışır.

---

## 1.11. Brier Skoru ve Expected Calibration Error (ECE) — kalibrasyon

Accuracy modelin ne kadar **doğru** olduğunu ölçer; kalibrasyon ise modelin **kendi doğruluğu hakkındaki güveninin** ne kadar gerçekçi olduğunu ölçer. Mükemmel kalibre bir model, "%80 eminim" dediği sorularda gerçekten yaklaşık %80 oranında doğru cevap vermelidir.

### Brier Skoru

\[
Brier=\frac{1}{N}\sum_{i=1}^{N}(p_i-o_i)^2
\]

Burada \(p_i\), modelin \(i\)'inci soruya verdiği doğruluk olasılığı (güven), \(o_i\) ise gerçek sonuçtur (doğruysa 1, yanlışsa 0). **Düşük Brier skoru daha iyidir**; 0, mükemmel kalibrasyona karşılık gelir.

### Örnek

Model bir soruya %90 güvenle yanlış cevap verirse:

\[
(0{,}9-0)^2=0{,}81
\]

Aynı model başka bir soruya %60 güvenle doğru cevap verirse:

\[
(0{,}6-1)^2=0{,}16
\]

İki sorunun ortalama Brier'i: \((0{,}81+0{,}16)/2=0{,}485\).

### Expected Calibration Error (ECE)

ECE, güven düzeylerini kutulara (bin) ayırıp her kutudaki **ortalama güven** ile **gerçek doğruluk** arasındaki farkın ağırlıklı ortalamasını alır:

\[
ECE=\sum_{b=1}^{B}\frac{n_b}{N}\left|\text{acc}(b)-\text{conf}(b)\right|
\]

Burada \(B\) kutu sayısı, \(n_b\) o kutudaki örnek sayısı, \(\text{acc}(b)\) o kutunun gerçek doğruluğu, \(\text{conf}(b)\) o kutunun ortalama güvenidir.

### Neden önemli?

Güncel kalibrasyon araştırmaları, modern LLM'lerin genel olarak **iyi kalibre olmadığını** göstermektedir: incelenen modellerde ECE **0,120 ile 0,395** arasında değişmekte, en iyi kalibre edilmiş modeller bile ortalama **12 yüzde puanlık** bir güven-doğruluk farkı sergilemektedir. Bu, bir modelin "%95 eminim" dediği pek çok cevabın gerçekte çok daha düşük oranda doğru çıkabileceği anlamına gelir — özellikle HLE gibi çok zor testlerde yüksek güvenle yanlış cevap vermek (bkz. `01_genel_bilgi_ve_muhakeme.md`, HLE bölümü), kalibrasyonun neden ayrı bir eksen olarak raporlanması gerektiğini gösterir.

---

## 1.12. Makro ve Mikro Ortalama Tuzakları

Çok konulu (multi-subject) bir benchmarkta (ör. MMLU'nun 57 alt konusu) toplam skoru hesaplamanın iki yolu vardır:

- **Makro ortalama:** Her konunun accuracy'sini ayrı hesapla, sonra konuların **eşit ağırlıklı** ortalamasını al. Bir konuda 1.000, diğerinde 100 soru olması sonucu değiştirmez.
- **Mikro ortalama:** Tüm konulardaki doğru cevapları topla, toplam soru sayısına böl. Büyük konular sonucu daha fazla etkiler.

### Örnek

Üç görevde model: Mantıksal çıkarım %80 (1.000 soru), Nesne takibi %55 (100 soru), Tarih anlama %65 (100 soru) almışsa:

**Makro ortalama** (her görev eşit ağırlıklı):

\[
(80+55+65)/3=\%66{,}7
\]

**Mikro ortalama** (soru sayısına göre ağırlıklı):

\[
(800+55+65)/1200 \approx \%76{,}7
\]

İki sayı arasında **10 puana yakın** bir fark var — hangisinin raporlandığı, "modelin genel yeteneği" hakkındaki izlenimi kökten değiştirebilir.

### Tuzak

Dengesiz veri kümelerinde (imbalanced classes) mikro ortalama, büyük/kolay sınıflardaki iyi performansın arkasına küçük/zor sınıflardaki kötü performansı gizleyebilir. Örnek bir sınıflandırma çalışmasında (Covertype veri kümesi) Mikro F1 **0,8916** iken Makro F1 **0,8425** çıkmıştır — aradaki fark, modelin azınlık sınıflarında belirgin biçimde daha kötü performans gösterdiğini ortaya koyar. MMLU gibi testlerde de bazı laboratuvarlar makro (57 konunun eşit ağırlıklı ortalaması), bazıları mikro raporlar; bir model kartında hangisinin kullanıldığı açıkça yazmıyorsa, sayı tek başına yanıltıcı olabilir.

---

## 1.13. Bootstrap Güven Aralıkları ve İstatistiksel Anlamlılık

İki model aynı 500 soruluk testte **%82** ve **%83** aldığında, bu 1 puanlık fark gerçek bir yetenek farkı mıdır, yoksa örnekleme gürültüsü müdür?

### Bootstrap yöntemi

1. Test setinden (ör. 500 soru), **yerine koyarak (with replacement)** 500 sorudan oluşan yeni bir örneklem çek.
2. Bu yeni örneklemde her iki modelin accuracy'sini yeniden hesapla.
3. Bu işlemi 1.000–10.000 kez tekrarla.
4. Ortaya çıkan skor dağılımının %2,5 ve %97,5 persentillerini al — bu, **%95 güven aralığıdır**.

### Basitleştirilmiş örnek

500 soruluk bir testte model A %82±1,8, model B %83±1,7 güven aralığıyla raporlanmış olsun:

```text
Model A: %82 (%95 GA: 80,2–83,8)
Model B: %83 (%95 GA: 81,3–84,7)
```

Aralıklar büyük ölçüde örtüştüğü için, tek bir test setinde 1 puanlık farkın **istatistiksel olarak anlamlı olmadığı** söylenebilir. Bunun yerine, aynı sorularda iki modelin **eşleştirilmiş (paired)** doğru/yanlış desenine bakan bir **McNemar testi** veya eşleştirilmiş bootstrap, ham accuracy karşılaştırmasından çok daha güvenilir bir sonuç verir.

### Neden önemli?

"The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing" (Dror, Baumer, Reichart, ACL 2018) ve Card vd. (2020) gibi çalışmalar, NLP/LLM literatüründe modeller arasındaki küçük farkların çoğu zaman anlamlılık testi yapılmadan "daha iyi" ilan edildiğini; küçük test kümelerinin (challenge set) genellikle **yeterli istatistiksel güce (statistical power)** sahip olmadığını göstermiştir. Güncel uygulamada Anthropic gibi laboratuvarlar SWE-bench Verified gibi testlerde tek bir deneme yerine **50 bağımsız deneme üzerinden ortalama** raporlar; OpenAI da GPT-5 sistem kartında güven aralıklarını paylaşır, ancak geçme oranı 0'a veya 1'e çok yakın olan az denemeli testlerde bu aralıkların **yapay biçimde dar** çıkabileceğini de not eder.

### Pratik kural

Liderlik tablosundaki (leaderboard) 1–2 puanlık farklara "kesin üstünlük" gözüyle bakmadan önce şu soruları sormak gerekir: Kaç soru üzerinden hesaplandı? Kaç deneme (trial) ortalandı? Güven aralığı veya anlamlılık testi raporlandı mı?

---

## 1.14. Cohen's Kappa — Hakem Tutarlılığı

İki değerlendirici (iki insan, iki LLM hakemi veya bir insan + bir LLM hakemi) aynı cevapları puanlarken ne kadar **gerçekten** uyuşuyor? Ham uyum yüzdesi (raw agreement) yanıltıcı olabilir, çünkü şans eseri de yüksek uyum çıkabilir — özellikle sınıflar dengesizse.

\[
\kappa=\frac{p_o-p_e}{1-p_e}
\]

Burada \(p_o\) gözlenen (ham) uyum oranı, \(p_e\) rastgele şans eseri beklenen uyum oranıdır.

### Örnek

İki hakem 100 cevabı "iyi/kötü" diye etiketliyor. Ham uyum %90 olsun, ama her iki hakem de cevapların %85'ini zaten "iyi" diye etiketliyorsa (yani sınıflar dengesiz), şans eseri beklenen uyum da yüksektir (\(p_e \approx 0{,}745\)):

\[
\kappa=\frac{0{,}90-0{,}745}{1-0{,}745}\approx0{,}61
\]

Ham uyum "%90" gibi etkileyici görünse de gerçek (şansı düzeltilmiş) uyum orta düzeydedir.

### Yorumlama bandı

| Kappa aralığı | Yorum |
|---|---|
| < 0,20 | Zayıf uyum |
| 0,21–0,40 | Makul uyum |
| 0,41–0,60 | Orta uyum |
| 0,61–0,80 | Önemli uyum |
| > 0,80 | Neredeyse mükemmel uyum |

### Kullanım alanı

LLM-as-a-Judge sistemlerinde, bir laboratuvar yeni bir otomatik hakem tanıttığında genellikle bu hakemin **insan hakemlerle** kappa uyumunu raporlar. Nesnel görevlerde hedef genelde 0,90 üzeri, orta derecede öznel görevlerde 0,70–0,85, doğası gereği öznel tercih verilerinde (ör. RLHF tercih etiketleme) 0,60–0,75 bandı makul kabul edilir. Bir "hakem LLM" ile insan arasındaki kappa 0,3 gibi düşük çıkıyorsa, o hakemin ürettiği win rate veya puanlara güvenmek risklidir.

---

## 1.15. Büyük laboratuvarlar puanları nasıl raporluyor?

Aynı benchmark adı, farklı laboratuvarlarda farklı protokollerle koşulabildiği için model kartlarını okurken şu farklara dikkat etmek gerekir:

| Laboratuvar | Gözlenen pratik |
|---|---|
| **OpenAI** | GPT-5 sistem kartında bazı değerlendirmeler için `pass@1`'in çoklu deneme (bazı testlerde 30 deneme) üzerinden ortalaması kullanılır; "ipuçsuz" (no hints) ve "ipuçlu" ayrımı yapılır; uzunluğa göre ayarlanmış skorlar ve güven aralıkları raporlanır. |
| **Anthropic** | Serbest üretimli (free-form) değerlendirmelerde varsayılan olarak sıcaklık \(T=1\) kullanılır; Claude Opus/Sonnet sistem kartlarında sonuçlar genellikle **5 bağımsız deneme üzerinden ortalama**, SWE-bench Verified gibi ajan testlerinde ise **50 deneme üzerinden ortalama** olarak raporlanır; düşünme bütçesi (thinking budget), bağlam penceresi ve efor (effort) ayarları sabitlenip açıkça belirtilir. |
| **Google DeepMind** | Gemini model değerlendirme raporlarında tüm skorlar varsayılan olarak `pass@1`/**tek deneme**dir; aksi belirtilmedikçe çoğunluk oylaması (majority voting) veya paralel test-zamanı hesaplama kullanılmaz. Küçük benchmarklarda varyansı azaltmak için birden fazla deneme ortalanır; "tek deneme" ile "çoklu deneme" (ör. n=64 çoğunluk oylaması) skorları ayrı sütunlarda gösterilir, birbirine karıştırılmaz. |
| **Meta** | Llama model kartlarında iç değerlendirme kütüphanesi ve harici `lm-evaluation-harness` ile **tutarlı ayarlar** altında tüm modeller (kendi modelleri dahil rakip modeller de) yeniden koşulur; amaç, farklı laboratuvarların kendi bildirdiği sayıları değil, aynı protokolle üretilmiş **adil karşılaştırmalı** sayıları sunmaktır. |

Bu farklılıklar yüzünden, iki farklı model kartından alınan aynı isimli benchmark sayılarını doğrudan yan yana koymak metodolojik olarak risklidir; ideal olan, bağımsız bir üçüncü taraf liderlik tablosunun (LiveBench, LMSYS/Arena, Artificial Analysis, HELM gibi) aynı protokolle ürettiği sayılara bakmaktır.

---

## Kaynaklar

- Chen vd., "Evaluating Large Language Models Trained on Code" (Codex / pass@k), 2021 — https://arxiv.org/abs/2107.03374
- Zheng vd., "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena", NeurIPS 2023 — https://arxiv.org/abs/2306.05685
- Dubois vd., "Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators", COLM 2024 — https://arxiv.org/abs/2404.04475
- Chiang vd., "Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference", 2024 — https://arxiv.org/abs/2403.04132
- "Cheating Automatic LLM Benchmarks: Null Models Achieve High Win Rates", 2024 — https://arxiv.org/html/2410.07137v1
- Gema vd., "Are We Done with MMLU?" (MMLU-Redux), NAACL 2025 — https://arxiv.org/abs/2406.04127
- Dror, Baumer, Reichart, "The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing", ACL 2018 — https://aclanthology.org/P18-1128/
- Card vd., istatistiksel güç ve challenge set sınırlamaları üzerine çalışma (2020), özet: https://www.ruder.io/nlp-benchmarking/
- Cui vd., "OR-Bench: An Over-Refusal Benchmark for Large Language Models", 2024 — https://arxiv.org/html/2405.20947v2
- Röttger vd., "XSTest" aşırı reddetme test seti — https://arxiv.org/html/2405.20947v2 (OR-Bench makalesi içinde XSTest karşılaştırması)
- "The Refusal–Compliance Tradeoff: A Large-Scale Safety Behavior Audit of Large Language Models", 2026 — https://arxiv.org/abs/2605.05427
- LLM kalibrasyon / ECE-Brier bulguları (Claude Opus 4.5 dahil model karşılaştırması) — https://arxiv.org/pdf/2512.16030
- OpenAI, GPT-5 System Card, 2025 — https://cdn.openai.com/gpt-5-system-card.pdf
- Anthropic, Claude Opus 4.5 System Card, Kasım 2025 — https://www.anthropic.com/claude-opus-4-5-system-card
- Anthropic, Claude Opus 4.6 System Card, Şubat 2026 — https://www-cdn.anthropic.com/0dd865075ad3132672ee0ab40b05a53f14cf5288.pdf
- Google DeepMind, Gemini 3 Pro Model Evaluation (Approach, Methodology & Results), Şubat 2026 — https://storage.googleapis.com/deepmind-media/gemini/gemini_3-1_pro_model_evaluation.pdf
- Meta, Llama 3.1 Eval Details / lm-evaluation-harness reprodüksiyon rehberi — https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md
- ARC Prize, ARC-AGI-2 sayfası (maliyet-verimlilik sınırı ve ödül eşiği) — https://arcprize.org/arc-agi/2
- Macro/micro ortalama tuzakları ve Covertype örneği — https://safjan.com/micro-and-macro-averages-in-multiclass-multilabel-problems/
- Cohen's kappa tanımı ve LLM hakemi uygulaması — https://zeroentropy.dev/concepts/cohens-kappa/
