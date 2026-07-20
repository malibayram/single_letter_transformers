# 4. Talimat izleme ve sohbet kalitesi benchmarkları

**Güncelleme tarihi:** 21 Temmuz 2026

Bu bölümdeki benchmarklar iki farklı soruyu yanıtlamaya çalışır:

1. **Model verilen kısıtlara tam olarak uyuyor mu?** (IFEval, M-IFEval, WildIFEval, IFBench — kural tabanlı, nesnel doğrulama)
2. **Modelin cevabı insanların gözünde iyi mi?** (MT-Bench, Chatbot Arena, AlpacaEval, Arena-Hard, WildBench, MixEval — LLM hakem veya insan tercihi)

Bu ayrım önemlidir çünkü bir model talimatlara harfiyen uyup sıkıcı, mekanik cevaplar üretebilir; ya da tam tersine akıcı ve hoş yazıp örtük kısıtları (kelime sayısı, biçim, yasaklı kelimeler) gözden kaçırabilir. İyi bir değerlendirme her iki eksenine de bakar.

Kullanılan gösterim aynı kalır:

- **Beklenen cevap / kural:** Veri kümesindeki gold etiket veya doğrulanabilir kural.
- **Model çıktısı:** Puanlama mekanizmasını göstermek için hazırlanmış, gerçek görev biçimine sadık örnek.
- **Yayımlanmış sonuç:** Makalede veya resmi bir liderlik tablosunda gerçekten raporlanmış sayı.

---

## 4.1. IFEval

IFEval (*Instruction-Following Eval*), Google Research tarafından 2023'te yayımlanan, cevabın "iyi yazılmış" olup olmadığını öznel biçimde değerlendirmek yerine **otomatik olarak doğrulanabilen talimatlara** odaklanan bir benchmarktır. Makale: Zhou ve arkadaşları, "Instruction-Following Evaluation for Large Language Models" (arXiv:2311.07911).

### Tasarım

IFEval'ın temel fikri şudur: bir talimatın yerine getirilip getirilmediğini bir LLM hakeme veya insana sormak yerine, **basit, yorumlanabilir ve deterministik bir programla** kontrol edilebilecek talimatlar seçmek.

- Yaklaşık **500 istem**.
- **25 doğrulanabilir talimat türü**: sözcük sayısı, paragraf sayısı, belirli anahtar kelimeleri belirli sayıda kullanma, yasaklı kelimelerden kaçınma, başlık kullanma, madde işaretiyle yazma, tamamen büyük harfle yazma, JSON biçiminde cevap verme, belirli bir ifadeyle bitirme, iki cevap arasına ayraç koyma gibi kategoriler.
- Her istem bir veya birden fazla verifiable instruction içerir.

### Gerçek talimat türüyle örnek

İstem:

```text
Yapay zekâ hakkında 400 kelimeden uzun bir metin yaz.
"AI" ifadesini en az üç kez kullan.
```

Model cevabı:

- 430 kelime,
- `AI` dört kez geçiyor.

```text
Kural puanı: 2/2
İstem tamamen başarılı: 1
```

İkinci model:

- 450 kelime,
- `AI` yalnızca iki kez.

```text
Kural puanı: 1/2 = %50
İstem tamamen başarılı: 0
```

### İki farklı doğruluk metriği

IFEval ve türevlerinde iki ana metrik raporlanır:

1. **Instruction-level accuracy:** Toplam kaç ayrı kural (istemler genelinde) sağlandı?
2. **Prompt-level accuracy:** Bütün kuralları *aynı anda* sağlayan kaç istem var?

\[
\text{Instruction-level accuracy}=\frac{\text{Sağlanan kural sayısı}}{\text{Toplam kural sayısı}}
\]

\[
\text{Prompt-level accuracy}=\frac{\text{Tüm kuralları sağlayan istem sayısı}}{\text{Toplam istem sayısı}}
\]

Örneğin 100 istemde toplam 300 kuralın 270'i sağlanmış, fakat bütün kurallar yalnızca 70 istemde aynı anda sağlanmışsa:

```text
Instruction accuracy: %90
Prompt accuracy: %70
```

Ayrıca IFEval, her kuralı hem **strict** (katı, ham metin üzerinde) hem **loose** (biçimlendirme farklılıklarına — yıldız işaretleri, tırnak, giriş cümlesi gibi — tolerans tanıyan) modda kontrol eder; bu da toplamda dört sayı (strict/loose × instruction/prompt) üretir. Kod ve veri Google Research'ün GitHub deposunda (`google-research/instruction_following_eval`) açıktır.

### Ne ölçer, ne ölçmez?

IFEval, cevabın *doğru bilgi içerip içermediğini* değil, yalnızca **biçimsel/yapısal kurallara uyup uymadığını** ölçer. Bu yüzden IFEval'da yüksek puan alan bir model aynı anda yanlış bilgi de üretebilir; bu iki eksen bağımsızdır.

---

## 4.2. M-IFEval — çok dilli sürüm

IFEval'ın en büyük sınırlarından biri yalnızca İngilizce olmasıdır. **M-IFEval** (Dussolle ve arkadaşları, arXiv:2502.04688, NAACL 2025 Findings), bu boşluğu kapatmak için Fransızca, Japonca ve İspanyolca'ya genel ve dile özgü talimatlar ekler.

Dile özgü talimatlara örnekler:

- Japonca'da belirli bir nezaket biçimini (*keigo*) kullanma,
- Fransızca'da belirli bir dilbilgisi yapısını (örn. subjonctif kipi) kullanma,
- İspanyolca'da aksan işaretlerine uyma.

Makalenin bulgusu: 8 güncel LLM üzerinde test edildiğinde, dil ve talimat türüne göre başarı **belirgin biçimde dalgalanmaktadır** — yani İngilizce IFEval puanı yüksek bir modelin başka dillerde aynı oranda talimat izlediği varsayılamaz. Bu, çok dilli ürünler geliştiren ekipler için önemli bir uyarıdır.

---

## 4.3. WildIFEval — gerçek kullanıcı kısıtları

Sentetik/şablon talimatlar yerine **gerçek kullanıcı isteklerini** temel almak isteyen bir başka çalışma **WildIFEval**'dır (Lior ve arkadaşları, arXiv:2503.06573).

### Yapı

- **~7.000 gerçek kullanıcı isteminden** türetilmiş, çok kısıtlı (multi-constraint) görevler.
- Kısıtlar, doğal kullanıcı taleplerinden çıkarılıp **8 üst düzey sınıfa** ayrılmıştır (örn. içerme/dışlama, uzunluk, kalite, biçim ve yapı gibi kategoriler).
- WildChat gibi gerçek sohbet kayıtlarından toplanan istemler kullanılır; dolayısıyla kısıtlar IFEval'daki gibi tek tip şablonlara değil, gerçek kullanıcıların doğal biçimde art arda sıraladığı taleplere dayanır.

### Önemli bulgu

WildIFEval, kısıt sayısı arttıkça modellerin **genel** başarı oranının keskin biçimde düştüğünü, ancak **kısıt başına** başarı oranının nispeten sabit kaldığını gösterir. Bu, sorunun tek tek kısıtları anlayamamaktan değil, **aynı anda birden çok kısıtı bir arada tutabilme kapasitesinden** kaynaklandığına işaret eder.

### Basitleştirilmiş örnek

İstem:

```text
Bana 150 kelimeyi geçmeyen, resmi bir dille yazılmış,
madde işareti kullanmayan ve sonunda bir soru soran
bir e-posta taslağı hazır.
```

Burada dört ayrı kısıt vardır: uzunluk, ton, biçim, kapanış. Model üç kısıta uysa bile son kısmı unutursa (soru sormazsa) görev **prompt-level** olarak başarısız sayılır, ama **kısıt-level** analizde "soru sorma" kısıtının kendisi tek başına test edildiğinde model bunu genelde başarabiliyor olabilir — WildIFEval'ın asıl gösterdiği bağlam budur.

---

## 4.4. IFBench — görülmemiş kısıtlar

IFEval'ın önemli bir zayıflığı, modellerin bu 25 sabit kısıt türüne **aşırı uyum sağlamış (overfit)** olabilmesidir: bir model IFEval'daki kısıtları RL sırasında ödül sinyali olarak görmüşse, IFEval puanı artık genel talimat izleme becerisini değil, o spesifik teste özel bir hazırlığı yansıtabilir.

**IFBench** (Pyatkin ve arkadaşları, Allen Institute for AI, arXiv:2507.02833, "Generalizing Verifiable Instruction Following", NeurIPS 2025 Datasets and Benchmarks Track) bu sorunu doğrudan hedefler.

### Yapı

- **58 yeni, elle tasarlanmış, alan dışı (out-of-domain) doğrulanabilir kısıt.**
- Kısıtlar 7 geniş kategoriye ayrılır: **count** (sayma), **ratio** (oran), **words** (kelime), **sentence** (cümle), **format** (biçim), **custom** (özel), **copy** (kopyalama).
- Yaklaşık **300 tek turlu istem** (WildChat'ten alınan gerçek, görülmemiş konuşmalardan türetilmiş) ve yaklaşık **1.300 çok turlu örnek** (kısıtın konuşma turları arasında korunup korunmadığını izole eden).

### Neden önemli?

IFBench'in tasarım felsefesi şudur: gerçek talimat izleme becerisi, modelin eğitim sırasında gördüğü **belirli** kısıt türlerine değil, **hiç görmediği yeni kısıt türlerine** genellenebilmelidir. Makalenin bulgusu, çoğu modelin küçük, sabit bir doğrulanabilir kısıt kümesine güçlü biçimde aşırı uyum sağladığını ve yeni çıktı kısıtlarına iyi genellenemediğini gösteriyor.

### Örnek kısıt türü

```text
İstem: Bir ürün açıklaması yaz. Cümlelerin tam olarak
%40'ı soru cümlesi olsun ve hiçbir cümle 12 kelimeyi geçmesin.
```

Bu, klasik IFEval'ın "en az X kelime" türü basit kısıtlarından farklı olarak **oran tabanlı (ratio)** ve **çoklu** bir kısıttır; modelin cümleleri sayıp oranı hesaplaması ve buna göre üretim yapması gerekir.

---

## 4.5. MT-Bench

MT-Bench (Zheng ve arkadaşları, "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena", arXiv:2306.05685, NeurIPS 2023), modele **iki turlu** sohbet soruları yöneltir. İkinci soru çoğu zaman ilk cevaba bağlıdır ve modelin bağlamı koruyup koruyamadığını test eder.

### Yapı

- **80 çok turlu soru**, 8 kategoriye ayrılmış: yazma, rol yapma, çıkarım (extraction), muhakeme, matematik, kodlama, STEM bilgisi, beşerî bilimler/sosyal bilimler bilgisi.
- Her kategoriden 10 soru, her biri 2 turlu.
- Güçlü bir LLM (orijinal makalede GPT-4) hakem olarak cevapları **1-10 arası** puanlar.

### Örnek

Birinci tur:

```text
Bir girişim için üç pazarlama fikri öner.
```

İkinci tur:

```text
İkinci fikri 5.000 TL bütçeyle uygulanabilir hâle getir.
```

Model ikinci turda hangi fikrin "ikinci fikir" olduğunu hatırlamalı ve bütçe kısıtına uymalıdır.

Hakem LLM cevapları doğruluk, ilgililik, ayrıntı, akıcılık ve ikinci tur bağlamını koruma bakımından puanlar.

### Örnek puan

Sekiz kategoride:

```text
8.0, 7.5, 8.5, 6.0, 7.0, 8.0, 7.5, 7.5
```

Ortalama:

\[
60/8=7{,}5
\]

### Yayımlanmış sonuç: hakem güvenilirliği

Makalenin en önemli bulgusu, MT-Bench sorularına ait GPT-4 hakem kararlarının, **çiftli insan uzman tercihleriyle %80'in üzerinde uyum** gösterdiğidir — bu da insanların birbirleriyle olan uyum düzeyine (yaklaşık aynı seviye) yakındır. Aynı çalışma, LLM hakemlerin sistematik önyargılarını da belgeler:

- **Konum (position) önyargısı:** Hakem, cevapların gösterilme sırasına duyarlı olabilir.
- **Ayrıntı/uzunluk (verbosity) önyargısı:** Daha uzun cevaplar haksız yere ödüllendirilebilir.
- **Kendini kayırma (self-enhancement) önyargısı:** Hakem model, kendi ailesinden gelen (benzer tarzda yazılmış) cevapları kayırabilir.
- **Sınırlı muhakeme yeteneği:** Hakem, karmaşık matematik/mantık hatalarını her zaman fark edemeyebilir.

### Sınırlaması

Hakem model değişirse skor değişebilir. Aynı cevap farklı hakemlerde 7 veya 9 alabilir. Ayrıca 80 soruluk küme, geniş yetenek yelpazesini temsil etmek için küçüktür; bu yüzden MT-Bench günümüzde tek başına değil, genelde daha geniş bir değerlendirme paketinin bir parçası olarak kullanılır.

---

## 4.6. Chatbot Arena (LMArena)

Chatbot Arena'da (bugünkü adıyla **LMArena**) kullanıcı aynı isteme iki anonim modelin cevabını görür ve "A daha iyi", "B daha iyi", "Berabere" veya "İkisi de kötü" seçeneklerinden birini işaretler. Model isimleri oy verildikten sonra açıklanır; amaç marka önyargısını azaltmaktır.

### Puanlama: Bradley–Terry

Her model her modelle eşit sayıda karşılaşmadığından yalnızca ham kazanma oranı yeterli değildir. Bradley–Terry modeli, karşılaşma sonuçlarından her modele gizli bir "güç puanı" tahmin eder. Puanın mutlak anlamı yoktur; **diğer modeller karşısındaki göreli konumu** gösterir.

### Style Control (biçim kontrolü)

LMArena ekibi 2024 sonlarında **Style Control** adlı bir sıralama seçeneği ekledi. Amaç, ham Arena puanının biçimsel etkenlerden (cevap uzunluğu, markdown kullanımı — kalın yazı, başlıklar, madde işaretleri) ne kadar etkilendiğini ayıklamaktır. Style Control, bu biçimsel özellikleri bir regresyon modeliyle kontrol ederek, modelin **saf içerik kalitesine** daha yakın bir sıralama üretmeye çalışır. Bir modelin ham sıralamada yüksek, Style Control sıralamasında daha düşük çıkması, avantajının bir kısmının biçimsel sunumdan (uzun, düzenli, madde işaretli cevaplar) geldiğine işaret eder.

### 2026 ortası durum (örnek görünüm)

LMArena skorları haftalık, hatta günlük değişebilen canlı bir sıralamadır; bu nedenle aşağıdaki sayılar **belirli bir tarihe ait anlık görüntü** olarak okunmalıdır, sabit bir gerçek olarak değil:

```text
[Nisan 2026 civarı, metin arenası - yaklaşık görünüm]
1. Claude Opus 4.6 (Thinking)   ~1504 Elo
2. GPT-5.4 (High)               ~1502 Elo
3. Gemini 3.1 Pro Preview       ~1493 Elo
```

Aynı dönemde kodlama alt kategorisinde farklı bir model (rapor edilen: Claude Opus 4.6) liderlik etmiştir; bu da genel sıralama ile kategori bazlı sıralamaların (kodlama, uzun bağlam, zor istemler) birbirinden **farklı kazananlar** üretebileceğini gösterir. LMArena artık tek bir genel tabloyla değil, kategori bazlı (Text, Coding, Vision, Long Context, Hard Prompts vb.) ayrı liderlik tablolarıyla raporlama yapmaktadır. Kesin ve güncel sayılar için `lmarena.ai` üzerindeki canlı tablo esas alınmalıdır; buradaki değerler yalnızca 2026 ortasındaki genel eğilimi göstermek içindir.

### Örnek karşılaşma

Kullanıcı:

```text
Bu sözleşmedeki fesih maddesini sade Türkçe ile açıkla.
```

Model A hukuken doğru ve anlaşılır bir açıklama verir. Model B kısa fakat maddede olmayan bir cayma hakkı uydurur. Kullanıcı A'yı seçerse karşılaşma "A kazandı, B kaybetti" olarak kaydedilir. Binlerce karşılaşma Bradley–Terry modeliyle birleştirilir.

### Neyi iyi ölçer, neyi kötü ölçer?

| İyi ölçtüğü | Kötü ölçebileceği |
|---|---|
| Gerçek kullanıcı tercihleri | İnsanların fark etmediği teknik yanlışlar |
| Açık uçlu cevap kalitesi | Az kullanılan diller ve uzmanlık alanları |
| Yazım, yararlılık, iletişim | Güzel yazılmış fakat yanlış cevaplar |
| Gündelik kullanım çeşitliliği | Model kimliği/uzunluk gibi dolaylı önyargılar |

Arena puanı bir "doğruluk yüzdesi" değildir.

---

## 4.7. AlpacaEval 2.0 ve Length-Controlled Win Rate

AlpacaEval, aday modelin cevabını sabit bir referans modelin cevabıyla karşılaştırır. AlpacaEval 2.0'da referans ve otomatik hakem olarak GPT-4 (Preview) tabanlı bir düzen kullanılır; sonuçlar `tatsu-lab.github.io/alpaca_eval` üzerinde canlı olarak yayımlanır.

### Örnek

İstem:

```text
Evden çalışan biri için odaklanmayı artıracak uygulanabilir bir günlük plan hazırla.
```

Referans cevap ile aday cevap hakem modele gösterilir.

100 istemde aday model 54 kez kazanır, 36 kez kaybeder, 10 kez berabere kalırsa (beraberlikler yarım sayılarak):

\[
(54+5)/100=\%59
\]

win rate elde edilir.

### Length-Controlled Win Rate (LC-WR)

GPT-4 gibi LLM hakemlerin **daha uzun cevapları haksız yere tercih etme eğilimi** iyi belgelenmiş bir sorundur. AlpacaEval 2.0'ın en önemli katkısı, bu uzunluk önyargısını istatistiksel olarak ayıklayan **length-controlled win rate**'tir.

Yöntem özetle: bir **lojistik regresyon** modeli, kazanma olasılığını hem içerik kalitesi hem de uzunluk farkı üzerinden tahmin eder; ardından uzunluk-farkı özniteliği sıfıra sabitlenerek "eğer bütün modellerin cevapları referans kadar uzun olsaydı, kazanma oranı ne olurdu?" sorusunun cevabı hesaplanır.

Bir modelin:

```text
Ham win rate: %65
Uzunluk kontrollü win rate: %54
```

alması, avantajının önemli bir kısmının yalnızca daha uzun yazmasından geldiğini gösterebilir.

### Kalan sınır

LC-WR uzunluk önyargısını azaltsa da, hâlâ GPT-4 tabanlı bir hakem kullanır; bu nedenle **GPT-4'ün kendi yazım tarzına ve tercihine yakın çıktı üreten** (örn. GPT-4 çıktılarıyla ince ayarlanmış) modelleri dolaylı olarak kayırma riski tamamen ortadan kalkmaz.

---

## 4.8. Arena-Hard / Arena-Hard-Auto

**Arena-Hard**, LMArena ekibinin MT-Bench'e göre **daha zor, daha otomatik ve insan tercihleriyle daha yüksek uyumlu** bir halef olarak geliştirdiği benchmarktır (LMSYS blog, "From Live Data to High-Quality Benchmarks: The Arena-Hard Pipeline", Nisan 2024).

### Tasarım

- **BenchBuilder** adlı bir veri hattı, canlı Chatbot Arena trafiğinden **ayırt ediciliği yüksek, zor, gerçek kullanıcı istemlerini** otomatik seçer (özellik çıkarımı + LLM ile filtreleme yoluyla).
- İlk sürüm **500 gerçek dünya kullanım örneği** içerir.
- Değerlendirme, güçlü bir LLM hakemin (orijinalde GPT-4-Turbo, sonraki sürümlerde GPT-4.1/Gemini-2.5) **çiftli (pairwise) karşılaştırma** yapmasıyla gerçekleşir; hakem önce kendi cevabını taslak olarak üretir, sonra karşılaştırma yapar (bu, yargı hatalarını azaltmayı amaçlar).
- Sonuçlar yine **Bradley–Terry** ile toplulaştırılır.

### Yayımlanmış sonuç

LMSYS'in duyurusuna göre Arena-Hard, **canlı Chatbot Arena'daki insan tercihlerinin yaklaşık %89'unu** LLM-hakem kullanarak yakalamaktadır — bu oran, aynı karşılaştırmada MT-Bench'in ulaştığı uyum oranından daha yüksektir.

### Arena-Hard-v2.0 (Nisan 2025)

Arena-Hard'ın ikinci sürümü şunları ekler:

- **500 yeni, taze istem** (LMArena'dan, veri sızıntısını azaltmak için yenilenmiş),
- **250 yaratıcı yazım (creative writing) istemi** — ilk sürümde eksik olan bir kategori,
- Daha güçlü/daha yeni hakem modelleri (GPT-4.1, Gemini-2.5),
- **30'dan fazla dilde** çok dilli destek,
- Daha güçlü temel (baseline) modeller karşısında karşılaştırma.

### Örnek

İstem (yazılım mühendisliği türünden, Arena-Hard'ın tipik zorluk seviyesine benzer):

```text
Bir REST API'de eşzamanlı istekler yüzünden oluşan
"race condition" hatasını düzelten bir kod yamasını,
geriye dönük uyumluluğu bozmadan yaz.
```

Bu tür istemler MT-Bench'teki genel sohbet sorularından daha teknik ve ayırt edicidir; zayıf bir model kolayca gözden kaçırılabilecek bir eşzamanlılık hatası bırakabilirken güçlü bir model kilit (lock) veya atomik işlem kullanabilir.

---

## 4.9. WildBench

**WildBench** (Lin ve arkadaşları, arXiv:2406.04770, ICLR 2025), gerçek kullanıcılardan toplanan zorlu görevlerle modelleri değerlendiren bir başka canlı benchmarktır. Sorular WildChat gibi gerçek sohbet kayıtlarından süzülür; amaç laboratuvar ortamında üretilmiş yapay istemler yerine **gerçek dünyadaki çeşitliliği ve zorluğu** yansıtmaktır.

### Yöntem: görev-özel kontrol listeleri

WildBench, LLM hakemlerin güvenilirliğini artırmak için her göreve özel bir **kontrol listesi (checklist)** üretir; hakem, cevabı bu listedeki maddelere göre adım adım (chain-of-thought benzeri) değerlendirir. Bu, hakemin yalnızca genel bir izlenimle değil, somut kriterlerle puan vermesini sağlamayı amaçlar.

### İki metrik

- **WB-Score:** Tekil bir cevabın kalitesini doğrudan puanlar.
- **WB-Reward:** Çiftli karşılaştırmaya dayanır; beş olası sonuç vardır — "A çok daha iyi", "A biraz daha iyi", "Berabere", "B biraz daha iyi", "B çok daha iyi". Bu ince taneli (fine-grained) skala, basit "kazandı/kaybetti" ikiliğinden daha fazla bilgi taşır.

### Örnek

İstem:

```text
Elimde SQLite veritabanı var, 2 milyon satırlık bir tabloda
belirli bir sorgu çok yavaş çalışıyor. Neden yavaş olabileceğini
ve nasıl hızlandırabileceğimi adım adım anlat.
```

Kontrol listesi maddeleri örnek olarak şunları içerebilir:

```text
[ ] İndeks eksikliği olasılığını değerlendirdi mi?
[ ] EXPLAIN QUERY PLAN kullanmayı önerdi mi?
[ ] En az bir somut SQL örneği verdi mi?
[ ] Yanlış/güvenilmez bir iddiada bulunmadı mı?
```

Hakem model, cevabı bu maddelere göre puanlayarak nihai WB-Score'u belirler.

---

## 4.10. MixEval ve MixEval-Hard

**MixEval** (Ni ve arkadaşları, arXiv:2406.06565, NeurIPS 2024), gerçek kullanıcı sorgularının çeşitliliği ile klasik benchmarkların **gerçek referans cevaba (ground truth) dayalı** güvenilirliğini birleştirmeyi hedefler.

### Yöntem

İki aşamalı bir hat kullanılır:

1. **Web sorgusu tespiti:** Common Crawl gibi büyük bir korpustan gerçek kullanıcı sorguları tespit edilir.
2. **Benchmark karışımı (mixture):** Bu gerçek sorgular, anlam bakımından en yakın oldukları **mevcut benchmarklardaki** (MMLU, GSM8K vb.) sorularla eşleştirilir; böylece hem gerçekçi soru dağılımı hem de nesnel, ground-truth tabanlı puanlama bir arada elde edilir.

### MixEval-Hard

Ayırt ediciliği artırmak için MixEval-Hard, modelleri en çok zorlayan alt kümeye odaklanır ve veri sızıntısını azaltmak için **her ay güncellenir**.

### Yayımlanmış sonuç

Makaleye göre MixEval, Chatbot Arena ile **0,93** (standart) ve **0,96** (Hard sürüm) korelasyon gösterir; bunu MMLU'yu çalıştırma süresinin/maliyetinin yaklaşık **%6'sı** ile başarır. Bu, MixEval'ı hem hızlı hem de insan tercihiyle yüksek uyumlu bir "vekil" (proxy) ölçüm hâline getirir.

### Neden önemli?

Chatbot Arena gibi insan oylamasına dayalı sistemler yavaş ve pahalıdır (canlı trafik, moderasyon, zaman gerektirir). MMLU gibi statik testler ise gerçek kullanıcı davranışını iyi temsil etmeyebilir. MixEval, ikisi arasında bir köprü kurmaya çalışır.

---

## 4.11. RewardBench — hakemlerin hakemi

Buraya kadarki benchmarkların çoğu **LLM'leri** değerlendiriyordu. Ama LLM-as-a-judge yaygınlaştıkça yeni bir soru ortaya çıktı: **hakem modellerin/ödül modellerinin kendisi ne kadar güvenilir?**

**RewardBench** (Allen Institute for AI / Ai2), RLHF'te kullanılan **ödül modellerini (reward models)** ve LLM hakemleri değerlendirmek için tasarlanmış ilk büyük ölçekli, açık liderlik tablolu benchmarktır.

### Yapı

- **2.985 görev**, 4 ana kategoride: **Chat**, **Chat-Hard**, **Safety**, **Reasoning** — toplam 23 alt kategori.
  - **Chat:** Genel sohbet ortamında "kötü" bir cevap ile "iyi" bir cevabı ayırt edebiliyor mu?
  - **Chat-Hard:** "İyi" bir cevap ile "çok iyi" bir cevabı ayırt edebiliyor mu? (Daha ince bir ayrım.)
  - **Safety:** Reddetmesi gereken bir isteğe uyum gösteren cevap ile doğru biçimde reddeden cevap arasında tercihi doğru yapabiliyor mu?
  - **Reasoning:** Matematik/kod gibi alanlarda doğru çözüm ile yanlış (ama ikna edici görünen) çözüm arasında ayrım yapabiliyor mu?
- Her görev bir istem, bir "seçilen" (chosen) ve bir "reddedilen" (rejected) cevap içerir; etiketler insan tarafından doğrulanmıştır.
- Zorluk seviyesini artırmak için LLMBar'dan alınan **kasıtlı olarak zorlaştırılmış (adversarial)** örnekler, doğru/hatalı kod çiftleri ve reddetmesi/etmemesi gereken güvenlik örnekleri kullanılır.

### Puanlama

\[
\text{Accuracy}=\frac{\text{Doğru sıralanan çift sayısı}}{\text{Toplam çift sayısı}}
\]

Bir ödül modeli 100 çiftten 82'sinde "chosen" cevabı "rejected" cevaptan daha yüksek puanlarsa:

\[
82/100=\%82
\]

### Neden önemli?

RLHF, DPO gibi hizalama (alignment) yöntemlerinin kalitesi, büyük ölçüde kullanılan ödül modelinin/hakemin güvenilirliğine bağlıdır. Zayıf bir ödül modeli, yanlış davranışları ödüllendirip modelin kalitesini düşürebilir. RewardBench, "bu hakemi ne kadar güvenebiliriz?" sorusuna nesnel bir cevap aramanın standart yoludur; benzer mantıkla geliştirilen **JudgeBench** gibi takip çalışmaları, bilgi, muhakeme, matematik ve kodlama alanlarında hakem modelleri özellikle zorlayan çiftler üzerinde test eder.

---

## 4.12. Karşılaştırma tablosu

| Benchmark | Değerlendirici türü | Neyi hedefler | Bilinen önyargılar / sınırlar |
|---|---|---|---|
| **IFEval** | Kural tabanlı (program) | Biçimsel/yapısal kısıtlara uyma (25 tür) | İçerik doğruluğunu ölçmez; sabit kısıt setine aşırı uyum riski |
| **M-IFEval** | Kural tabanlı (program) | Çok dilli talimat izleme (FR/JA/ES) | Yalnızca 3 dile ek dil kapsıyor; İngilizce'deki kadar geniş değil |
| **WildIFEval** | Kural tabanlı + LLM ayrıştırma | Gerçek kullanıcı çoklu-kısıt istekleri | Kısıt çıkarımı otomatik yapıldığından gürültülü olabilir |
| **IFBench** | Kural tabanlı (program) | Görülmemiş/alan dışı 58 kısıt, genelleme | Yeni kısıtlar da zamanla eğitim verisine sızabilir |
| **MT-Bench** | LLM hakem (1-10 puan) | Çok turlu sohbet kalitesi, bağlam koruma | Uzunluk, konum, kendini kayırma önyargıları; küçük soru seti (80) |
| **Chatbot Arena** | İnsan (çiftli oylama) | Gerçek kullanıcı tercihi, genel kullanılabilirlik | Uzunluk/biçim önyargısı (Style Control ile kısmen giderilir); teknik hataları insan fark etmeyebilir |
| **AlpacaEval 2.0 (LC)** | LLM hakem (GPT-4, çiftli) | Tek turlu talimat takip + genel kalite | LC bile GPT-4'ün kendi tarzına yakın çıktıyı kayırabilir |
| **Arena-Hard(-Auto v2)** | LLM hakem (çiftli, Bradley–Terry) | Zor, ayırt edici gerçek dünya istemleri | Hakem modelin kendi önyargıları; insan-Arena'ya göre ~%89 uyum (tam değil) |
| **WildBench** | LLM hakem (kontrol listeli) | Gerçek kullanıcı zorlu görevleri | Kontrol listesi kalitesine bağımlı; hakem yine de LLM |
| **MixEval / -Hard** | Kural/ground-truth tabanlı (karma) | Arena tercihine yakın, hızlı, ucuz ölçüm | Eşleştirme kalitesine bağlı; temel benchmarkların sınırlarını miras alır |
| **RewardBench** | İnsan etiketli çiftler (statik) | Ödül modeli / LLM hakem güvenilirliği | Statik veri seti; adversarial örnekler gerçek dağılımı abartabilir |

---

## 4.13. Hangi benchmark ne zaman kullanılır?

Pratik bir seçim rehberi:

```text
Modelin kurallara/kısıtlara harfiyen uyup uymadığını mı merak ediyorsunuz?
  → IFEval / IFBench (görülmemiş kısıtlarla) / WildIFEval (gerçekçilik için)

Modelin genel sohbet kalitesini, insan tercihine yakın biçimde mi
karşılaştırmak istiyorsunuz?
  → Chatbot Arena (mümkünse Style Control açık) veya Arena-Hard-v2

Hızlı, ucuz, otomatik ama insan tercihiyle yüksek korelasyonlu bir
proxy mi arıyorsunuz?
  → Arena-Hard-Auto veya MixEval

Kendi ödül modelinizi / LLM hakeminizi mi doğrulamak istiyorsunuz?
  → RewardBench (veya JudgeBench gibi takip çalışmaları)

Uzun/karmaşık gerçek kullanıcı görevlerinde ince taneli
karşılaştırma mı istiyorsunuz?
  → WildBench (WB-Score / WB-Reward)
```

### Genel uyarı

Bu bölümdeki hiçbir benchmark tek başına yeterli değildir. Bir model IFEval'da mükemmel, ama Chatbot Arena'da vasat olabilir (kurallara uyuyor ama sıkıcı yazıyor); ya da tam tersi (akıcı ama kısıtları kaçırıyor). Üretim kararı verirken bu bölümdeki testleri, doğruluk/halüsinasyon benchmarkları (bkz. dosya 05) ve alan-özel testlerle **birlikte** okumak gerekir.
