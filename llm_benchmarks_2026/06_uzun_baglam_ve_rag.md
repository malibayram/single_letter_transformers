# 06. Uzun Bağlam ve RAG Benchmarkları

> Bu bölüm, büyük dil modellerinin (LLM) uzun bağlam pencerelerini (128K, 1M, hatta 10M+ token) ne kadar *gerçekten* kullanabildiğini ve retrieval-augmented generation (RAG) sistemlerinin ne kadar güvenilir olduğunu ölçen benchmarkları ele alır. 2023-2024 döneminde "context window yarışı" (128K → 1M → 10M token) modelleri pazarlama açısından öne çıkardı; ama 2024-2026 arasında yayımlanan araştırmalar, **bağlam penceresinin büyüklüğü ile modelin o bağlamı etkin biçimde kullanabilmesinin aynı şey olmadığını** kanıtladı. Bu dosyadaki benchmarklar tam olarak bu boşluğu ölçmek için tasarlandı.

---

## 1. Neden "uzun bağlam" ayrı bir değerlendirme kategorisi?

Kısa bağlamlı (birkaç bin token) klasik NLP benchmarkları (SQuAD, GLUE vb.) bir modelin dil anlama becerisini ölçer. Ama bir model 200K token'lık bir bağlamda:

- Bağlamın **ortasındaki** bilgiyi bulabiliyor mu? ("lost in the middle" etkisi)
- Birden fazla, birbirine uzak konumdaki bilgiyi **birleştirip** akıl yürütebiliyor mu? (multi-hop)
- Alakasız ama *anlamca benzer* metinlerle (distractor) karşılaştığında yanılmıyor mu?
- Bağlam uzadıkça basit görevlerde bile (ör. metni birebir kopyalama) performansı düşüyor mu?

Bu sorular klasik "doğruluk skoru" ile değil, bağlam uzunluğuna göre **performans eğrisi** çizerek yanıtlanır. Bu yüzden bu bölümdeki neredeyse her benchmark, sonucu tek bir sayı değil, "bağlam uzunluğu × görev türü" ısı haritası (heatmap) olarak raporlar.

---

## 2. LongBench ve LongBench v2

### LongBench (orijinal, 2023)

Tsinghua Üniversitesi tarafından yayımlanan **LongBench**, İngilizce ve Çince'de **21 veri kümesi** içeren, 6 görev kategorisine (tek-doküman QA, çok-doküman QA, özetleme, few-shot öğrenme, sentetik görevler, kod tamamlama) ayrılmış iki dilli bir benchmarktır. Ortalama bağlam uzunluğu İngilizce'de ~6.711 kelime, Çince'de ~13.386 karakterdir — yani bugünün ölçeğine göre aslında "orta uzunlukta" (çoğu örnek 4K-20K token aralığında) kabul edilir.

- Kaynak: [LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding](https://arxiv.org/abs/2308.14508)

### LongBench v2 (Aralık 2024)

Orijinal LongBench'in "artık çok kolay" hale gelmesi üzerine yayımlanan **LongBench v2**, çok daha zorlu ve gerçekçi bir versiyon sunar:

| Özellik | Değer |
|---|---|
| Soru sayısı | 503 zorlu çoktan seçmeli soru |
| Bağlam uzunluğu aralığı | 8.000 – 2.000.000 kelime |
| Görev kategorisi | 6 ana kategori (tek-doküman QA, çok-doküman QA, uzun in-context öğrenme, uzun diyalog geçmişi anlama, kod deposu anlama, uzun yapılandırılmış veri anlama) |
| Zorluk seviyesi | İnsan uzmanların bile zaman baskısı altında ~%53,7 doğruluk gösterdiği bir seviyede tasarlandı |

LongBench v2'nin asıl amacı, modellerin **derin anlama ve akıl yürütme** gerektiren senaryolarda (sadece "metni bul" değil, "metinden çıkarım yap") test edilmesidir.

- Kaynak: [LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks (arXiv:2412.15204)](https://arxiv.org/abs/2412.15204)

**Beklenen cevap** (LongBench v2, kod deposu anlama görevinden basitleştirilmiş bir örnek):
> "Bu 300K token'lık kod deposunda, `UserAuthManager` sınıfının `refresh_token` metodunun hangi diğer sınıflar tarafından çağrıldığını ve bu çağrıların hangi koşullar altında başarısız olabileceğini açıklayın."

**Model çıktısı (örnek/illustrative):** Küçük bağlamlı modeller genellikle sadece `refresh_token` metodunun tanımını bulup döndürür, çağıran sınıfları (call sites) gözden kaçırır çünkü bunlar dosyanın çok uzağındadır — bu, LongBench v2'nin tam olarak yakalamayı hedeflediği "sığ retrieval, derin anlama yok" hatasıdır.

---

## 3. Needle-in-a-Haystack (NIAH) — klasik test ve zayıflıkları

### Ne ölçer?

**Needle-in-a-Haystack (NIAH)**, Greg Kamradt tarafından 2023'te popülerleştirilen basit ama etkili bir testtir: rastgele, alakasız bir metin yığınının (haystack) içine tek bir cümle ("iğne", ör. "Gizli sayı 7492'dir") gömülür ve modelden bu bilgiyi bağlamın farklı derinliklerinde (%0, %10, ..., %100) ve farklı bağlam uzunluklarında geri getirmesi istenir. Sonuç genelde bir **ısı haritası** (x ekseni: bağlam uzunluğu, y ekseni: iğnenin derinliği, renk: doğruluk) olarak sunulur.

**Beklenen cevap:** "Gizli sayı 7492'dir."

**Yayımlanmış sonuç:** GPT-4 ve Claude gibi öncü modeller 2024 itibarıyla klasik tek-iğne NIAH testinde 128K-200K bağlamda genellikle %95+ doğruluk gösterdi — bu da testin "çözülmüş" görülmesine ve eleştirilmesine yol açtı.

### Bilinen zayıflıkları

1. **Gerçek dünya ile düşük korelasyon.** Güçlü NIAH performansı, çeşitli pratik uzun-bağlam görevlerinde güçlü performansla tutarlı biçimde korele değildir; NIAH'ta neredeyse mükemmel skor alan modeller çok-adımlı akıl yürütme gerektiren gerçekçi görevlerde başarısız olabilir.
2. **Retrieval'ı aşırı basitleştirir.** Çoğu gerçek RAG uygulaması tek değil, **çok sayıda** ilgili gerçeği indeksten çekip bunlar üzerinde akıl yürütmeyi gerektirir; araştırmalar modele daha fazla gerçek getirmesi ve bunlar üzerinde muhakeme yapması istendiğinde performansın belirgin biçimde düştüğünü gösteriyor (bkz. Multi-Needle NIAH, aşağıda).
3. **Bilgi-yoğun bağlamlarda çöküyor.** Bağlam sadece sorguyla ilgili, birbirine bağlı bilgilerden oluştuğunda (yani "gürültü" azaldığında bile) modellerin performansı ciddi biçimde bozuluyor — 200 token gibi kısa bir bağlamda bile bazı modeller gerekli bilgiyi tam yakalayamıyor. Bu da "her yeri yeşil" ısı haritalarının ve "neredeyse mükemmel uzun-bağlam performansı" iddialarının güvenilirliğini sorgulatıyor.
4. **Negatif örneklerde halüsinasyon.** Bazı üst düzey modeller (ör. GPT-4o), iğne bağlamda **olmadığında** bile bir cevap "uydurma" eğilimi gösteriyor — yani NIAH sadece "buluyor mu" değil, "bulamadığını kabul edebiliyor mu" sorusunu da (IDK — "I Don't Know" görevleri, bkz. §9 Michelangelo) gündeme getiriyor.
5. **Konum yanlılığı (positional bias).** Bilginin bağlamın başında mı, ortasında mı, sonunda mı olduğu doğruluğu büyük ölçüde etkiliyor — "lost in the middle" etkisi.

### NIAH doğruluğunun formülleştirilmesi

Klasik NIAH ısı haritasındaki her hücre, belirli bir bağlam uzunluğu \(L\) ve iğne derinliği \(d\) için tekrarlanan denemelerin ortalama doğruluğunu gösterir:

\[
\text{Acc}(L, d) = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}\left[\hat{y}_i = y_i \mid \text{context\_len}=L,\ \text{depth}=d\right]
\]

burada \(N\) o hücre için yapılan deneme sayısı, \(\hat{y}_i\) modelin çıktısı, \(y_i\) gerçek (ground-truth) iğne değeridir. Isı haritasının "her yerde yeşil" (uniform yüksek doğruluk) görünmesi çekicidir ama §3'te belirtilen zayıflıklar nedeniyle bu tek başına güvenilir bir "uzun bağlam yetkinliği" göstergesi sayılmamalıdır.

### Multi-Needle NIAH

LangChain ve diğer araştırmacılar, tek-iğne testin yetersizliğine yanıt olarak **Multi-Needle NIAH** varyantını popülerleştirdi: bağlama birden fazla (ör. 10) bağımsız gerçek gömülür ve modelden bunların tümünü (veya belirli bir alt kümesini) geri getirmesi istenir. Bulgular: iğne sayısı arttıkça ve bağlam uzadıkça geri getirme doğruluğu tutarlı biçimde düşer — bu, tek-iğne NIAH'ın neden RAG sistemlerini yeterince temsil etmediğini açıklar.

- Kaynak: [Multi Needle in a Haystack (LangChain Blog)](https://blog.langchain.com/multi-needle-in-a-haystack/)

---

## 4. RULER (NVIDIA)

**RULER**, NVIDIA tarafından geliştirilen sentetik bir benchmark olup "gerçek bağlam boyutu nedir?" sorusuna cevap arar — yani bir modelin *iddia ettiği* bağlam penceresi (ör. "128K destekler") ile *etkin biçimde kullanabildiği* bağlam penceresi arasındaki farkı ortaya çıkarır.

### Görev yapısı: 13 görev, 4 kategori

| Kategori | Görev sayısı | Açıklama |
|---|---|---|
| Retrieval (NIAH varyantları) | 8 | Tek iğne, çoklu iğne, çoklu anahtar, çoklu değer gibi 8 farklı NIAH varyantı |
| Multi-hop Tracing (Değişken Takibi) | 1 | Bir "değişken bağlama zinciri" (variable binding chain) izlenip doğru zincirin geri döndürülmesi gerekir |
| Aggregation (Toplulaştırma) | 2 | Bağlamdaki en sık geçen kelime(ler)in bulunması gibi, bağlamın **tamamının** işlenmesini gerektiren görevler |
| Question Answering | 2 | Gürültü eklenmiş gerçek dünya QA görevleri |

### 13 görevin tam listesi (RULER v1)

| # | Görev adı | Kategori |
|---|---|---|
| 1 | Single NIAH (tek anahtar-değer) | Retrieval |
| 2 | Multi-key NIAH 1 | Retrieval |
| 3 | Multi-key NIAH 2 | Retrieval |
| 4 | Multi-key NIAH 3 | Retrieval |
| 5 | Multi-value NIAH | Retrieval |
| 6 | Multi-query NIAH | Retrieval |
| 7 | Needle çeşitleme (ör. sayı/kelime tipi değişimi) | Retrieval |
| 8 | "Haystack" tipi çeşitleme (tekrar eden/rastgele/gerçek metin gürültüsü) | Retrieval |
| 9 | Değişken Takibi (Variable Tracking, VT) | Multi-hop Tracing |
| 10 | Common Words Extraction (CWE) | Aggregation |
| 11 | Frequent Words Extraction (FWE) | Aggregation |
| 12 | QA görev 1 (gürültü eklenmiş gerçek QA seti, ör. SQuAD tabanlı) | Question Answering |
| 13 | QA görev 2 (farklı bir gerçek QA seti üzerinde) | Question Answering |

Bu tablo RULER'ın kamuya açık uygulamasındaki görev isimlendirmesinin sadeleştirilmiş bir özetidir; tam parametre uzayı (kaç anahtar, kaç distractor vb.) için resmi depoya bakılmalıdır.

### Neden önemli?

RULER'ın en çarpıcı bulgusu şudur: **17 uzun-bağlam LLM** değerlendirildiğinde, modeller klasik (tek iğneli) NIAH testinde neredeyse mükemmel doğruluk gösterse bile, bağlam uzunluğu arttıkça RULER'ın 13 görevinin ortalamasında büyük düşüşler yaşanıyor. Yani bir model "128K context window" etiketiyle satılsa da, RULER'a göre etkin/kullanılabilir bağlam uzunluğu bunun çok altında kalabiliyor.

- Kaynak: [RULER: What's the Real Context Size of Your Long-Context Language Models? (GitHub)](https://github.com/NVIDIA/RULER), [arXiv:2404.06654](https://arxiv.org/html/2404.06654v1)

**Yayımlanmış sonuç:** RULER liderlik tablosunda (llm-stats.com üzerinden takip edilen güncel sürüm) 2026 ortası itibarıyla NVIDIA'nın Nemotron 3 Ultra (550B, 55B aktif parametre) modeli ~0,947 skorla listeyi açık ara önde götürüyor — bu da RULER'ın hâlâ zorlayıcı bir ayraç (differentiator) olarak kaldığını gösteriyor.

```bash
# RULER'ı yerel olarak koşmak için (basitleştirilmiş örnek)
git clone https://github.com/NVIDIA/RULER
cd RULER
python scripts/run.py \
  --model_name my-model \
  --task niah_multikey_1 --task vt --task cwe \
  --max_seq_length 131072
```

---

## 5. OneRuler — RULER'ın çok dilli versiyonu (26 dil)

**OneRuler**, RULER'ın İngilizce-merkezli sentetik görev tasarımını **26 dile** genişleterek çok dilli uzun-bağlam performansını ölçer. Yedi sentetik görev (retrieval + aggregation, klasik NIAH'ın "iğne bağlamda olmayabilir" varyantı dahil), önce İngilizce olarak yazılmış, sonra anadili konuşucularla birlikte 25 dile çevrilmiştir.

### Temel bulgular

- Bağlam 8K'dan 128K token'a çıktıkça, **düşük kaynaklı diller** ile **yüksek kaynaklı diller** arasındaki performans farkı genişliyor.
- Şaşırtıcı biçimde **İngilizce en iyi performans gösteren dil değil** — 26 dil arasında 6. sırada; en iyi performansı **Lehçe (Polish)** gösteriyor.
- Talimat (instruction) ve bağlamın **farklı dillerde** olduğu senaryolarda (cross-lingual), talimatın hangi dilde verildiğine bağlı olarak performans **%20'ye varan** dalgalanma gösterebiliyor.

- Kaynak: [One ruler to measure them all: Benchmarking multilingual long-context language models (arXiv:2503.01996)](https://arxiv.org/abs/2503.01996)

> Not: OneRuler henüz Türkçe için ayrıntılı, ayrı raporlanmış bir skor içermiyor (26 dilden biri olarak dahil edilmiş olabilir, ancak makalede dil-bazlı kırılım tüm diller için ayrı ayrı raporlanmıyor) — Türkçeye özel uzun-bağlam ölçümü için bkz. `07_cok_dilli_benchmarklar.md` dosyasındaki Türkçe bölümü.

---

## 6. "Context Rot" — bağlam uzadıkça neden akıl yürütme bozuluyor?

2025 ortasında Chroma tarafından yayımlanan **"Context Rot: How Increasing Input Tokens Impacts LLM Performance"** raporu, RULER ve NIAH'ın ötesine geçerek şunu net biçimde gösterdi: **iğne bulunabilir olsa bile**, salt bağlam uzunluğu arttıkça model performansı düşüyor — yani sorun sadece "aramak/bulmak" değil, bağlamın kendisinin model üzerinde yarattığı yük.

### Anahtar bulgular

- 18 güncel model test edildi (GPT-4.1, Claude 4 ailesi, Gemini 2.5, Qwen3 dahil).
- Performans **doğrusal olarak değil, düzensiz biçimde** bozuluyor: 200K'lık bir pencere için bile 50K token civarında ciddi doğruluk kaybı görülebiliyor; 1M'lik pencere iddia edilen 1M token boyunca güvenilir akıl yürütme sağlamıyor.
- İğne ile soru **anlamca** benzer ama **sözcük olarak** farklıysa (lexical eşleşme yoksa), bozulma daha hızlı gerçekleşiyor.
- Tek bir **distractor** (dikkat dağıtıcı, alakasız ama benzer görünen metin parçası) bile doğruluğu düşürüyor; bu etki modelden modele farklı büyüklükte.
- Modeller, bağlam uzadıkça basit **metni birebir kopyalama** (verbatim text replication) görevinde bile başarısız olmaya başlıyor.

### Mekanizmalar

1. **Lost-in-the-middle etkisi** — modeller bağlamın başına ve sonuna daha iyi "dikkat ediyor", ortasına daha zayıf.
2. **Dikkat seyrelmesi (attention dilution)** — transformer dikkat mekanizması ikinci dereceden (quadratic) olduğundan, token sayısı arttıkça milyarlarca ikili (pairwise) ilişki oluşuyor ve her bir token'a düşen "dikkat payı" küçülüyor.
3. **Distractor girişimi (distractor interference)** — anlamca benzer ama alakasız içerik, modeli aktif olarak yanlış yöne itiyor.

- Kaynak: [Context Rot: How Increasing Input Tokens Impacts LLM Performance (Chroma, 14 Temmuz 2025)](https://www.beri.net/learning/chroma-context-rot-report), [Context Rot, RAG, and Long Context: How to Architect LLM Systems in 2026](https://glasp.co/articles/context-rot-rag-long-context-hybrid)

**Beklenen cevap** (context-rot testinden basitleştirilmiş örnek — 300 kelimelik bir metni birebir tekrar ettirme görevi):
> Kaynak metnin birebir aynısı, tek bir karakter hatası olmadan.

**Model çıktısı (örnek/illustrative):** Bağlam 4K token iken model metni kusursuz kopyalıyor; bağlam 100K token'a çıkıp hedef metin ortalara gömüldüğünde model bazı cümleleri atlıyor, parafraze ediyor veya cümle sırasını karıştırıyor — doğru bilgi teknik olarak "bulunabilir" olsa da, üretim kalitesi bozuluyor.

### Michelangelo (Google DeepMind) — "haystack ötesi" bir yaklaşım

Context rot problemine paralel olarak Google DeepMind, klasik NIAH'ın yetersizliğine karşı **Michelangelo** benchmarkını yayımladı. Bu benchmark, "gizli yapı sorguları" (Latent Structure Queries) fikrine dayanır: model, bağlamdaki alakasız bilgiyi ayıklayarak (bir heykeltıraşın mermerden fazlalığı yontması gibi) altında yatan gizli bir yapıyı ortaya çıkarmalı, sonra bu yapı hakkında sorgulanmalıdır. Üç çekirdek görevi vardır:

- **Latent List** — bir Python listesi üzerinde yapılan uzun bir işlem dizisini takip edip listenin son halini çıkarma.
- **MRCR (Multi-round Co-reference Resolution)** — uzun bir kullanıcı-model diyaloğunun önceki turlarına yapılan referansları çözme.
- **IDK (I Don't Know)** — bağlamda cevap olmayan sorularda "bilmiyorum" diyebilme (halüsinasyon yerine).

- Kaynak: [Michelangelo: Long Context Evaluations Beyond Haystacks via Latent Structure Queries (arXiv:2409.12640)](https://arxiv.org/html/2409.12640v2), [DeepMind Michelangelo Benchmark](https://deepmind.google/research/publications/michelangelo-long-context-evaluations-beyond-haystacks-via-latent-structure-queries/)

---

## 7. ∞Bench (InfiniteBench) — 100K token sınırının ötesi

**∞Bench (InfiniteBench)**, "100K+ token bağlamları işleyebilir" iddia eden modelleri değerlendirmek için tasarlanmış, İngilizce ve Çince'de **12 farklı görev** içeren bir benchmarktır. LongBench gibi öncül benchmarkların ~10K token civarında sıkışıp kalmasına karşı geliştirilmiştir.

- Görevler, salt sınırlı sayıda pasajı bağlamdan çekmenin (retrieval) yeterli olmadığı, **uzun bağımlılıkları anlamayı** gerektiren senaryolar içerir (roman özetleme, kod hata ayıklama, matematiksel bulmaca çözme, çoklu belge QA vb.).
- Deneysel sonuçlar, mevcut uzun-bağlam LLM'lerin 100K+ bağlamları etkin işlemek için hâlâ önemli gelişime ihtiyaç duyduğunu gösterdi.

- Kaynak: [∞Bench: Extending Long Context Evaluation Beyond 100K Tokens (ACL 2024)](https://aclanthology.org/2024.acl-long.814/), [GitHub - OpenBMB/InfiniteBench](https://github.com/OpenBMB/InfiniteBench)

---

## 8. BABILong — milyonlarca token'a kadar akıl yürütme

**BABILong**, Facebook'un eski bAbI görev setini (basit mantıksal çıkarım görevleri: gerçek zinciri, sayma, tümevarım, tümdengelim, liste/küme işleme) alıp bu "iğneleri" PG-19 kitap veri kümesinden alınan uzun, alakasız metinlerin içine gömerek son derece uzun bağlamlarda **akıl yürütmeyi** test eder.

### Ölçek

| Özellik | Değer |
|---|---|
| Görev sayısı | 20 farklı akıl yürütme görevi (gerçek zincirleme, tümevarım, tümdengelim, sayma, liste/küme işleme vb.) |
| Bağlam uzunluğu | Binlerce token'dan **50 milyon token'a** kadar ölçeklenebilir; kamuya açık bölünmüş (split) veri setleri 10 milyon token'a kadar sağlanır |
| Gürültü kaynağı | PG-19 (halka açık kitaplar) |

### Kritik bulgu

Popüler LLM'ler bağlamın yalnızca **%10-20'sini** etkin biçimde kullanabiliyor ve akıl yürütme karmaşıklığı arttıkça performans keskin biçimde düşüyor. Bağlam uzatma yöntemleri arasında, ince ayar (fine-tuning) sonrası en iyi performansı **rekürren bellek transformerleri (recurrent memory transformers)** gösteriyor — bu mimariler 50 milyon token'a kadar işlem yapabiliyor.

- Kaynak: [BABILong: Testing the Limits of LLMs with Long Context Reasoning-in-a-Haystack (NeurIPS 2024, arXiv:2406.10149)](https://arxiv.org/abs/2406.10149), [GitHub - booydar/babilong](https://github.com/booydar/babilong)

**Beklenen cevap** (BABILong "gerçek zincirleme" görevinden basitleştirilmiş örnek):
> Soru: "Elma nerede?" — Bağlamda milyonlarca token'lık alakasız kitap metni arasına gömülü üç cümle: "Ayşe mutfaktaydı. Ayşe elmayı aldı. Ayşe bahçeye gitti." → Doğru cevap: "Bahçede."

**Model çıktısı (örnek/illustrative):** Kısa bağlamda (birkaç bin token) neredeyse tüm modeller bu zinciri doğru takip eder; bağlam birkaç milyon token'a çıktığında birçok model ya "mutfakta" (ilk konumu, güncellenmemiş halini) söyler ya da alakasız kitap metninden bir yer adı "halüsinasyon" eder.

---

## 9. FRAMES (Google) — Factuality, Retrieval, Reasoning bir arada

**FRAMES (Factuality, Retrieval, and Reasoning MEasurement Set)**, Google tarafından yayımlanan ve RAG sistemlerini **uçtan uca** (retrieval + factuality + reasoning ayrı ayrı değil, birlikte) değerlendiren bir veri kümesidir.

### Yapı

| Özellik | Değer |
|---|---|
| Soru sayısı | 824 zorlu çok-adımlı (multi-hop) soru |
| Konu çeşitliliği | Tarih, spor, bilim, sağlık vb. |
| Gerekli kaynak sayısı | Her soru, **2 ile 15 arası** Wikipedia makalesinden bilginin birleştirilmesini gerektirir |
| Akıl yürütme etiketleri | Sayısal (numerical), tablo (table), çoklu kısıt (multiple constraints), zamansal (temporal), son-işleme (post-processing) |

Veri kümesinin **~%36'sı** çoklu kısıt (multiple constraints) içeren akıl yürütme, **~%20'si** sayısal karşılaştırma gerektiriyor.

### Neden önemli?

FRAMES'in özgün katkısı, retrieval kalitesi ile üretim (generation) kalitesini **ayrı ayrı değil birlikte** ölçmesidir — gerçek dünya RAG hatalarının çoğu, bu iki bileşen arasındaki etkileşimden kaynaklanır. Tek adımlı (single-step: doğrudan sorgula-cevapla) yaklaşımlar ~0,40 doğruluk gösterirken, çok adımlı (multi-step: sorguyu ayrıştır, birden çok kez ara, sonra sentezle) yaklaşımlar ~0,66'ya çıkıyor.

- Kaynak: [Google Releases FRAMES (MarkTechPost)](https://www.marktechpost.com/2024/10/01/google-releases-frames-a-comprehensive-evaluation-dataset-designed-to-test-retrieval-augmented-generation-rag-applications-on-factuality-retrieval-accuracy-and-reasoning/), [google/frames-benchmark (Hugging Face)](https://huggingface.co/datasets/google/frames-benchmark)

---

## 10. RAGBench ve TRACe değerlendirme çerçevesi

### RAGBench

**RAGBench**, Galileo Technologies tarafından yayımlanan, **~100.000 örnek** içeren büyük ölçekli bir RAG değerlendirme veri kümesidir. 12 açık-kitap (open-book) QA veri kümesinden, 5 farklı endüstri alanından (biyomedikal, hukuk, müşteri desteği, finans, genel bilgi vb.) derlenmiştir.

### TRACe çerçevesi — neden gerekli?

Geleneksel RAG değerlendirmesi genelde tek bir "doğru/yanlış" veya "faithfulness skoru" ile yapılır — ama bu, hatanın **hangi bileşenden** (retriever mi, generator mi) kaynaklandığını ayırt etmez. TRACe, dört ayrı metrik tanımlayarak bu sorunu çözer:

| Metrik | Ne ölçer | Hangi bileşeni test eder |
|---|---|---|
| **R**elevance (İlgililik) | Getirilen (retrieved) pasajların soruyla ne kadar alakalı olduğu | Retriever |
| **U**tilization (Kullanım) | Getirilen bağlamın ne kadarının üretilen cevapta gerçekten kullanıldığı | Generator |
| **C**ompleteness (Bütünlük) | Üretilen cevabın, bağlamdaki tüm ilgili/gerekli bilgiyi kapsayıp kapsamadığı | Generator |
| **A**dherence (Bağlılık / "e" harfi kelimeyi tamamlamak için) | Cevabın verilen bağlama sadık kalıp kalmadığı — "faithfulness", "groundedness" veya "attribution" ile eş anlamlı | Generator (halüsinasyon tespiti) |

Bu dört metriğin ilk harfleri (Utilization, Relevance, Adherence, Completeness) bir araya gelerek **TRACe** kısaltmasını oluşturur.

Matematiksel olarak Adherence (bağlılık) kabaca şu şekilde formüle edilebilir (illüstratif):

\[
\text{Adherence}(a, C) = \frac{|\{\text{a içindeki iddialar} \mid \text{C tarafından desteklenen}\}|}{|\{\text{a içindeki tüm iddialar}\}|}
\]

burada \(a\) üretilen cevap, \(C\) getirilen bağlamdır (retrieved context).

Benzer biçimde Completeness (bütünlük):

\[
\text{Completeness}(a, C^{*}) = \frac{|\{\text{C* içindeki gerekli bilgiler} \mid \text{a içinde yer alan}\}|}{|\{\text{C* içindeki tüm gerekli bilgiler}\}|}
\]

burada \(C^{*} \subseteq C\), soruyu cevaplamak için gerçekten gerekli olan bağlam alt kümesidir.

### Dikkat çekici sonuç

RAGBench üzerinde ince ayar yapılmış **400M parametrelik bir DeBERTa modeli**, halüsinasyon tespiti ve bağlam kullanımı ölçümünde çok daha büyük LLM'leri "hakem" (judge) olarak kullanan yaklaşımlardan **daha iyi** performans gösterdi — bu da özel eğitim verisinin (specialized training data) genel amaçlı büyük modellerden daha değerli olabileceğini gösteriyor.

- Kaynak: [RAGBench: Explainable Benchmark for Retrieval-Augmented Generation Systems (arXiv:2407.11005)](https://arxiv.org/abs/2407.11005)

**Beklenen cevap** (RAGBench tarzı bir örnek — finans alanından):
> Soru: "Şirketin 2024 yılı net kârı bir önceki yıla göre ne kadar arttı?"
> Bağlam: "...2023 net kâr: 4,2 milyar TL... 2024 net kâr: 5,1 milyar TL..."
> Beklenen cevap: "Net kâr 0,9 milyar TL (yaklaşık %21) arttı."

**Model çıktısı (örnek/illustrative):** Bir model doğru sayıları bulup çıkarma işlemini doğru yapmayabilir (düşük **Utilization** — bağlamı buldu ama doğru kullanmadı) ya da yüzde hesaplamasını atlayıp sadece mutlak farkı verebilir (düşük **Completeness**).

---

## 11. T²-RAGBench — finans, metin ve tablo

**T²-RAGBench (Text-and-Table Benchmark)**, finansal raporlardaki **metin + tablo** karışık verisi üzerinde RAG sistemlerini değerlendirmek için tasarlanmıştır. Hamburg Üniversitesi tarafından geliştirilmiştir.

### Yapı

| Özellik | Değer |
|---|---|
| Soru-bağlam-cevap üçlüsü | ~23.088 – 32.908 arası (versiyona göre değişir; ana makalede 23.088 rapor edilir) |
| Kaynak finansal rapor sayısı | 7.318 gerçek finansal rapor |
| Temel veri kümeleri | FinQA, ConvFinQA, TAT-DQA |
| Odak | Sayısal akıl yürütme (numerical reasoning) + retrieval sağlamlığı |

### Neden özel bir benchmark gerekiyor?

Çoğu klasik finansal QA veri kümesi, "oracle context" varsayımıyla çalışır — yani doğru bağlam parçası modele **zaten verilmiştir**, model sadece o parçadan cevabı çıkarır. Ama gerçek bir RAG sisteminde model önce doğru pasajı/tabloyu **kendi bulmalı**, sonra üzerinde sayısal işlem yapmalıdır. T²-RAGBench, veri kümesini "bağlamdan bağımsız" (context-independent) formata dönüştürerek bu daha gerçekçi ve zor senaryoyu test eder.

- Kaynak: [T²-RAGBench: Text-and-Table Benchmark for Evaluating Retrieval-Augmented Generation (arXiv:2506.12071)](https://arxiv.org/html/2506.12071v1), [Hugging Face - G4KMU/t2-ragbench](https://huggingface.co/datasets/G4KMU/t2-ragbench)

---

## 12. Karşılaştırma tablosu

| Benchmark | Maks. test edilen bağlam uzunluğu | Görev türü | Diller |
|---|---|---|---|
| LongBench | ~20K token (ort. 6,7K kelime EN / 13,4K karakter ZH) | Retrieval + özetleme + few-shot + kod | İngilizce, Çince |
| LongBench v2 | 2.000.000 kelime | Çok-adımlı (multi-hop) derin anlama ve akıl yürütme | İngilizce (ağırlıklı), Çince |
| Needle-in-a-Haystack (klasik) | Model bağlam limiti kadar (tipik 128K-1M) | Tekli retrieval | Dil-bağımsız (genelde İngilizce) |
| Multi-Needle NIAH | Model bağlam limiti kadar | Çoklu retrieval | Dil-bağımsız |
| RULER | 128K (standart), bazı çalışmalarda daha fazla | Retrieval, multi-hop tracing, aggregation, QA | Öncelikle İngilizce |
| OneRuler | 128K | Retrieval + aggregation (çok dilli) | 26 dil |
| Context Rot (Chroma) | 1.000.000 token | Retrieval + literal metin kopyalama | İngilizce |
| Michelangelo | Arbitrer (test-zamanı ölçeklenebilir) | Latent yapı çıkarımı, coreference, IDK | İngilizce |
| ∞Bench (InfiniteBench) | 100K+ token | Özetleme, kod hata ayıklama, çoklu-belge QA, matematik | İngilizce, Çince |
| BABILong | 50.000.000 token | 20 mantıksal akıl yürütme görevi (zincirleme, sayma, tümevarım/tümdengelim) | İngilizce (bAbI kaynaklı) |
| FRAMES | Değişken (2-15 Wikipedia makalesi birleştirme) | Uçtan uca factuality + retrieval + multi-hop reasoning | İngilizce |
| RAGBench (TRACe) | Değişken (doküman-bazlı, uzun bağlam odaklı değil) | RAG bileşen-bazlı değerlendirme (relevance/utilization/completeness/adherence) | İngilizce |
| T²-RAGBench | Değişken (finansal rapor uzunluğu) | Metin+tablo retrieval, sayısal akıl yürütme | İngilizce (finans) |

---

## 13. Pratik çıkarımlar: hangi benchmark ne zaman kullanılmalı?

1. **Yeni bir modelin "context window" iddiasını doğrulamak** için → RULER veya Context Rot metodolojisi (tek NIAH testi yetersiz — mutlaka multi-needle/aggregation içeren bir set kullanın).
2. **Bir RAG pipeline'ının hangi bileşeninin (retriever/generator) hataya sebep olduğunu ayırt etmek** için → RAGBench + TRACe (Relevance retriever hatasını, Utilization/Completeness/Adherence generator hatasını izole eder).
3. **Çok dilli / Türkçe içerik üreten bir sistemin uzun-bağlamda dil-bazlı performans farkını ölçmek** için → OneRuler (26 dil) — bkz. `07_cok_dilli_benchmarklar.md`.
4. **Finansal/sayısal doküman analizi yapan bir asistanı test etmek** için → T²-RAGBench veya FRAMES (tablo + sayısal akıl yürütme ağırlıklı).
5. **"Modelim gerçekten milyonlarca token'ı kullanabiliyor mu?" sorusunu uç noktada test etmek** için → BABILong (10M-50M token'a kadar ölçekli split'ler).
6. **Modelin "bilmiyorum" diyebilme (aşırı-güven/halüsinasyon karşıtı) becerisini** test etmek için → Michelangelo'nun IDK görevi veya NIAH'ın negatif-örnek varyantları.

---

## 14. Toplu skor hesaplama: "etkin bağlam uzunluğu" nasıl tanımlanır?

RULER makalesi ve onu izleyen çalışmalar, ham bir "ortalama doğruluk" yerine **etkin bağlam uzunluğü (effective context length)** kavramını popülerleştirdi: bir modelin, belirli bir eşik doğruluğun (ör. %85) altına düşmeden işleyebildiği en uzun bağlam.

\[
L_{\text{eff}} = \max \{ L \mid \text{Acc}_{\text{ortalama}}(L) \geq \tau \}
\]

burada \(\text{Acc}_{\text{ortalama}}(L)\), bağlam uzunluğu \(L\) için tüm görevlerin (RULER'da 13 görev) ortalama doğruluğu, \(\tau\) ise kabul edilen eşik değeridir (yaygın olarak \(\tau = 0{,}85\) veya modelin kısa-bağlam performansının belirli bir yüzdesi kullanılır).

Bu tanım, pazarlama materyallerindeki "128K context window" gibi **nominal** (iddia edilen) değerler ile modelin gerçekte güvenilir çalıştığı **etkin** değer arasındaki farkı somutlaştırır. Örneğin bir model nominal 128K desteklese bile \(L_{\text{eff}} = 32K\) çıkabilir — yani 32K'nın ötesinde doğruluk kabul edilebilir eşiğin altına düşüyor demektir.

### Ağırlıklı ortalama vs. görev-bazlı raporlama

Çoğu uzun-bağlam benchmarkı iki şekilde skor raporlar:

1. **Makro ortalama** — her görevin ağırlığı eşit: \(\text{Skor} = \frac{1}{K}\sum_{k=1}^{K} \text{Acc}_k\) (K = görev sayısı). Bu yöntem, az sayıda örneğe sahip görevlerin skorunu abartılı biçimde etkileyebilir.
2. **Bağlam-uzunluğu bazlı kırılım** — her bağlam uzunluğu (8K, 16K, 32K, 64K, 128K...) için ayrı skor raporlanır ve bir eğri (curve) çizilir; bu, "modelin nerede çökmeye başladığını" görsel olarak ortaya koyar ve genelde tek sayıdan daha bilgilendiricidir.

Pratikte bir model karşılaştırması yaparken **sadece tek bir ortalama sayıya** güvenmemek, mutlaka bağlam-uzunluğu bazlı eğriye bakmak önerilir — çünkü iki model aynı ortalamaya sahip olsa bile biri kısa bağlamda güçlü/uzun bağlamda zayıf, diğeri tam tersi bir profil sergileyebilir.

---

## 15. Kendi uzun-bağlam/RAG testinizi kurmak — pratik kontrol listesi

Hazır benchmarkların hiçbiri kendi veri kümenizin/kullanım senaryonuzun (ör. Türkçe hukuk dokümanları, Türkçe müşteri destek transkriptleri) özelliklerini birebir yansıtmayabilir. Bu durumda kendi mini-benchmarkınızı kurarken aşağıdaki ilkeler işe yarar (RULER, Context Rot ve RAGBench metodolojilerinden damıtılmıştır):

1. **Sadece tek-iğne testiyle yetinmeyin.** En az bir multi-needle ve bir aggregation görevi ekleyin — aksi halde context rot'u kaçırırsınız.
2. **Negatif örnekler ekleyin** (cevabın bağlamda olmadığı durumlar) — modelin halüsinasyon eğilimini ölçmeden "başarı" ilan etmeyin.
3. **Bağlam uzunluğunu kademeli artırın** (ör. 2K, 8K, 32K, 128K) ve her adımda aynı görev tipini tekrarlayın — tek bir uzunlukta test etmek "çökme noktasını" (breaking point) gizler.
4. **Distractor (dikkat dağıtıcı) içerik ekleyin** — rastgele gürültü yerine, sorguyla *anlamca* benzer ama yanlış bilgi içeren pasajlar kullanın; Context Rot raporu bunun gerçek dünya hata oranını daha iyi yansıttığını gösterdi.
5. **RAG sistemi test ediyorsanız retriever ve generator hatalarını ayrıştırın** — TRACe çerçevesindeki Relevance/Utilization/Completeness/Adherence ayrımını kendi metrik setinize uyarlayın.

```python
# Basitleştirilmiş, illüstratif bir "mini-RULER" test iskeleti
import random

def build_haystack(needle: str, distractors: list[str], target_tokens: int, depth_pct: float) -> str:
    """Bağlama tek bir iğne + N distractor yerleştirir, depth_pct konumunda."""
    filler = generate_filler_text(target_tokens)  # PG-19 benzeri nötr metin
    insert_pos = int(len(filler) * depth_pct)
    context = filler[:insert_pos] + " " + needle + " " + filler[insert_pos:]
    for d in distractors:
        pos = random.randint(0, len(context))
        context = context[:pos] + " " + d + " " + context[pos:]
    return context

def score_run(model, needle_value: str, context: str, question: str) -> bool:
    answer = model.generate(context=context, question=question)
    return needle_value.lower() in answer.lower()

# Bağlam uzunluğu x derinlik x distractor sayısı ızgarasında tara
results = {}
for L in [2_000, 8_000, 32_000, 128_000]:
    for depth in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for n_distractors in [0, 1, 5]:
            ctx = build_haystack(needle="Gizli sayı 7492'dir.",
                                  distractors=make_distractors(n_distractors),
                                  target_tokens=L, depth_pct=depth)
            results[(L, depth, n_distractors)] = score_run(
                model, "7492", ctx, "Gizli sayı nedir?"
            )
```

**Beklenen cevap** (bu iskelet örneğinde): `"7492"` — modelin cevabında bu değerin geçip geçmediği kontrol edilir.

**Model çıktısı (örnek/illustrative):** `n_distractors=0` iken doğruluk genelde tüm derinliklerde yüksek kalırken, `n_distractors=5` ve `L=128_000` kombinasyonunda doğruluk belirgin biçimde düşer — bu davranış, gerçek dünya RAG sistemlerinde birden fazla alakalı-görünen ama yanlış pasajın getirildiği (retrieval gürültüsü) senaryoyu simüle eder.

---

## 16. Sık yapılan yorumlama hataları

1. **"Modelim NIAH'ta %100 aldı, demek ki 1M bağlamı sorunsuz kullanabiliyor."** — Yanlış; NIAH tek başına context rot'u, multi-hop akıl yürütmeyi veya distractor dayanıklılığını ölçmez (bkz. §3, §6).
2. **"RAGBench/TRACe skorları yüksekse retrieval sistemim mükemmeldir."** — TRACe'in Relevance metriği retrieval kalitesini generator'dan ayırsa da, nihai uygulamanızın gerçek sorgu dağılımı RAGBench'in kaynak alanlarından (biyomedikal, hukuk vb.) farklı olabilir; alan-uyarlaması (domain adaptation) olmadan doğrudan aktarılabilirlik varsayılmamalı.
3. **"Bağlam penceresi ne kadar büyükse o kadar iyi."** — Context Rot bulguları, gereksiz yere uzun bağlam vermenin (alakalı bilgi kısa bir pasajda bulunabiliyorken bile) performansı **düşürebileceğini** gösteriyor; bu yüzden "mümkün olduğunca fazla bağlam ver" sezgisel yaklaşımı yanlış olabilir — iyi bir retrieval/reranking adımı çoğu zaman ham bağlam uzunluğunu artırmaktan daha değerlidir.
4. **"Sentetik benchmark sonucu = gerçek dünya performansı."** — RULER, BABILong, NIAH gibi sentetik testler kontrollü ve tekrarlanabilir olsa da, gerçek dokümanların yapısal düzensizliğini (tablolar, başlıklar, çok yazarlı içerik) tam yansıtmaz; FRAMES ve T²-RAGBench gibi gerçek-dünya kaynaklı benchmarklarla tamamlanmalıdır.

---

## 17. Kaynakça

- [LongBench (arXiv:2308.14508)](https://arxiv.org/abs/2308.14508)
- [LongBench v2 (arXiv:2412.15204)](https://arxiv.org/abs/2412.15204)
- [Needle in a Haystack — Medium açıklaması](https://medium.com/@imrohitkushwaha2001/needle-in-a-haystack-evaluating-llm-performance-in-long-context-retrieval-99bf2887d974)
- [Multi Needle in a Haystack (LangChain)](https://blog.langchain.com/multi-needle-in-a-haystack/)
- [RULER (GitHub, NVIDIA)](https://github.com/NVIDIA/RULER)
- [RULER Leaderboard](https://llm-stats.com/benchmarks/ruler)
- [OneRuler (arXiv:2503.01996)](https://arxiv.org/abs/2503.01996)
- [Context Rot Raporu (Chroma)](https://www.beri.net/learning/chroma-context-rot-report)
- [Michelangelo (arXiv:2409.12640)](https://arxiv.org/html/2409.12640v2)
- [∞Bench / InfiniteBench (ACL 2024)](https://aclanthology.org/2024.acl-long.814/)
- [BABILong (arXiv:2406.10149)](https://arxiv.org/abs/2406.10149)
- [FRAMES (Google, MarkTechPost duyurusu)](https://www.marktechpost.com/2024/10/01/google-releases-frames-a-comprehensive-evaluation-dataset-designed-to-test-retrieval-augmented-generation-rag-applications-on-factuality-retrieval-accuracy-and-reasoning/)
- [RAGBench / TRACe (arXiv:2407.11005)](https://arxiv.org/abs/2407.11005)
- [T²-RAGBench (arXiv:2506.12071)](https://arxiv.org/html/2506.12071v1)

---

*Sonraki dosya: [`07_cok_dilli_benchmarklar.md`](./07_cok_dilli_benchmarklar.md) — çok dilli ve Türkçe LLM değerlendirme kaynakları.*
