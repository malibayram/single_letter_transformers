# 8. Embedding Benchmarkları

**Güncelleme tarihi:** 21 Temmuz 2026

Bu dosya, [`llm_benchmarks_guide_2026.md`](../llm_benchmarks_guide_2026.md) rehberinin embedding (metin gömme) modellerine ayrılmış bölümünün genişletilmiş ve güncellenmiş sürümüdür. Üretken dil modellerinin aksine embedding modelleri metin, cümle veya belge üretmez; bir metni **sabit boyutlu bir sayı vektörüne** dönüştürür. Bu vektörler arama (retrieval), kümeleme (clustering), sınıflandırma (classification) ve RAG (retrieval-augmented generation) sistemlerinin temelini oluşturur. Dolayısıyla embedding benchmarkları, üretken metin kalitesini değil, **vektör uzayının anlamsal ilişkileri ne kadar doğru temsil ettiğini** ölçer.

Aşağıdaki örneklerde ana rehberle aynı kural geçerlidir:

- **Beklenen cevap / etiket**, veri kümesindeki gerçek (gold) değerdir.
- **Model çıktısı**, puanlama mekanizmasını göstermek amacıyla oluşturulmuş **örnek/açıklayıcı** bir çıktıdır; gerçek bir modelin ürettiği birebir sayı değildir.
- **"Yayımlanmış sonuç"** ifadesi yalnızca bir makalede, resmî teknik raporda veya doğrulanabilir bir liderlik tablosunda gerçekten raporlanmış bir sayı için kullanılır.
- Liderlik tabloları (leaderboard) sürekli güncellendiğinden, bu dosyadaki anlık skorlar **Temmuz 2026 civarındaki bir görünümü** yansıtır; birkaç ay içinde sıralama değişebilir.

---

## 8.0. Neden embedding modelleri ayrı bir değerlendirme rejimine ihtiyaç duyar?

Bir üretken model "doğru cevabı üretti mi" sorusuna cevap ararken, bir embedding modeli için soru şudur: **"Anlamca yakın iki metin, vektör uzayında da birbirine yakın mı yerleşti?"**

Bu, tek bir sayıyla özetlenemeyecek kadar çok boyutlu bir sorudur, çünkü "yakınlık" bağlama göre değişir:

- Hukuki bir sözleşmede "fesih" ile "sözleşmenin sona ermesi" anlamca yakın olmalıdır (STS görevi).
- Bir arama motorunda kullanıcı sorgusuyla en ilgili belge üstte çıkmalıdır (retrieval görevi).
- Binlerce haber makalesi otomatik olarak konularına göre gruplanmalıdır (clustering görevi).
- İki cümlenin aynı soruyu sorup sormadığı ayırt edilmelidir (pair classification görevi).

Tek bir embedding modeli bu görevlerin hepsinde aynı derecede iyi olmayabilir. Bu yüzden modern embedding değerlendirmesi, MMLU benzeri tek boyutlu bir "doğruluk" yerine **çok görevli bir değerlendirme paketine** (benchmark suite) dayanır. MTEB tam olarak bu ihtiyaçtan doğmuştur.

---

## 8.1. BEIR — MTEB'in retrieval odaklı öncülü

Kronolojik olarak MTEB'den önce gelen ve MTEB'in retrieval bölümünün temelini oluşturan BEIR (**Benchmarking Information Retrieval**), 2021'de yayımlanmış, sıfır-atış (zero-shot) bilgi getirimi için tasarlanmış bir benchmarktır.

### Temel tasarım fikri

BEIR'den önceki retrieval sistemleri genellikle tek bir veri kümesinde (örneğin MS MARCO) eğitilip yine aynı dağılımdaki test setinde değerlendiriliyordu. Bu, modelin gerçekten genelleyebildiğini değil, o veri kümesine ne kadar iyi uyum sağladığını gösteriyordu. BEIR'in cevabı şuydu:

> Bir retrieval modelini, hiç görmediği alanlarda (domain) ve hiç eğitim verisi almadan test et.

### Kapsamı

- **18 veri kümesi**, birbirinden çok farklı 9 görev türünden derlenmiştir: gerçek doğrulama (fact checking), atıf tahmini (citation prediction), yinelenen soru bulma (duplicate question retrieval), argüman getirimi (argument retrieval), haber getirimi (news retrieval), soru-cevap, tweet getirimi, biyomedikal bilgi getirimi ve varlık (entity) getirimi.
- Veri kaynakları arasında MS MARCO, Natural Questions, HotpotQA, FEVER, SciFact, TREC-COVID, Quora, ArguAna, DBPedia gibi çok farklı alanlardan derlenmiş külliyatlar bulunur.
- Değerlendirilen sistem türleri lexical (BM25), sparse, dense, late-interaction (ColBERT tarzı) ve reranker mimarilerini kapsar.

### Temel bulgu

BEIR makalesinin en çok alıntılanan sonucu şudur: klasik, öğrenmesiz **BM25** (anahtar kelime eşleştirmeye dayanan istatistiksel bir yöntem), birçok dense embedding modelinden **sıfır-atış koşulunda daha sağlam** çıkmıştır. Yani bir dense retriever, eğitildiği alanda çok iyi çalışsa bile, hiç görmediği bir alana (örneğin genel bilgiden tıbbi literatüre) geçince BM25'in gerisinde kalabilir.

### Örnek görev biçimi

Sorgu (TREC-COVID benzeri bir alt kümeden):

```text
Koronavirüs kaynaklı akciğer hasarında sitokin fırtınasının rolü nedir?
```

Külliyatta doğru kabul edilen 12 ilgili belge var. Model ilk 10 sonuç içinde bunlardan 6'sını buluyor.

\[
Recall@10=6/12=\%50
\]

nDCG@10 hesaplanırken yalnızca kaç belge bulunduğu değil, bulunanların **hangi sırada** geldiği de tartılır; en ilgili belge birinci sırada çıkarsa puan, onuncu sırada çıkmasından daha yüksek olur.

### MTEB ile ilişkisi

MTEB'in retrieval görev ailesi, büyük ölçüde BEIR'in veri kümelerini ve nDCG@10 protokolünü doğrudan miras alır. Yani MTEB'de "Retrieval" sekmesini incelediğinizde, aslında büyük ölçüde BEIR'i (ve onun sonradan eklenen genişlemelerini) görürsünüz. Bu nedenle BEIR'i "MTEB'in içine gömülü, retrieval'a özel bir alt benchmark" olarak düşünmek doğrudur.

---

## 8.2. MTEB — Massive Text Embedding Benchmark

MTEB, embedding modellerinin sekiz farklı görev ailesinde aynı anda test edilmesini sağlayan, 2022'de yayımlanmış kapsamlı bir değerlendirme paketidir. İlk sürümü **58 veri kümesi** ve **112 dili** kapsar.

### Neden ortaya çıktı?

MTEB'den önce, her yeni embedding modeli makalesi kendi seçtiği birkaç veri kümesinde (genellikle yalnızca STS veya yalnızca bir retrieval veri kümesinde) rapor veriyordu. Bu durum iki soruna yol açıyordu:

1. **Seçici raporlama (cherry-picking):** Bir model kendi güçlü olduğu veri kümelerini öne çıkarıp zayıf olduğu görevleri hiç raporlamayabiliyordu.
2. **Karşılaştırılamazlık:** İki farklı makale iki farklı veri kümesi setinde sonuç verince, "Model A mı Model B mi daha iyi?" sorusuna doğrudan cevap verilemiyordu.

MTEB, bütün modellerin **aynı sabit görev listesinde** değerlendirilmesini zorunlu kılarak bu iki sorunu ortadan kaldırmayı hedefler. Bu, tıpkı üretken modeller dünyasında MMLU'nun oynadığı "ortak zemin" rolüne benzer.

### Sekiz görev ailesi

1. **Classification** — embedding üstüne basit bir sınıflandırıcı (genellikle lojistik regresyon) eğitilir; embedding'in ne kadar ayırt edici olduğu ölçülür.
2. **Clustering** — embedding'ler k-means gibi bir kümeleme algoritmasına verilir; oluşan kümelerin gerçek kategorilerle örtüşmesi ölçülür.
3. **Pair Classification** — iki metnin aynı anlama gelip gelmediği (örn. yinelenen soru, çelişki/gerekçe) ikili olarak sınıflandırılır.
4. **Reranking** — bir sorguya karşı önceden getirilmiş aday belge listesi, embedding benzerliğine göre yeniden sıralanır.
5. **Retrieval** — büyük bir belge havuzundan sorguya en ilgili belgeler bulunur (BEIR'in devamı).
6. **Semantic Textual Similarity (STS)** — iki cümle arasındaki anlamsal benzerlik, insan puanlarıyla karşılaştırılır.
7. **Summarization değerlendirmesi** — bir özetin, referans özetlere embedding benzerliği açısından ne kadar yakın olduğu ölçülür.
8. **Bitext Mining** — iki dilli bir külliyattan, birbirinin çevirisi olan cümle çiftleri bulunur.

### Örnek: STS görevi

Cümle 1:

```text
Çocuk parkta futbol oynuyor.
```

Cümle 2:

```text
Bir çocuk açık alanda top oynuyor.
```

**Beklenen (insan) benzerlik puanı:** 4,5 / 5

**Model çıktısı** (cosine similarity, açıklayıcı örnek):

```text
0,86
```

Tek bir çiftte 0,86 değerinin iyi mi kötü mü olduğuna karar verilemez; binlerce çiftte modelin ürettiği benzerlik sıralamasıyla insan sıralaması arasındaki **Spearman korelasyonu** hesaplanır.

### Örnek: Classification görevi

Görev: Ürün yorumlarını olumlu/olumsuz olarak ayırma.

Metin:

```text
Kargo çok hızlı geldi ama ürün açıklamadaki gibi değildi, iade ettim.
```

**Beklenen etiket:** Olumsuz

Bu metnin embedding vektörü üstünde eğitilmiş basit bir sınıflandırıcı **Model çıktısı** olarak "Olumlu" tahmin ederse, madde puanı 0'dır. 200 yorumdan 164'ü doğru sınıflandırılırsa:

\[
164/200=\%82
\]

### Örnek: Clustering görevi

20 Bilim, Teknoloji, Spor ve Ekonomi haberinden oluşan karışık bir kümeye embedding modeli uygulanır ve k-means ile 4 kümeye ayrılır. Modelin ürettiği kümelerin gerçek kategori etiketleriyle örtüşme derecesi **V-measure** ile ölçülür; 1,0 tam örtüşmeyi, 0 rastgele kümelemeyi ifade eder.

**Model çıktısı (açıklayıcı):**

```text
V-measure: 0,71
```

### Örnek: Pair Classification görevi

Cümle A:

```text
İnternet bankacılığı şifremi nasıl değiştirebilirim?
```

Cümle B:

```text
Mobil uygulamada parolamı sıfırlama adımları nelerdir?
```

**Beklenen etiket:** Aynı niyet (duplicate)

Model, iki cümlenin embedding'leri arasındaki benzerliğe bir eşik (threshold) uygulayarak "aynı/farklı" kararı verir. Bu kararların kalitesi eşiğe göre değişmeyen bir metrikle, **Average Precision (AP)**, özetlenir.

### Örnek: Reranking görevi

Sorgu:

```text
Python'da liste kopyalama yöntemleri
```

İlk aşamada (örneğin BM25 ile) getirilen 5 aday belgenin gerçek ilgi sıralaması:

```text
Gerçek sıra: [3, 1, 4, 2, 5]
```

Embedding modeli belgeleri sorguya benzerliğe göre yeniden sıralar:

```text
Model sırası: [1, 3, 2, 4, 5]
```

Bu iki sıralama arasındaki uyum **Mean Average Precision (MAP)** ile ölçülür; ilgili belgeleri üst sıralara taşıyan model daha yüksek puan alır.

### Örnek: Summarization görevi

Referans özet:

```text
Şirket üçüncü çeyrekte gelirini %12 artırdı, ancak kâr marjı hammadde
maliyetleri nedeniyle daraldı.
```

Model tarafından üretilmiş (veya değerlendirilen) özet:

```text
Üçüncü çeyrekte gelir %12 büyüdü fakat kâr marjı maliyet baskısıyla düştü.
```

Bu görevde embedding modeli özetleri **üretmez**; iki özetin embedding'leri arasındaki benzerlik, referans özetlerle karşılaştırılarak bir "özet kalite puanı" oluşturulur. Amaç, klasik ROUGE gibi yüzeysel kelime örtüşmesi metriklerinin kaçırdığı anlamsal örtüşmeyi yakalamaktır.

### Örnek: Bitext Mining görevi

Türkçe cümle:

```text
Toplantı yarın saat 10'da başlayacak.
```

İngilizce külliyattaki karşılığı:

```text
The meeting will start tomorrow at 10 o'clock.
```

Model, iki dilli bir cümle havuzunda her Türkçe cümle için en yakın embedding'e sahip İngilizce cümleyi bulmaya çalışır. Doğru çeviri çiftini bulma oranı **F1** ile raporlanır.

### Genel skor nasıl hesaplanır?

MTEB'in üstteki "genel ortalama" sütunu, tek bir işlemle değil, **iki katmanlı bir ortalamayla** hesaplanır:

1. Önce her veri kümesi için kendi metriği hesaplanır (STS için Spearman, retrieval için nDCG@10, classification için accuracy, vb.).
2. Sonra bu veri kümesi puanları önce **görev ailesi içinde** ortalanır, ardından **görev aileleri arasında** eşit ağırlıkla ortalanır.

Bu iki katmanlı yapı önemlidir, çünkü örneğin retrieval görev ailesinde 15 veri kümesi, bitext mining'de yalnızca 3 veri kümesi olabilir. Eğer bütün veri kümeleri tek bir düz ortalamaya (mikro ortalama) sokulsaydı, retrieval o kadar baskın çıkardı ki genel skor neredeyse yalnızca retrieval performansını yansıtırdı. İki katmanlı (makro) ortalama, her görev ailesine —içindeki veri kümesi sayısından bağımsız— eşit söz hakkı verir.

**Basitleştirilmiş örnek:**

| Görev ailesi | Veri kümesi sayısı | Görev ailesi ortalaması |
|---|---:|---:|
| Retrieval | 15 | 68 |
| STS | 10 | 82 |
| Classification | 12 | 74 |
| Clustering | 8 | 55 |
| Diğer 4 aile (ortalama) | — | 70 |

\[
\text{Genel skor} = (68+82+74+55+70)/5 = 69{,}8
\]

Bu sayı hesaplandıktan sonra bile, yukarıda anlatıldığı gibi **tek başına yeterli değildir** — hangi görev ailesinin sizin kullanım senaryonuz için önemli olduğuna bakmadan yalnızca bu tek sayıya güvenmek yanıltıcı olur.

### Neden tek bir MTEB ortalama skoru yanıltıcı olabilir?

Bir model:

- STS'de çok iyi,
- Retrieval'da orta,
- Clustering'de zayıf

olabilir. Sekiz görevin ortalamasını almak, bu farkları gizler. RAG sistemi kuran birinin asıl bakması gereken skor **Retrieval** sekmesidir; bir öneri sistemi kuran birinin asıl bakması gereken skor ise **Clustering** veya **Pair Classification** olabilir.

| Görev ailesi | Kullanılan metrik | Örnek kullanım senaryosu |
|---|---|---|
| Classification | Accuracy (embedding üstü basit sınıflandırıcı ile) | Duygu analizi, spam/istenmeyen içerik tespiti |
| Clustering | V-measure | Binlerce haberi otomatik olarak konu başlıklarına ayırma |
| Pair Classification | Average Precision (AP) | İki destek talebinin aynı sorunu anlatıp anlatmadığını bulma |
| Reranking | MAP (Mean Average Precision) | Arama motorunun ilk getirdiği sonuçları yeniden sıralama |
| Retrieval | nDCG@10 | RAG sisteminde soruya en ilgili belge parçalarını bulma |
| STS | Spearman korelasyonu | Anlamsal arama, cümle benzerliği tabanlı öneri |
| Summarization değerlendirmesi | Referans özete embedding benzerliği | Otomatik özetleme sisteminin kalite kontrolü |
| Bitext Mining | F1 | İki dilli paralel çeviri korpusu oluşturma |

---

## 8.3. MMTEB — Massive Multilingual Text Embedding Benchmark

MTEB büyük ölçüde İngilizce merkezli kalmıştı; 112 dil iddiasına rağmen dillerin çoğu az sayıda görevde temsil ediliyordu. **MMTEB**, bu boşluğu kapatmak için 2025 başında yayımlanmış, topluluk odaklı (community-driven) devasa bir genişlemedir.

### Kapsamı

- **500'den fazla kaliteli, denetimli görev**
- **250'den fazla dil**
- Klasik sekiz göreve ek olarak yeni görev türleri: **talimat izleme (instruction following)** değerlendirmesi, **uzun belge getirimi (long-document retrieval)** ve **kod getirimi (code retrieval)**.
- Katkı, tek bir laboratuvardan değil geniş bir açık kaynak topluluğundan gelir; bu da düşük kaynaklı dillerin (Türkçe dâhil birçok orta/düşük kaynaklı dil) daha önce hiç temsil edilmediği görevlerin eklenmesini sağlamıştır.

### Neden "sadece MTEB'in büyütülmüş hâli" değil?

MMTEB yalnızca veri kümesi sayısını artırmaz; değerlendirme protokolünü de sağlamlaştırır. Örneğin:

- Aynı görevin farklı dillerdeki sürümleri birbiriyle karşılaştırılabilir hâle getirilmiştir.
- Talimat izleme görevleri, embedding modelinin yalnızca "ne aradığını" değil, **"nasıl aranmasını istediğini"** (örneğin "yalnızca resmî kaynaklardan" gibi bir yönerge) de ne kadar dikkate aldığını test eder — bu, klasik MTEB'de yoktu.
- Kod getirimi görevleri, bir doğal dil açıklamasından doğru kod parçasını (veya tersini) bulma başarısını ölçer; bu, kod arama motorları ve IDE asistanları için doğrudan ilgilidir.

### Örnek: çok dilli tutarsızlık

Aynı embedding modelinin farklı dillerdeki performansı:

| Görev | Türkçe | İngilizce | Arapça |
|---|---:|---:|---:|
| Retrieval | 62 | 74 | 55 |
| STS | 78 | 81 | 70 |
| Classification | 69 | 76 | 64 |

Diller eşit ağırlıkla ortalanırsa:

\[
Türkçe: (62+78+69)/3=69{,}7
\]

\[
Arapça: (55+70+64)/3=63{,}0
\]

Bu tablo, "çok dilli" olarak pazarlanan bir modelin diller arasında ciddi kalite farkı taşıyabileceğini gösterir. Bir Türkçe RAG sistemi kuran ekip, yalnızca genel MMTEB ortalamasına değil, **Türkçe alt sekmesine** bakmalıdır.

### Sınırlaması

Topluluk katkılı yapısı hem güç hem zayıflıktır: bazı diller hâlâ yalnızca 1-2 küçük veri kümesiyle temsil edilir ve bu, o dildeki skorun istatistiksel güvenilirliğini düşürür. Ayrıca yeni eklenen görevler farklı zamanlarda eklendiği için, "MMTEB ortalaması" zamanla neyi kapsadığına bağlı olarak kayabilir; sürüm numarası mutlaka belirtilmelidir.

---

## 8.4. C-MTEB — dile özel bir MTEB çatallanması örneği

MTEB/MMTEB genel çok dilli kapsam sunsa da, tek bir dile **derinlemesine** odaklanan ayrı benchmark'lar da ortaya çıkmıştır. Bunların en bilinen örneği, BAAI (Beijing Academy of Artificial Intelligence) tarafından geliştirilen **C-MTEB** (Chinese Massive Text Embedding Benchmark)'dir.

### Neden ayrı bir benchmark?

Çince, MTEB'in orijinal 112 dilinden biri olsa da, bu kapsam yalnızca yüzeysel bir kapsamdır: az sayıda veri kümesi, sınırlı alan çeşitliliği. C-MTEB ekibi, Çince için:

- **6 görev ailesi** (retrieval, reranking, STS, classification, pair classification, clustering),
- **~35 test veri kümesi**

içeren, MTEB protokolüyle uyumlu ama Çinceye özgü alanları (haber, e-ticaret, hukuk, tıp, sosyal medya) kapsayan bağımsız bir paket geliştirmiştir. C-MTEB, BGE (BAAI General Embedding) model ailesiyle birlikte yayımlanmış ve BGE modelleri, çıktıkları dönemde önceki Çince embedding modellerini C-MTEB'de **%10'dan fazla** farkla geride bırakmıştır.

### Genel prensip: "dile özel MTEB çatalı" deseni

C-MTEB, tek başına önemli olmasının ötesinde, bir **desen** oluşturmuştur: MTEB'in görev taksonomisini (classification/clustering/retrieval/…) alıp, belirli bir dil veya bölgenin gerçek kullanım alanlarına (yerel haber siteleri, yerel e-ticaret metinleri, yerel hukuk dili) uyarlamak. Bu desenin sonraki örnekleri arasında Fransızca, Korece, Japonca ve İskandinav dilleri için benzer çatallanmalar da ortaya çıkmıştır. MMTEB bu çatallanmaların bir kısmını kendi şemsiyesi altına toplamaya da çalışmaktadır, ancak C-MTEB kendi bağımsız liderlik tablosunu sürdürmeye devam etmektedir.

### Örnek görev (retrieval, Çince e-ticaret alanı)

Sorgu (Çince, açıklayıcı çeviriyle):

```text
"Kablosuz kulaklık, gürültü önleyici, 30 saat pil ömürlü"
```

Ürün kataloğunda 500 ürün arasından en ilgili 10 sonuç istenir. Beklenen ilgili ürün sayısı 6 iken model ilk 10'da 5'ini buluyorsa:

\[
Recall@10=5/6\approx\%83{,}3
\]

### Ne öğretir?

C-MTEB örneği, Türkçe gibi diller için de benzer bir ihtiyacın var olabileceğini gösterir: genel çok dilli MMTEB skoru iyi görünen bir model, Türkçeye özgü alanlarda (örneğin resmî yazışma dili, hukuki Türkçe, e-ticaret Türkçesi) hâlâ zayıf kalabilir. Şu an için MTEB/MMTEB şemsiyesi altında C-MTEB ölçeğinde bağımsız, kapsamlı bir "TR-MTEB" bulunmamaktadır; Türkçe görevler büyük ölçüde MMTEB'in çok dilli veri kümeleri (ör. Belebele, MIRACL, XNLI türevleri) üzerinden temsil edilir.

---

## 8.5. Güncel MTEB liderlik tablosu (Hugging Face) nasıl yapılandırılmıştır?

MTEB'in resmî liderlik tablosu Hugging Face üzerinde (`huggingface.co/spaces/mteb/leaderboard`) barındırılır ve 2026 ortasına doğru önemli bir altyapı güncellemesi geçirmiştir (FastAPI + Svelte tabanlı yeni sürüm). Aylık yaklaşık 500 bin görüntülenmeye ulaşan bu tablo artık şu şekilde yapılandırılmıştır:

### Sekmeler (tabs)

- **Genel / İngilizce (English v2)** sekmesi: klasik MTEB görevlerinin güncellenmiş, doygunlaşmayı azaltmaya çalışan bir sürümü.
- **Çok dilli (Multilingual / MMTEB)** sekmesi: yukarıda anlatılan 250+ dilli genişlemeyi kapsar.
- **Alana özel (domain-specific) filtreler**: hukuk, tıp, kod gibi alanlara göre daraltma imkânı.
- **Model türü filtreleri**: dense embedding, static embedding, geç-etkileşim (late-interaction, ColBERT-tarzı) ve reranker modelleri artık ayrı kategoriler olarak işaretlenir; bunları karıştırıp tek bir sıralamada göstermek artık "elma ile armut" karşılaştırması sayılmaktadır.

### Model boyutu kategorileri

2024-2025 döneminde embedding modelleri hızla büyüdü: bugün liderlik tablosunun tepesinde milyarlarca parametreli, kapalı kaynaklı API modelleri (Gemini Embedding, OpenAI text-embedding, Cohere embed, Voyage) ve büyük açık ağırlıklı modeller (Qwen3-Embedding 8B gibi) yer almaktadır. Bu durum, tek bir "genel sıralama" tablosunun artık pratik bir soruyu — *"benim laptopumda veya tek bir GPU'da çalışacak, ucuz ve hızlı bir embedding modeli hangisi?"* — cevaplamaz hâle gelmesine yol açmıştır.

Bunun çözümü olarak liderlik tablosu, modelleri **parametre sayısına göre büyüklük dilimlerine (size bracket)** ayırıp her dilim için ayrı bir "en iyi model" gösterimi sunar; ayrıca çalışma zamanı (runtime) başına performans analitiği de eklenmiştir. Böylece "küçük, verimli embedding modeli" karşılaştırması, dev API modelleriyle aynı tabloda boğulmak yerine kendi dilim(ler)inde yapılabilir.

### 2026 ortası itibarıyla gözlemlenen genel tablo (anlık görüntü, sürekli değişir)

Aşağıdaki sayılar araştırma sırasında (Temmuz 2026) çeşitli liderlik tablosu takip siteleri ve blog analizlerinden derlenmiştir; resmî bir makalede raporlanmış sabit sonuçlar değildir, bu yüzden "Yayımlanmış sonuç" olarak değil, **liderlik tablosu anlık görüntüsü** olarak sunulmaktadır:

| Model | Kategori | Yaklaşık MTEB ortalaması |
|---|---|---:|
| Google Gemini Embedding 001 | Kapalı kaynak, API | ~68,3 (İngilizce v2) |
| Qwen3-Embedding (8B) | Açık ağırlık, büyük | ~75 (MTEB v2, kendi raporlanan ortalaması) |
| Voyage 3.1 | Kapalı kaynak, API | Gemini'ye yakın, farkı daraltmış |
| BGE-M3 (BAAI) | Açık ağırlık, çok işlevli (dense+sparse+multi-vector) | Üretimde en çok indirilen açık modellerden |
| Jina v4 / v5 | Açık ağırlık, çok modlu destekli | Boyut dilimine göre değişken |

**Önemli uyarı:** MTEB v1 ve MTEB v2 skorları **doğrudan karşılaştırılabilir değildir**, çünkü değerlendirme protokolü ve bazı veri kümeleri değişmiştir. Ayrıca "İngilizce v2" tablosu ile "çok dilli MMTEB" tablosu farklı sıralamalar üretebilir; bir modelin İngilizcede birinci olması, çok dilli sıralamada da birinci olacağı anlamına gelmez.

### Üretimde en çok kullanılan modeller ile "en yüksek skorlu" modeller aynı değildir

RAG framework'lerinin (LangChain, LlamaIndex gibi) telemetri raporlarına göre, üretimde fiilen en çok kullanılan embedding modelleri arasında **BGE-M3**, **Qwen3-Embedding**, OpenAI'nin `text-embedding-3-large` modeli ve Voyage AI'nin `voyage-3` modeli öne çıkmaktadır. Bu, saf liderlik tablosu sıralamasıyla birebir örtüşmez; çünkü üretim kararlarında lisans (açık ağırlık mı, API mi), maliyet, gecikme süresi ve dağıtım kolaylığı da en az ham skor kadar belirleyicidir — tıpkı HELM'in "tek sayıya indirgeme" eleştirisinde olduğu gibi.

---

## 8.6. Küçük ve verimli embedding modelleri: ayrı bir karşılaştırma ekseni

Büyük, milyarlarca parametreli embedding modelleri en yüksek ham skoru alsa da, birçok gerçek uygulama (mobil cihaz, düşük gecikmeli arama, büyük ölçekli toplu vektörleştirme) küçük modelleri tercih eder. Bu yüzden liderlik tablosu artık "boyut başına kalite" (quality-per-parameter) eksenini de öne çıkarır.

### Örnek karşılaştırma (açıklayıcı boyut kategorileri)

| Model sınıfı | Yaklaşık parametre | Kullanım senaryosu |
|---|---:|---|
| Büyük API modeli (örn. Gemini Embedding, Qwen3-Embedding 8B) | 1B+ | En yüksek doğruluk gereken kurumsal arama |
| Orta ölçekli açık model (örn. Jina v5-text-small sınıfı) | ~0,6–0,7B | Dengeli kalite/maliyet, kendi sunucusunda barındırma |
| Küçük/nano model (örn. Jina v5-nano sınıfı, `all-MiniLM-L6-v2` ve benzerleri) | <100M | Mobil, uç cihaz (edge), çok yüksek hacimli ucuz vektörleştirme |

Küçük bir modelin büyük bir modele göre MTEB ortalamasında birkaç puan geride kalması, pratikte önemli olmayabilir; çünkü küçük model 10 kat daha hızlı ve ucuz çalışabilir. Bu yüzden "en iyi embedding modeli hangisi" sorusunun tek bir cevabı yoktur — cevap her zaman **bütçe, gecikme kısıtı ve gereken dil/alan kapsamına** bağlıdır.

---

## 8.7. Türkçe embedding değerlendirmesi için pratik notlar

C-MTEB örneğinin gösterdiği gibi, genel çok dilli bir skor tek başına bir dildeki gerçek kaliteyi garanti etmez. Türkçe için henüz C-MTEB ölçeğinde bağımsız bir paket bulunmasa da, bir embedding modelini Türkçe bir ürün için seçerken şu pratik kontrol listesi izlenebilir:

1. **MMTEB'in çok dilli sekmesinde Türkçe alt kümesine bakın**, genel ortalamaya değil.
2. **Kendi alanınıza yakın küçük bir değerlendirme kümesi oluşturun.** Örneğin bir hukuk asistanı için 50-100 gerçek soru-belge çifti hazırlayıp `Recall@10` ve `nDCG@10` hesaplamak, genel benchmarkın vermediği bir sinyal verir.
3. **Aynı modelin farklı dillerdeki tutarlılığını ölçün.** Aşağıdaki gibi basit bir örnek:

| Görev | Türkçe skor | İngilizce skor | Fark |
|---|---:|---:|---:|
| STS | 74 | 83 | 9 puan |
| Retrieval (nDCG@10) | 58 | 71 | 13 puan |

Retrieval'daki 13 puanlık fark, STS'deki 9 puanlık farktan büyüktür; bu da modelin Türkçe **arama** senaryolarında, Türkçe **benzerlik** senaryolarına göre nispeten daha zayıf kaldığını gösterebilir — RAG sistemi kuran bir ekip için bu, retrieval tarafına ekstra dikkat (örn. yeniden sıralama/reranking katmanı ekleme) gerekebileceği anlamına gelir.

4. **Morfolojik zenginliği göz önünde bulundurun.** Türkçe eklemeli bir dildir; "kitaptan", "kitaplıkta", "kitaplarımızın" gibi çekimlenmiş biçimler, alt-kelime (subword) tokenizasyonuna dayalı embedding modellerinde bazen aynı kökten geldiği hâlde vektör uzayında beklenenden daha uzak düşebilir. Genel MMTEB skorları bu tür dile özgü etkileri ayrıştırmaz; yalnızca nihai görev başarısını gösterir.

---

## 8.8. 2026 ve sonrası için embedding benchmark trendleri

1. **Boyut kategorilerine göre parçalanmış liderlik tabloları kalıcı hâle geliyor.** Tek bir "genel şampiyon" fikri, milyar parametreli API modelleriyle küçük açık modellerin aynı tabloda anlamsızca yarışması nedeniyle terk edilmektedir.
2. **Görev-özel ve alan-özel mini-benchmark'lar yaygınlaşıyor.** C-MTEB deseninin (genel MTEB taksonomisini bir dile/alana uyarlama) yeni dillerde ve yeni sektörlerde (finans, hukuk, sağlık) tekrarlanması bekleniyor.
3. **Talimat izleyen (instruction-following) embedding modelleri için değerlendirme olgunlaşıyor.** MMTEB'in eklediği talimat izleme görevleri, "yalnızca resmî kaynaklardan ara" gibi yönergelerin embedding aşamasında nasıl temsil edildiğini ölçmeye başlıyor; bu, klasik STS/retrieval ayrımının ötesinde yeni bir eksen.
4. **Çok modlu (multimodal) embedding değerlendirmesi büyüyor.** Metin+görüntü birlikte embedding üreten modellerin (Jina v4 ve benzerleri) artmasıyla, MTEB benzeri ama görüntü-metin çiftlerini de kapsayan değerlendirme paketlerine olan ihtiyaç artmaktadır; bu alan henüz MTEB/MMTEB kadar standartlaşmamıştır.
5. **Üretim telemetrisi, liderlik tablosunu tamamlayan ikinci bir sinyal hâline geliyor.** RAG framework'lerinin hangi embedding modellerinin fiilen en çok kullanıldığını raporlaması, saf benchmark skorunun yanında "pratikte kim kazanıyor" sorusuna da cevap vermeye başlıyor.

---

## 8.9. Embedding benchmarklarının ölçemedikleri

- **Chunking (parçalama) stratejisinin etkisi:** MTEB, genellikle önceden parçalanmış, temiz metinlerle çalışır. Gerçek bir RAG sisteminde belgenin nasıl parçalara ayrıldığı, embedding modelinin ham kalitesinden bağımsız olarak sonucu büyük ölçüde etkiler; bu benchmarkta ölçülmez.
- **Güncellik ve tazelik:** Bir embedding modeli, eğitim verisi kesim tarihinden sonra ortaya çıkan yeni terimleri (yeni ürün adları, yeni kısaltmalar) iyi temsil edemeyebilir. Statik benchmark veri kümeleri bunu yakalamaz.
- **Alan kayması (domain shift):** MTEB/MMTEB geniş bir alan yelpazesi sunsa da, çok özel bir kurumsal alanı (örneğin bir bankanın iç mevzuat dili) hiçbir genel benchmark tam temsil edemez.
- **Gömme boyutunun (embedding dimension) pratik maliyeti:** Daha yüksek boyutlu vektörler genelde daha iyi skor alır, fakat depolama ve arama maliyetini de artırır; benchmark tabloları bu ödünleşimi (trade-off) otomatik göstermez, ayrıca okunmalıdır.
- **Çok modlu (multimodal) embedding:** Klasik MTEB metin odaklıdır. Görsel+metin birlikte embedding üreten modeller (Jina v4 gibi) için ayrı, henüz MTEB kadar olgunlaşmamış değerlendirme çerçeveleri gerekir.

---

## Kaynaklar / ek okuma

- MTEB orijinal makalesi ve resmî liderlik tablosu: `huggingface.co/spaces/mteb/leaderboard`
- MMTEB makalesi: arXiv 2502.13595 — *"MMTEB: Massive Multilingual Text Embedding Benchmark"*
- BEIR orijinal makalesi: arXiv 2104.08663 — *"BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models"*
- C-MTEB / C-Pack makalesi: arXiv 2309.07597 — *"C-Pack: Packed Resources For General Chinese Embeddings"* (FlagOpen/FlagEmbedding)
- MTEB liderlik tablosu güncellemeleri ve model karşılaştırma raporları (2026 ortası): çeşitli bağımsız izleme siteleri; skorlar sürekli değiştiğinden bu dosyadaki sayılar anlık görüntü olarak ele alınmalıdır.
