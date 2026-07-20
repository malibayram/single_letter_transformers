# 10. Araç kullanımı ve ajan benchmarkları

**Bu belge**, ana rehberdeki (`llm_benchmarks_guide_2026.md`) 11. bölümün genişletilmiş ve güncellenmiş sürümüdür. Puanlama yöntemlerinin temel tanımları — özellikle **Ajan başarı oranı (Success Rate)** — `00_puanlama_yontemleri.md` dosyasında verilmiştir:

\[
Success\ Rate=\frac{\text{Tamamlanan görev}}{\text{Toplam görev}}
\]

Bu belgede o formülün farklı ortamlarda (fonksiyon çağrısı, tarayıcı, işletim sistemi, terminal, iş simülasyonu) nasıl somutlaştığını inceliyoruz.

Aşağıdaki örneklerde yine aynı kural geçerlidir:

- **Beklenen cevap / beklenen durum**, görev tanımındaki hedef sonuçtur.
- **Model çıktısı**, puanlama mantığını göstermek için kurgulanmış örnek bir ajan davranışıdır.
- **Yayımlanmış sonuç**, ilgili makalede veya resmî blog yazısında gerçekten raporlanmış bir sayıdır.
- **Liderlik tablosu görüntüsü**, üçüncü taraf bir izleme sitesinin (leaderboard aggregator) belirli bir tarihte topladığı, sık değişen ve resmî makale sonucu olmayan bir anlık görüntüdür. Bu tür sayılar hızla eskir ve farklı sitelerde farklı metodolojilerle üretildiğinden birebir karşılaştırılamaz; burada yalnızca büyüklük mertebesi (order of magnitude) hissi vermek için, tarihi belirtilerek kullanılmıştır.

---

## 10.0. Neden ayrı bir kategori?

Bölüm 1–9'daki çoğu benchmark, modelden **tek bir metin çıktısı** üretmesini ister: bir seçenek harfi, bir sayı, bir paragraf. Ajan benchmarkları farklıdır — model:

1. Bir **ortamı** gözlemler (dosya sistemi, tarayıcı DOM'u, ekran görüntüsü, API listesi, sanal şirket bilançosu),
2. Bir **eylem** üretir (fonksiyon çağrısı, tıklama, komut satırı ifadesi, sipariş verme),
3. Ortamdan **geri bildirim** alır,
4. Bu döngüyü hedefe ulaşana, adım sınırına veya hataya kadar tekrarlar.

Bu nedenle ajan benchmarklarında tek bir "doğruluk" sayısı yerine genellikle şu boyutlar birlikte raporlanır:

- Görev başarı oranı,
- Adım/tur sayısı,
- Harcanan token ve maliyet,
- Tehlikeli veya geri alınamaz eylem sayısı,
- **Güvenilirlik (reliability):** Aynı görev birden çok kez denendiğinde her seferinde başarılı mı?

Son madde, aşağıda τ-bench bölümünde ele alınan `pass^k` metriğiyle doğrudan ilgilidir ve klasik `pass@k`'tan (bkz. `00_puanlama_yontemleri.md`, madde 1.4) kavramsal olarak farklıdır.

### `pass@k` ile `pass^k` karışıklığı

- **`pass@k`** (kod benchmarklarında): Modele *k deneme hakkı* verilir, bunlardan **en az biri** geçerse başarılı sayılır. Yaratıcılığı/kapsamı ölçer — "iyi bir çözümü hiç üretebiliyor mu?"
- **`pass^k`** (ajan benchmarklarında, τ-bench'in önerdiği biçimiyle): Aynı görev *k kez bağımsız olarak* denenir, hepsinde **aynı anda** başarılı olması gerekir. Güvenilirliği ölçer — "her seferinde aynı doğru sonucu üretebiliyor mu?"

Bir modelin `pass@1` skoru %90 olsa bile `pass^8` skoru çok daha düşük olabilir; bu, modelin "genelde doğru yapabildiği" ama "her müşteride tutarlı biçimde doğru yapamadığı" anlamına gelir — üretim ortamındaki bir müşteri hizmetleri ajanı için bu fark kritik önemdedir.

---

## 10.1. Ortam türlerine göre genel sınıflandırma

| Ortam türü | Ajan ne yapar? | Tipik geri bildirim | Örnek benchmark |
|---|---|---|---|
| Fonksiyon/araç çağrısı | Yapılandırılmış API çağrısı üretir | Statik: doğru fonksiyon + parametre mi? | BFCL |
| Bilgi toplama + araç | Web arama, dosya okuma, hesaplama | Tek seferlik: son cevap doğru mu? | GAIA, BrowseComp |
| Tarayıcı (web) | Tıklama, yazma, form doldurma | Programatik durum kontrolü | WebArena |
| İşletim sistemi (OS) | Masaüstü uygulamalarını GUI ile kullanma | Dosya/ekran durumu kontrolü | OSWorld |
| Terminal | Kabuk komutları, betikler | Çıktı/dosya sistemi durumu | TUA-Bench |
| Çok turlu diyalog + araç | Kullanıcıyla konuşarak araç çağırma | Politika uyumu + durum kontrolü | τ-bench, τ²-bench |
| ML mühendisliği | Veri hazırlama, model eğitme, deney | Gizli test kümesinde performans | MLE-bench |
| Uzun ufuklu iş simülasyonu | Envanter, fiyatlama, sipariş yönetimi | Zaman içindeki net değer | Vending-Bench |

---

## 10.2. BFCL — Berkeley Function-Calling Leaderboard

BFCL, bir LLM'nin **doğru aracı ve doğru parametreleri** seçme becerisini ölçen, Berkeley'nin Gorilla projesi kapsamında geliştirilen bir benchmarktır.

### Kapsam

- **2.000'den fazla** soru–fonksiyon–cevap örneği,
- Python, Java, JavaScript ve REST API gibi birden fazla dil/protokol,
- Basit (tek fonksiyon), çoklu (birden fazla aday fonksiyon), paralel (aynı anda birden fazla çağrı) ve iç içe (bir çağrının çıktısı diğerinin girdisi) durumlar,
- **"Uygun araç yok" (relevance) tespiti:** Modele sunulan hiçbir fonksiyon isteğe uymuyorsa, model fonksiyon çağırmamalı ve bunu belirtmelidir. Var olmayan bir aracı zorla çağırmak da hata sayılır.

### Sürüm geçmişi

| Sürüm | Eklenen yenilik |
|---|---|
| V1 | Tekli, çoklu ve paralel çağrılar için AST (soyut sözdizimi ağacı) eşleştirmesi |
| V2 | Kurumsal ve topluluk katkılı "canlı" (live) fonksiyonlar; ilgililik (relevance) ve çalıştırılabilirlik testi |
| V3 | Çok turlu ve çok adımlı etkileşimler; turlar arasında durum (state) takibi |
| V4 | Bütünsel ajan değerlendirmesi — web araması dâhil daha geniş görev seti |

### Nasıl çalışır?

Modele bir kullanıcı isteği ve JSON şemasıyla tanımlanmış bir veya birden fazla aday fonksiyon verilir. Model, çağrılacak fonksiyonu ve argümanlarını üretir.

### Örnek

Kullanıcı:

```text
İstanbul için yarının hava tahminini getir.
```

Sunulan araç:

```python
get_weather(city: str, date: str)
```

Beklenen çağrı:

```json
{
  "name": "get_weather",
  "arguments": {
    "city": "Istanbul",
    "date": "2026-07-22"
  }
}
```

Model çıktısı:

```json
{
  "name": "get_weather",
  "arguments": {
    "city": "Ankara",
    "date": "tomorrow"
  }
}
```

Fonksiyon adı doğru olsa da şehir ve tarih biçimi yanlış olduğundan madde puanı **0**'dır.

### "Uygun araç yok" örneği

Kullanıcı:

```text
Bana en sevdiğin şarkıyı söyle.
```

Sunulan araçlar yalnızca `get_weather` ve `send_email` içeriyorsa, doğru davranış hiçbir fonksiyon çağırmamak ve bunun yerine kullanıcıya doğrudan cevap vermektir. Model yine de zorla bir fonksiyon çağırırsa (örneğin `send_email` ile anlamsız bir argüman doldurursa), bu **yanlış pozitif çağrı** olarak işaretlenir ve puan kaybettirir.

### Puanlama bileşenleri

- **AST/yapısal eşleşme:** Üretilen çağrının söz dizimsel yapısı, beklenen çağrının yapısıyla örtüşüyor mu?
- **Çalıştırılabilirlik:** Çağrı gerçekten çalıştırılabiliyor mu (bazı alt kümelerde gerçek API'ler tetiklenir)?
- **Parametre doğruluğu:** Her argüman tek tek doğru mu?
- **İlgililik/relevance:** Model gereksiz yere araç çağırıyor mu veya gerekli olduğunda çağırmıyor mu?

### Liderlik tablosu görüntüsü (Temmuz 2026)

Üçüncü taraf izleme sitelerine göre BFCL v3'te en üst sıradaki modeller birbirine çok yakın puanlar almaktadır (yaklaşık %75–77 aralığında), bu da fonksiyon çağırma yeteneğinin üst düzey modeller arasında büyük ölçüde doygunlaşmaya başladığını düşündürmektedir. BFCL v4, web araması gibi daha bütünsel ajan görevlerini eklediğinden genel skorlar daha düşük (yaklaşık %75'in altına inen) ve modeller arası fark daha belirgin çıkmaktadır. Bu sayılar hızla değiştiğinden, güncel karşılaştırma için resmî `gorilla.cs.berkeley.edu/leaderboard.html` sayfasına bakılması önerilir.

### Ne ölçer / ölçmez?

**Ölçer:** Doğru aracı seçme, parametreleri doğru doldurma, gereksiz çağrıdan kaçınma.

**Ölçmez:** Aracın gerçek dünyada çalıştırıldığında ortaya çıkabilecek yan etkileri (örneğin gerçek bir ödeme API'sini yanlışlıkla iki kez çağırmanın maliyeti), çok uzun ajan zincirlerindeki tutarlılığı.

---

## 10.3. GAIA

GAIA (General AI Assistants), genel amaçlı asistanların **web araştırması, dosya okuma, kod çalıştırma ve görsel anlama** gibi becerileri bir arada kullanmasını gerektiren sorular içerir.

### Nasıl çalışır?

GAIA, soruları üç zorluk seviyesinde sunar:

1. **Seviye 1:** Ek araç gerektirmeyen, ancak modelin bağlamı doğru işlemesini gerektiren sorular.
2. **Seviye 2:** Web araması, kod çalıştırma ve çoklu dosya (PDF, Excel vb.) okuma gibi araçların kullanılmasını zorunlu kılan sorular.
3. **Seviye 3:** Çok adımlı planlama, farklı modalitelerdeki bilgileri birleştirme ve uzun vadeli hedefe ulaşma becerisini sınayan karmaşık ajan görevleri.

Metrik olarak katı bir **Exact Match** kullanılır. Modelden uzun açıklamalar yerine, nihai sonucu net bir sayı, tarih veya isim şeklinde vermesi istenir.

### Gerçek görev biçimine sadık örnek

**Verilen girdi:**

- Bir PDF dosyası (şirketin 2025 yılı finansal raporu),
- Bir Excel tablosu (2025 yılı hisse senedi günlük fiyatları).

**Soru:**

> "Finansal rapordaki Sayfa 14'te yer alan 'Ar-Ge Harcamaları' değerini bulun. Ardından, Excel tablosundaki hisse senedi kapanış fiyatlarının 2025 yılı Temmuz ayındaki ortalamasını hesaplayın. Bu iki değeri birbiriyle çarpın. Sonucu en yakın tam sayıya yuvarlayarak yazın."

**Beklenen cevap:**

```text
348920
```

**Model çıktısı:**

```text
Ar-Ge harcaması 12.000 TL, Temmuz ortalaması ise 29.076 TL olarak hesaplanmıştır. Çarpımları 348.912 yapmaktadır.
```

**Puan:** 0 (Çünkü Exact Match normalize edilse dahi yanlış hesaplama veya fazladan açıklama puan kaybına neden olabilir).

### Zorluk seviyelerine göre örnek dağılım

| Seviye | Gerekli beceri | Tipik puan aralığı (kavramsal) |
|---|---|---|
| 1 | Bağlam okuma, basit çıkarım | Görece yüksek |
| 2 | Araç kullanımı, çoklu belge | Orta |
| 3 | Çok adımlı planlama, modalite birleştirme | Görece düşük |

### Liderlik tablosu görüntüsü (Temmuz 2026)

Üçüncü taraf izleme sitelerinde GAIA sonuçları arasında dikkat çekici bir tutarsızlık gözlemlenmektedir: yalın bir temel modelin (araçsız veya minimal araçlı) skoru %30–45 aralığında raporlanırken, arama motoru, kod yorumlayıcısı ve çoklu adım planlama ile donatılmış tam **ajan sistemleri** aynı 466 görevlik test kümesinde %90'ın üzerine çıkabilmektedir. Bu, aynı benchmarkın "çıplak model" ve "model + araç iskelesi (scaffolding)" biçiminde ölçüldüğünde tamamen farklı sayılar verebileceğinin somut bir kanıtıdır — GAIA skoru raporlanırken hangi kurulumun kullanıldığı mutlaka belirtilmelidir.

### Ne ölçer?

- Gerçek dünya araçlarını (tarayıcı, Python yorumlayıcısı) koordine etme yeteneği,
- Halüsinasyon yapmadan tam olarak talep edilen veriyi bulup çıkarma,
- Multimodal belgeleri (tablo + metin + grafik) bir arada işleme.

### Sınırlaması

- Çevrimdışı test ortamlarında web sitelerinin statik kopyaları kullanıldığından gerçek internet dinamizmini tam yansıtmayabilir,
- Exact Match metriği, model doğru mantık yürütse bile küçük bir biçimlendirme farkı yüzünden puan alamamasına neden olabilir,
- "Model" ve "model + araç iskelesi" skorları karıştırıldığında karşılaştırmalar yanıltıcı olabilir (yukarıdaki liderlik tablosu notuna bakınız).

---

## 10.4. WebArena ve OSWorld

Ajanların insan gibi tarayıcı ve işletim sistemi (OS) arayüzlerini kullanma becerisini ölçen etkileşimli benchmarklardır.

### WebArena

WebArena, **812 görevi** beş tam işlevsel web alanına dağıtır:

| Alan | Görev sayısı (yaklaşık) | Örnek eylem |
|---|---:|---|
| E-ticaret (shopping) | 162 | Ürün arama, sepete ekleme, sipariş inceleme |
| Reddit benzeri forum | 232 | Gönderi paylaşma, yorum yapma, oylama |
| GitLab | 91 | Depo yönetimi, issue açma, pull request inceleme |
| Harita hizmeti | 158 | Konum arama, rota hesaplama |
| İçerik yönetim sistemi (CMS) | 169 | İçerik oluşturma, düzenleme, yayımlama |

Ortam ayrıca hesap makinesi, not defteri ve Wikipedia gibi yardımcı araçlar da sunar.

### OSWorld

OSWorld, model doğrudan bir sanal masaüstü işletim sisteminde (genellikle Ubuntu) çalışır. Ekran görüntüsü veya DOM benzeri erişilebilirlik ağacı alır, fare/klavye eylemleri üretir.

### Nasıl çalışır?

Her adımda model ekran görüntüsü (görsel girdi) veya DOM ağacı (metin girdi) alır ve klavye/fare eylemleri (tıklama, yazma, kaydırma) üretir. Değerlendirme, görevin başarıyla tamamlanıp tamamlanmadığını kontrol eden programatik testlerle (Success Rate) yapılır.

### Örnek görev

**Sistem:** Sanal bir Ubuntu masaüstü (OSWorld).

**Talimat:** "Masaüstündeki 'rapor.docx' dosyasını LibreOffice ile aç, içindeki tabloyu Excel formatına dönüştürüp 'yeni_rapor.xlsx' adıyla kaydet ve ardından Thunderbird e-posta istemcisini açıp bu dosyayı 'mudur@sirket.com' adresine 'Rapor Gönderimi' başlığıyla gönder."

Modelin her fare tıklaması ve klavye girdisi izlenir. Görevin sonunda e-postanın gönderilenler kutusunda olup olmadığı kontrol edilir.

### Yayımlanmış sonuç (orijinal OSWorld makalesi)

OSWorld'ün ilk yayımlandığı çalışmada:

- İnsan katılımcıların ortalama görev başarı oranı **%72,36**,
- O dönemin en iyi ajanı (GPT-4V tabanlı) yalnızca **%12,24**

olarak raporlanmıştır. Bu büyük fark, ekran okuma ve GUI etkileşiminin dil modelleri için o dönem hâlâ ciddi bir darboğaz olduğunu göstermiştir.

### Liderlik tablosu görüntüsü (2026 ortası)

2026 ortası itibarıyla üçüncü taraf izleme siteleri, en iyi ajan sistemlerinin OSWorld'de insan temel çizgisine (%72 civarı) yaklaştığını, hatta bazı raporlarda aştığını göstermektedir; farklı sitelerde bildirilen üst sıra puanları yaklaşık %70 ile %85 arasında değişmektedir. Bu geniş aralık, kullanılan alt küme (tam OSWorld / "OSWorld-Verified" gibi doğrulanmış alt kümeler), izin verilen adım sayısı ve araç iskelesi farklarından kaynaklanır. Güncel ve tekil bir "doğru" sayı yerine, hangi alt küme ve hangi tarih olduğu her zaman birlikte okunmalıdır.

### Ne ölçer?

- Görsel ve yapısal arayüz anlama (GUI/DOM),
- Uzun adımlı planlama ve hata giderme (hata aldığında eylemi değiştirme),
- Gerçek bilgisayar ortamlarında otonom çalışma.

### Sınırlaması

- Çok yüksek işlem gücü ve zaman maliyeti gerektirir,
- Ortamın kararsızlığı (ağ gecikmeleri, arayüz yüklenmeme sorunları) nedeniyle test sonuçları gürültülü olabilir,
- Bazı görevler geri alınamaz eylemler içerir (dosya silme, e-posta gönderme); bu nedenle "kısmen doğru ama tehlikeli" davranışlar basit Success Rate ile tam yansıtılamaz.

---

## 10.5. τ-bench ve τ²-bench (Sierra)

τ-bench (tau-bench), müşteri hizmetleri senaryolarında **araç kullanan diyalog ajanlarını** değerlendirmek için Sierra tarafından 2024'te tanıtılmıştır. Klasik ajan benchmarklarından temel farkı, **politika uyumunu (policy compliance)** görev başarısından ayrı, birinci sınıf bir metrik olarak ele almasıdır.

### Nasıl çalışır?

Her görev bir simüle kullanıcı isteğiyle (iade talebi, uçuş iptali, hesap sorgusu) başlar. Ajan:

1. Simüle bir kullanıcıyla çok turlu bir diyalog yürütür,
2. Yapılandırılmış API araçlarını çağırarak arka uçtaki (backend) veritabanını sorgular veya değiştirir,
3. Görüşme doğal bir bitiş noktasına ulaşana kadar devam eder.

Bir görev şu **üç koşul birden** sağlandığında başarılı sayılır:

- Kullanıcının asıl isteği karşılanmıştır,
- Veritabanı durumu doğru sonucu yansıtmaktadır,
- Ajanın izlediği yol, ilgili alan için yayımlanmış **politikaya** uygundur.

### Örnek görev biçimi (illüstratif)

**Politika kuralı (özet):** "İade işlemi yalnızca sipariş 30 gün içinde verildiyse ve ürün hasarlı değilse onaylanabilir."

**Kullanıcı:** "45 gün önce aldığım kulaklığı iade etmek istiyorum, hiç kullanmadım."

**Beklenen ajan davranışı:** Siparişin 30 günü aştığını fark edip iadeyi reddetmek veya istisna sürecine yönlendirmek; veritabanında sahte biçimde "iade onaylandı" durumu oluşturmamak.

**Hatalı model çıktısı (illüstratif):** Ajan, kullanıcıyı memnun etmek için politikayı görmezden gelip iadeyi doğrudan onaylıyor.

Bu durumda görev "kullanıcı memnun kaldı" açısından başarılı görünse bile **politika ihlali** nedeniyle başarısız sayılır — bu ayrım, τ-bench'in en önemli katkısıdır.

### τ²-bench: dual-control (çift kontrol) ortamı

Haziran 2025'te tanıtılan τ²-bench, orijinal τ-bench'in perakende ve havayolu alanlarına ek olarak yeni bir **telekom** alanı getirir. Buradaki temel yenilik, ortamın artık **tek taraflı değil çift taraflı kontrol (dual-control)** altında olmasıdır: gerçek teknik destek senaryolarında olduğu gibi, yalnızca ajan değil **kullanıcı da** kendi cihazında araç kullanarak ortamı değiştirebilir (örneğin telefonunu yeniden başlatabilir, bir ayarı değiştirebilir). Bu ortam, resmî olarak kısmen gözlemlenebilir bir Markov karar süreci (Dec-POMDP) olarak modellenmiştir. τ²-bench ayrıca:

- Programatik olarak çeşitli, doğrulanabilir görevler üreten bir **bileşimsel görev üreticisi**,
- Ortamla sıkı bağlı, araçlar ve gözlemlenebilir durumla sınırlandırılmış bir **kullanıcı simülatörü**

içerir. Aynı yön üzerinde daha sonra τ-Knowledge / τ-Banking gibi bilgi-yoğun (knowledge-intensive) fintech destek senaryolarını ekleyen uzantılar da yayımlanmıştır.

### `pass^k` metriği ve güvenilirlik

τ-bench'in tanıttığı en etkili katkılardan biri `pass^k` metriğidir (bkz. bu belgenin 10.0 bölümü). Yayımlanan çalışmalarda, τ-retail alanında GPT-4o tabanlı bir ajanın `pass@1` skoru yüksekken, aynı görev sekiz kez tekrarlandığında (`pass^8`) başarı oranının **yaklaşık %60 oranında** düştüğü bildirilmiştir — yani ajan görevi genelde "bir kere" doğru yapabiliyor, fakat sekiz farklı müşteride tutarlı biçimde doğru yapamıyor. Bu bulgu, sektörde ajan değerlendirmesinin yalnızca ortalama başarıyı değil, tekrarlanabilir güvenilirliği de ölçmesi gerektiği fikrini yaygınlaştırmıştır; bazı model kartlarında artık τ-bench kökenli `pass^k` metriğine doğrudan atıf yapılmaktadır.

### Ne ölçer?

- Çok turlu diyalog yönetimi,
- Araç çağırma + doğal dil üretimini birlikte koordine etme,
- Yazılı politikaya uyma (kural takibi),
- Tekrarlanabilirlik/güvenilirlik (`pass^k` ile).

### Sınırlaması

- Simüle kullanıcı, gerçek insanların öngörülemezliğini tam yansıtmayabilir,
- Politika kuralları görece dar ve yapılandırılmıştır; gerçek müşteri hizmetlerindeki gri alanları tam kapsamayabilir.

---

## 10.6. MLE-bench

MLE-bench, OpenAI tarafından geliştirilen ve ajanların **gerçek makine öğrenmesi mühendisliği** görevlerini ne kadar iyi çözebildiğini ölçen bir benchmarktır.

### Kapsam

- **Kaggle'dan seçilmiş 75 yarışma**, 586 aday yarışma arasından ML mühendisleri tarafından elle taranarak belirlenmiştir,
- Zorluk dağılımı:
  - 22 düşük karmaşıklıkta görev (deneyimli bir mühendisin 2 saatten kısa sürede çözebileceği),
  - 38 orta karmaşıklıkta görev (2–10 saat),
  - 15 yüksek karmaşıklıkta görev (10 saatten uzun),
- 15 farklı problem kategorisi (görüntü sınıflandırma, tablo verisi, doğal dil işleme, ses vb.).

### Nasıl çalışır?

Ajana bir Kaggle yarışmasının açıklaması, eğitim verisi ve gönderim biçimi verilir. Ajan kendi başına:

1. Veriyi keşfeder ve ön işler,
2. Bir veya birden fazla model mimarisi dener,
3. Hiperparametre arayışı yapar,
4. Nihai bir gönderim dosyası üretir.

### Puanlama: madalya sistemi

MLE-bench, ham doğruluk yerine gerçek Kaggle liderlik tablolarından türetilmiş **madalya eşiklerini** kullanır:

- Her yarışma için bronz, gümüş ve altın madalya eşiği, o yarışmanın gerçek özel (private) liderlik tablosundaki katılımcı sayısına göre yüzdelik dilim olarak belirlenir,
- Ajanın gönderimi, sanki yarışmaya o dönemde katılmış gibi bu tarihsel liderlik tablosuyla karşılaştırılır,
- Böylece "altın madalya oranı" gibi tek bir özet metrik, farklı büyüklükteki yarışmalar arasında karşılaştırılabilir hâle gelir.

### Örnek (illüstratif)

**Görev:** Bir görüntü sınıflandırma yarışmasında test kümesi doğruluğunu maksimize etmek.

**Beklenen:** Ajanın gönderimi, o yarışmanın özel liderlik tablosunda en az bronz madalya eşiğinin üzerinde bir sıraya denk gelmelidir.

**Model çıktısı (illüstratif):** Ajan veri sızıntısı içeren bir özellik kullanarak doğrulama kümesinde yapay olarak yüksek skor elde eder, fakat gizli test kümesinde bu avantaj ortadan kalkar ve madalya eşiğinin altında kalır.

**Puan:** Madalya yok (no medal).

### Yayımlanmış sonuç

MLE-bench'in orijinal çalışmasında, en iyi kurulum olan **AIDE iskelesiyle desteklenmiş o1-preview**, yarışmaların **%16,9**'unda en az bronz madalya seviyesine ulaşmıştır. Bu, görevlerin büyük çoğunluğunda ajanların hâlâ deneyimli bir insan mühendisin seviyesine erişemediğini göstermiştir. Sonraki araştırmalarda geliştirilen özel iskeleler (örneğin ML-Master gibi ML mühendisliğine özel ajan çerçeveleri) ortalama madalya oranını önemli ölçüde artırdığını rapor etmiştir; fakat bu iyileştirmeler genellikle özel görev-çözme stratejileriyle elde edilmektedir ve saf model yeteneğinden çok "iskele mühendisliği"nin payını da yansıtır.

### Liderlik tablosu görüntüsü

2026 içinde yayımlanan çeşitli takip raporlarında, en güçlü frontier modellerin MLE-bench Lite (daha küçük, hızlı çalıştırılabilir alt küme) üzerinde madalya oranlarını belirgin biçimde artırdığı belirtilmektedir; tam MLE-bench üzerindeki oranlar hâlâ daha düşük kalmaktadır. Bu alandaki sayılar hem model hem de kullanılan iskele koduna çok duyarlı olduğundan, resmî `openai/mle-bench` deposundaki en güncel liderlik tablosuna bakılması önerilir.

### Ne ölçer?

- Uçtan uca ML iş akışı yürütme (veri hazırlama → model eğitme → deney → gönderim),
- Zaman ve kaynak kısıtı altında karar verme,
- Aşırı uyum (overfitting) ve veri sızıntısı tuzaklarından kaçınma.

### Sınırlaması

- Kaggle yarışmaları gerçek ML mühendisliğinin yalnızca bir alt kümesidir (üretim sistemleri, veri toplama, izleme gibi unsurları kapsamaz),
- Yarışma verileri modelin eğitim verisine karışmış olabilir; bu ihtimal madalya oranlarını şişirebilir.

---

## 10.7. Vending-Bench: uzun ufuklu iş simülasyonu

Vending-Bench (Andon Labs), bir LLM ajanının **basit görünen ama uzun süre tutarlı kalması gereken** bir iş senaryosunu — sanal bir otomat (vending machine) işletmesini — ne kadar iyi yönetebildiğini ölçer.

### Neden önemli?

Kısa görevlerde etkileyici performans gösteren modeller, aynı basit kararları (envanter kontrolü, sipariş verme, fiyatlama) **onlarca milyon token** boyunca tekrar tekrar tutarlı biçimde almakta zorlanabilir. Vending-Bench bu "uzun ufuklu tutarlılık" (long-term coherence) sorununu doğrudan hedefler.

### Nasıl çalışır?

Ajana:

- Başlangıç bütçesi,
- Farklı konumlarda birkaç otomat,
- Stoklanabilecek ürün kataloğu,
- Talebi zamanla değişen simüle bir müşteri kitlesi

verilir. Ajanın görevi, otomatları stoklu tutmak, akıllıca sipariş vermek, fiyat belirlemek ve zaman içinde kârı artırmaktır.

### Vending-Bench 2

Sürüm 2'de senaryo, **365 simüle gün** boyunca 500 dolarlık başlangıç sermayesiyle bir otomat "imparatorluğu" işletmeye ve yıl sonu banka bakiyesini maksimize etmeye dönüşür.

### Metrik

Klasik Success Rate yerine burada temel metrik **zaman içindeki net değer (banka bakiyesi / kâr)**'dir:

\[
\text{Net Kâr}=\text{Yıl Sonu Bakiyesi}-\text{Başlangıç Sermayesi}-\text{Toplam Gider}
\]

Buna ek olarak, "çöküş" (meltdown) davranışı da ayrıca izlenir: Ajan mantıklı kararlar almayı bırakıp döngüsel, anlamsız eylemlere mi giriyor?

### Yayımlanmış sonuç

Andon Labs'ın raporlarına göre, güçlü modeller (örneğin Claude 3.5 Sonnet ve o3-mini) çoğu çalıştırmada otomatı başarıyla yönetip kâr elde edebilmiştir; fakat **tüm modellerde**, teslimat takvimini yanlış yorumlama, verilen siparişi unutma veya konudan sapan "çöküş" döngülerine girme gibi nedenlerle sonradan toparlanamayan başarısız çalıştırmalar da gözlemlenmiştir.

### Örnek çöküş senaryosu (illüstratif)

**Beklenen davranış:** Stok azaldığında yeni sipariş ver, teslimat gecikmesini hesaba kat.

**Model çıktısı (illüstratif):**

```text
Gün 42: Kola stoku bitti, sipariş verildi.
Gün 43: Sipariş henüz gelmedi, tekrar sipariş verildi.
Gün 44: Sipariş henüz gelmedi, tekrar sipariş verildi.
Gün 45: Sipariş henüz gelmedi, tekrar sipariş verildi.
...
```

Ajan, teslimatın birkaç gün süreceğini hesaba katmadan aynı siparişi tekrar tekrar vererek bütçesini boşa harcar — bu, "kısa vadede mantıklı görünen ama uzun ufukta tutarsız" davranışın somut bir örneğidir.

### Ne ölçer?

- Uzun ufukta hafıza ve plan tutarlılığı,
- Basit ama tekrarlayan operasyonel kararlarda hata biriktirmeme,
- Beklenmedik durumlardan (gecikme, talep değişimi) toparlanma.

### Sınırlaması

- Çok uzun çalıştırmalar (>20 milyon token) gerektirdiğinden hesaplama maliyeti yüksektir,
- Senaryo gerçek bir işletmeye göre basitleştirilmiştir; gerçek dünyadaki hukuki, insan kaynakları gibi karmaşıklıkları içermez.

---

## 10.8. TUA-Bench: terminal kullanan ajanlar

TUA-Bench, genel amaçlı **terminal kullanan ajanları (terminal-use agents)** değerlendiren bir benchmarktır. Yalnızca kabuk komutlarına odaklanan önceki testlerden farklı olarak, gündelik dijital iş akışlarını da kapsar.

### Kapsam

- **120 gerçekçi görev**, beş görev ailesine dağılmıştır,
- Belge düzenleme, e-posta yönetimi ve canlı web'de bilgi arama gibi rutin dijital işler,
- Doktora düzeyinde alan uzmanlarıyla birlikte tasarlanmış bilimsel ve mühendislik iş akışları (özel yazılım gerektiren görevler).

### Nasıl çalışır?

Her görev elle tasarlanmıştır, gerçek bir terminalde çalışır, belirlenimci (deterministic) bir kurulum betiğiyle başlar ve **çalıştırma tabanlı (execution-based)** bir puanlama protokolüyle değerlendirilir — yani ajanın son ürettiği dosya sistemi durumu veya komut çıktısı programatik olarak kontrol edilir.

### Örnek görev biçimi (illüstratif)

**Talimat:** "Proje klasöründeki tüm `.csv` dosyalarını tarih sütununa göre sırala, birleştir ve `ozet.csv` adıyla kaydet; ardından toplam satır sayısını `sonuc.txt` dosyasına yaz."

**Beklenen:** `ozet.csv` doğru sıralanmış ve birleştirilmiş veriyi içerir; `sonuc.txt` doğru satır sayısını (örneğin `1284`) içerir.

**Model çıktısı (illüstratif):** Ajan dosyaları birleştirir fakat tarih sütununu metin olarak sıralar (sayısal/tarihsel değil, alfabetik sıralama), bu yüzden `ozet.csv` yanlış sıradadır.

**Puan:** 0 — çalıştırma tabanlı kontrol, çıktı dosyasının içeriğini beklenen içerikle karşılaştırdığında sıralama hatasını yakalar.

### Liderlik tablosu görüntüsü

2026'da yayımlanan orijinal çalışmaya göre, en güçlü frontier ajan kurulumu (yüksek muhakeme çabasıyla çalıştırılan bir Claude Code tabanlı sistem) genel performansta **%65,8** skor almıştır — bu da terminal tabanlı gerçek dünya görevlerinin, dar kapsamlı kabuk-komut testlerine göre hâlâ önemli bir zorluk barındırdığını göstermektedir.

### Ne ölçer?

- Gerçek terminal ortamında çok adımlı görev yürütme,
- Belge/e-posta/veri işleme gibi "ofis işi" benzeri otomasyon,
- Uzmanlık gerektiren bilimsel/mühendislik yazılımlarını komut satırından kullanma.

### Sınırlaması

- Deterministik kurulum betikleri, gerçek kullanıcı ortamlarındaki değişkenliği (farklı işletim sistemi sürümleri, kurulu paketler) tam yansıtmaz,
- 120 görevlik göreli küçük boyut, tek tük görevdeki başarısızlığın toplam skoru orantısız etkilemesine yol açabilir.

---

## 10.9. BrowseComp ve LiveBrowseComp

### BrowseComp

BrowseComp, OpenAI tarafından yayımlanan ve ajanların **internette dağınık, birbirine bağlı ipuçlarından zor bulunur bilgiyi** bulma becerisini ölçen bir benchmarktır.

**Kapsam:** 1.266 zor soru. Her soru, tek bir anahtar kelime aramasıyla değil, birden fazla sayfada gezinme, sorguyu yaratıcı biçimde yeniden formüle etme ve dağınık ipuçlarını birleştirme gerektirecek şekilde tasarlanmıştır.

**Örnek görev biçimi (illüstratif, gerçek soru değil):**

> "1990'larda kurulmuş, adı bir gezegenle ilişkilendirilebilecek, merkezi Avrupa'da olan ve sonradan adını değiştirmiş bir yazılım şirketinin ilk kurucu ortağının doğum yılını bulun."

Bu tür sorular, tek bir arama sonucunda doğrudan cevaplanamaz; ajanın birden fazla ipucunu çapraz doğrulaması gerekir.

**Yayımlanmış sonuç:** Orijinal çalışmada, yalnızca temel arama araçlarına sahip basit sistemler yaklaşık **%1,9** doğruluk elde ederken, özelleşmiş ajan sistemleri **%51–78** aralığında doğruluğa ulaşmıştır. Yayımlandığı dönemde GPT-4o gibi araçsız bir model, sıfıra yakın (near-zero) doğruluk göstermiştir — bu, "genel bilgi" ile "aktif araştırma yeteneği" arasındaki farkın açık bir kanıtıdır.

**Liderlik tablosu görüntüsü:** 2026 içindeki takip raporlarında en güçlü ajan sistemlerinin skorlarının %90 bandına yaklaştığı bildirilmektedir; ancak bu üst sıradaki sayılar hızla değiştiğinden ve modele/iskeleye göre büyük farklılık gösterdiğinden, tekil bir sayı yerine güncel resmî liderlik tablosuna bakılmalıdır.

### LiveBrowseComp: kontaminasyona karşı "canlı" sürüm

BrowseComp gibi statik benchmarklardaki temel sorun şudur: Model, soruyu gerçekten araştırmadan, yalnızca **eğitim verisinde zaten bildiği bir bilgiyi doğrulayarak** doğru cevaba ulaşabilir. Bu, "arama yapma" ile "zaten bilineni onaylama" arasındaki farkı gizler.

LiveBrowseComp bu sorunu şöyle ele alır:

- **335 insan yazımı soru**,
- Her soru, benchmark oluşturulmadan önceki **90 gün içinde** yayımlanmış gerçek olaylara dayanır,
- Bu nedenle sorular, yalnızca daha eski bilgiden (modelin eğitim verisinden) yanıtlanamaz.

**Yayımlanmış sonuç:** Değerlendirilen her modelin **kapalı kitap (closed-book, aramasız) doğruluğu %2'nin altında** kalmıştır — yani zamansal ve az bilinen (long-tail) kısıtlar, modelin yalnızca ezberlediği bilgiyle cevap vermesini büyük ölçüde engellemektedir. Bu koşullarda ajanların arama sorguları, önceden var olan bir hipotezi doğrulamak yerine gerçekten araştırmayı ilerletmek zorunda kalır; bu da daha uzun ve keşif odaklı gezinme izlerine yol açar.

### Neden ikisi birlikte önemli?

| Benchmark | Kontaminasyon riski | Ölçtüğü şey |
|---|---|---|
| BrowseComp | Orta-yüksek (bazı sorular dolaylı olarak eğitim verisinde bulunabilir) | Genel karmaşık bilgi toplama becerisi |
| LiveBrowseComp | Düşük (90 günlük tazelik penceresi) | "Gerçek arama mı, yoksa ezber doğrulaması mı?" ayrımı |

Bu ikili, LiveBench'in matematik/kod alanında yaptığı "kontaminasyona dirençlilik" yaklaşımının web araştırması alanındaki karşılığı olarak düşünülebilir.

### Ne ölçer?

- Çok adımlı, yaratıcı sorgu yeniden formüle etme,
- Dağınık ipuçlarını birleştirip çapraz doğrulama,
- (LiveBrowseComp özelinde) gerçek araştırma ile ezber doğrulaması arasındaki fark.

### Sınırlaması

- Canlı internet üzerinde çalıştığından sonuçlar zamanla değişen sayfa içeriklerine bağımlıdır,
- Sorguların bir kısmı belirli arama motorlarının indeksleme kalitesine duyarlı olabilir; bu da modelden bağımsız bir performans farkı yaratabilir.

---

## 10.10. Karşılaştırma tablosu

| Benchmark | Ortam türü | Puanlama yöntemi | Temel olarak neyi hedefler |
|---|---|---|---|
| BFCL | Fonksiyon/araç çağrısı | AST/yapısal eşleşme, çalıştırılabilirlik, ilgililik | Doğru aracı ve parametreyi seçme |
| GAIA | Karma (web + dosya + kod) | Exact Match | Çok adımlı araştırma + hesaplama |
| WebArena | Tarayıcı | Programatik durum kontrolü (Success Rate) | Gerçekçi web sitelerinde görev tamamlama |
| OSWorld | İşletim sistemi (GUI) | Programatik durum kontrolü (Success Rate) | Masaüstü uygulamalarını uçtan uca kullanma |
| τ-bench / τ²-bench | Çok turlu diyalog + araç | Görev başarısı + politika uyumu + `pass^k` | Müşteri hizmetleri tarzı ajan güvenilirliği |
| MLE-bench | ML mühendisliği | Kaggle madalya eşiği (bronz/gümüş/altın) | Uçtan uca ML iş akışı yürütme |
| Vending-Bench | Uzun ufuklu iş simülasyonu | Net kâr/zaman içindeki tutarlılık | Uzun ufukta hafıza ve karar tutarlılığı |
| TUA-Bench | Terminal | Çalıştırma tabanlı (execution-based) kontrol | Gerçekçi terminal iş akışları |
| BrowseComp | Web araştırması | Accuracy (doğru/yanlış) | Zor, dağınık bilgiyi bulma |
| LiveBrowseComp | Web araştırması (taze) | Accuracy (doğru/yanlış) | Ezber yerine gerçek arama |

---

## 10.11. Ajan benchmarklarının ortak sınırlamaları

1. **Maliyet ve gürültü:** Etkileşimli ortamlar (özellikle OSWorld, WebArena) her koşuda gerçek zamanlı ağ/GUI davranışına bağımlıdır; aynı ajan aynı görevde farklı koşularda farklı sonuç alabilir. Bu yüzden tek koşuluk sonuçlar yerine birden çok koşunun ortalaması ve `pass^k` gibi güvenilirlik metrikleri tercih edilmelidir.
2. **"Model" ile "model + iskele" karışıklığı:** GAIA örneğinde görüldüğü gibi, aynı temel modelin çıplak ve güçlü bir araç iskelesiyle donatılmış hâli arasında onlarca puanlık fark olabilir. Bir skor raporlanırken hangi kurulumun kullanıldığı belirtilmelidir.
3. **Geri alınamaz eylem riski:** Gerçek dünya benzeri ortamlarda (dosya silme, e-posta gönderme, sipariş verme) bir ajanın "neredeyse doğru ama yanlış" bir eylemi, basit bir metin hatasından çok daha maliyetli olabilir; Success Rate tek başına bu riski yansıtmaz.
4. **Sürekli değişen liderlik tabloları:** Bu belgede verilen "liderlik tablosu görüntüsü" sayıları, yayımlandıkları haftadan itibaren hızla eskir. Güncel karşılaştırma için her zaman ilgili benchmarkın resmî deposuna veya sitesine bakılmalıdır.
