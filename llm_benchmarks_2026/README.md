# Güncel LLM Benchmarkları: 2026 İçin Kapsamlı Rehber

Bu klasör, tek bir dosya yerine konuya göre bölünmüş, her biri ayrıca internet araştırmasıyla derinleştirilmiş ve doğrulanmış kaynaklarla desteklenmiş **12 ayrı benchmark dosyasından** oluşur. Eski tek-dosyalık sürümün (`llm_benchmarks_guide_2026.md`) yerini alır.

**Tarih:** 21 Temmuz 2026

## Ortak sözleşme

Her dosya aynı üç etiketi kullanır:

- **Beklenen cevap** — veri kümesindeki gerçek doğru cevap (gold answer).
- **Model çıktısı** — puanlama mekanizmasını göstermek için verilmiş, çoğu zaman illüstratif (temsilî) bir örnek.
- **Yayımlanmış sonuç** — makalede/model kartında/resmî kaynakta gerçekten raporlanmış, kaynağıyla birlikte verilen bir sonuç. Doğrulanamayan sayılar bu etiketle sunulmaz; bunun yerine "liderlik tablosu görüntüsü" gibi tarihli ve değişken olduğu açıkça belirtilen bir etiketle sunulur.

Her dosya kendi "Kaynaklar" bölümüyle biter.

## Dosyalar

| # | Dosya | Konu |
|---|---|---|
| 00 | [00_puanlama_yontemleri.md](00_puanlama_yontemleri.md) | Puanlama yöntemleri: accuracy, exact match, token F1, pass@k, win rate, Bradley–Terry, retrieval metrikleri, LLM-as-a-judge, ajan başarı oranı, güvenlik skorları, Brier/ECE, makro-mikro ortalama, bootstrap güven aralıkları, Cohen's kappa; laboratuvarların (OpenAI, Anthropic, Google DeepMind, Meta) skorları nasıl raporladığı |
| 01 | [01_genel_bilgi_ve_muhakeme.md](01_genel_bilgi_ve_muhakeme.md) | MMLU, MMLU-Pro, MMLU-Redux, GPQA/Diamond, Humanity's Last Exam, BBH/BBEH, ARC-AGI-2/3, LiveBench, HELM, AGIEval, C-Eval, SuperGPQA, KOR-Bench |
| 02 | [02_matematik_benchmarklari.md](02_matematik_benchmarklari.md) | GSM8K, GSM1K, MATH/MATH-500, AIME, HARP, Putnam-AXIOM, UGMathBench, U-MATH, OlympiadBench, MathArena, FrontierMath |
| 03 | [03_kod_ve_yazilim_benchmarklari.md](03_kod_ve_yazilim_benchmarklari.md) | HumanEval, MBPP, EvalPlus, LiveCodeBench, SWE-bench/Verified/Pro, DS-1000/DS-Bench, Aider Polyglot, METR zaman-ufku metodolojisi, Terminal-Bench, rekabetçi programlama (Codeforces/IOI/ICPC) |
| 04 | [04_talimat_izleme_ve_sohbet.md](04_talimat_izleme_ve_sohbet.md) | IFEval, M-IFEval, WildIFEval, IFBench, MT-Bench, Chatbot Arena/LMArena, AlpacaEval 2.0, Arena-Hard(-v2), WildBench, MixEval, RewardBench |
| 05 | [05_dogruluk_ve_halusinasyon.md](05_dogruluk_ve_halusinasyon.md) | TruthfulQA, SimpleQA, SimpleQA Verified, HaluEval, FActScore, Vectara Hallucination Leaderboard, HalluLens, kalibrasyon/abstention |
| 06 | [06_uzun_baglam_ve_rag.md](06_uzun_baglam_ve_rag.md) | LongBench/v2, Needle-in-a-Haystack, RULER, OneRuler, Context Rot, ∞Bench, BABILong, FRAMES, RAGBench (TRACe), T²-RAGBench |
| 07 | [07_cok_dilli_benchmarklar.md](07_cok_dilli_benchmarklar.md) | Global-MMLU, MMLU-ProX, Belebele, TyDi QA, XTREME/XTREME-R, FLORES-200; **geniş bir Türkçe benchmark bölümü** (TurkishMMLU, Cetvel, TurkBench, TUMLU, `lm-evaluation-harness` Türkçe görevleri, OpenLLM Turkish Leaderboard) |
| 08 | [08_embedding_benchmarklari.md](08_embedding_benchmarklari.md) | BEIR, MTEB (8 görev ailesi), MMTEB, C-MTEB, HF MTEB liderlik tablosu yapısı |
| 09 | [09_multimodal_benchmarklar.md](09_multimodal_benchmarklar.md) | MME, SEED-Bench, MMMU/MMMU-Pro, MathVista, OCRBench/v2, ChartQA, DocVQA, Video-MME/Video-MME-v2, uzun video benchmarkları, ScreenSpot(-Pro) |
| 10 | [10_arac_ve_ajan_benchmarklari.md](10_arac_ve_ajan_benchmarklari.md) | BFCL, GAIA, WebArena, OSWorld, τ-bench/τ²-bench, MLE-bench, Vending-Bench, TUA-Bench, BrowseComp/LiveBrowseComp |
| 11 | [11_guvenlik_ve_onyargi.md](11_guvenlik_ve_onyargi.md) | HarmBench, XSTest (aşırı reddetme), BBQ/StereoSet/WinoBias/Winogender, AdvBench/JailbreakBench (yalnızca metodoloji), red-teaming yaklaşımları |

## Bu bölünmüş sürüm eskisinden ne bakımdan farklı?

- **Kapsam:** ~2.440 satırlık tek dosya yerine, toplam ~6.000 satırı aşan 12 dosya.
- **Doğrulama:** Her dosya yazılırken internet araştırması yapıldı; gerçek makale/model kartı kaynaklı sayılar kaynak linkiyle birlikte "Yayımlanmış sonuç" olarak işaretlendi, doğrulanamayanlar illüstratif olarak bırakıldı.
- **Yeni eklenen benchmarklar:** MMLU-Redux, AGIEval, C-Eval, SuperGPQA, KOR-Bench, FrontierMath, Putnam-AXIOM, HARP, Aider Polyglot, METR zaman-ufku, Terminal-Bench, Arena-Hard, WildBench, MixEval, RewardBench, HaluEval, FActScore, Vectara Hallucination Leaderboard, HalluLens, ∞Bench, BABILong, FRAMES, XTREME, BEIR, C-MTEB, MME, SEED-Bench, ChartQA, DocVQA, ScreenSpot, τ²-bench, MLE-bench, Vending-Bench, TUA-Bench, BrowseComp, XSTest, BBQ, StereoSet, WinoBias/Winogender, AdvBench/JailbreakBench.
- **Yeni bölüm:** Güvenlik ve önyargı benchmarkları (11) — orijinal dosyada eksik kalmış bir bölümdü, sıfırdan araştırılıp eklendi.
- **Türkçe odağı:** Çok dilli dosyada (07), Türkçe/İngilizce tokenizer ve veri kümesi projeleri için doğrudan kullanılabilecek somut Türkçe değerlendirme kaynaklarına geniş yer verildi.

## Sınırlamalar ve trendler

Benchmarkların neyi ölçemediği (veri sızıntısı, Goodhart yasası, LLM hakem önyargıları, sosyal/duygusal zekâ) ve 2026 sonrası trendler (statik testlerden etkileşimli ortamlara geçiş, canlı/dinamik benchmarklar, uzman düzeyi zorluk) konusundaki genel tartışma her ilgili dosyanın "Sınırlaması" bölümlerine dağıtılmıştır; en yoğun tartışma [01_genel_bilgi_ve_muhakeme.md](01_genel_bilgi_ve_muhakeme.md) ve [00_puanlama_yontemleri.md](00_puanlama_yontemleri.md) içindedir.
