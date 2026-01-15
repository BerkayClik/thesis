# Tez Konusu Karsilastirma Raporu

## Ozet Tablo

| Kriter | Beige Book + LLM | Quaternion NN | Video Transformers |
|--------|------------------|---------------|-------------------|
| Yenilik | Orta-Yuksek | Yuksek | Cok Yuksek |
| Acik Kaynak | COKIYI | IYI | ZAYIF |
| Veri Erisimi | KOLAY | ORTA | ZOR |
| Teknik Zorluk | ORTA | YUKSEK | COK YUKSEK |
| Tez Suresi | EN HIZLI | ORTA | EN YAVAS |
| Yayinlanabilirlik | YUKSEK | ORTA | YUKSEK |

---

## KONU 1: Beige Book + LLM Sentiment Analysis (PDF'lerdeki Konu)

### Ne Yapiliyor?
Federal Reserve'un Beige Book raporlarini LLM'lerle analiz ederek:
- Resesyon tahmini (nowcasting/forecasting)
- Bolgesel issizlik orani tahmini
- Ekonomik aktivite sentiment skorlari

### Artilari

1. **Zengin Acik Kaynak Ekosistemi**
   - [oscarsuen/beige-book](https://github.com/oscarsuen/beige-book) - Veri scraping ve sentiment analizi
   - [hugodevere/FOMCAnalysis-Model](https://github.com/hugodevere/FOMCAnalysis-Model) - FOMC analiz modeli
   - [darrenwchang/fedspeak](https://github.com/darrenwchang/fedspeak) - Text mining projesi
   - FinBERT, Mistral, GPT-4o gibi modeller hazir

2. **Veri Erisimi Kolay**
   - Minneapolis Fed'den 1970'ten bugune tum Beige Book'lar erisime acik
   - FRED API ile ekonomik veriler (issizlik, GDP, vb.) alinabilir
   - 526 rapor, 56,330 giris, 55 yillik veri

3. **Kanitlanmis Sonuclar**
   - PDF-1'de F1 skoru 0.89'a kadar ulasiliyor (nowcasting)
   - AUC degerleri 0.95-0.96 (geleneksel yield curve'den iyi)
   - Out-of-sample testlerde de guclu performans

4. **Acik Arastirma Alanlari**
   - Topic-level sentiment (Employment, Manufacturing, Real Estate)
   - District-level vs National-level karsilastirma
   - Farkli LLM'lerin karsilastirilmasi (GPT vs Claude vs Gemini vs Mistral)

5. **Pratik Uygulama Alani**
   - Real-time recession probability index olusturulabilir
   - Policy-maker'lar ve yatirimcilar icin gercek deger

### Eksileri

1. **Yenilik Sinirli**
   - PDF-1 zaten cok kapsamli bir calisma (ICAIF '25)
   - Cleveland Fed arastirmacilari benzer calisma yapmis
   - "Incremental contribution" riski var

2. **ABD Odakli**
   - Sadece ABD ekonomisi icin gecerli
   - Turkiye veya diger ulkeler icin veri yok

3. **LLM Maliyetleri**
   - GPT-4o, Claude API maliyetleri yuksek olabilir
   - Acik kaynak alternatifler (Mistral, FinBERT) daha dusuk performans

### Teknik Limitler
- Beige Book yapisinin zamanla degismesi (2017 oncesi/sonrasi farkli)
- Look-ahead bias riski (tarih maskeleme gerekli)
- Imbalanced dataset (resesyon %10, genisleme %90)

### Tez Icin Onerim
**Farklilik Stratejisi:**
- Turkiye TCMB raporlari + Beige Book karsilastirmasi
- Cross-country sentiment analysis
- Yeni bir LLM (Claude 3.5 Opus veya Llama 3) benchmark'i
- Multi-modal: Text + ekonomik gostergeler birlesimi

---

## KONU 2: Quaternion Neural Networks for Time-Series

### Ne Yapiliyor?
Quaternion cebiri (4 boyutlu hiperkarmasik sayilar) kullanarak:
- Zaman serisi tahmini
- Coklu degisken iliskilerini koruyarak ogrenme
- Parametre verimliligi (gercek degerli NN'lerin 1/4'u)

### Artilari

1. **Guclu Acik Kaynak Destek**
   - [Orkis-Research/Pytorch-Quaternion-Neural-Networks](https://github.com/Orkis-Research/Pytorch-Quaternion-Neural-Networks)
   - [TParcollet/Quaternion-Neural-Networks](https://github.com/TParcollet/Quaternion-Neural-Networks)
   - [QGNN](https://github.com/daiquocnguyen/QGNN) - Graph Neural Networks
   - [Quatformer](https://github.com/DAMO-DI-ML/KDD2022-Quatformer) - KDD 2022

2. **Teorik Avantajlar**
   - Hamilton product ile kanallar arasi iliski korunuyor
   - 4x daha az parametre (ayni performans)
   - Rotasyon ve fazin dogal modellenmesi

3. **Aktif Arastirma Alani (2024-2025)**
   - QSTGNN (IEEE 2025) - Spatio-temporal forecasting
   - Quaternion CNN + BiLSTM - Ruzgar hizi tahmini
   - Time series compression with quaternions (Neural Networks 2025)

4. **Benzersiz Aci**
   - Finansal zaman serileri icin henuz kapsamli calisma yok
   - Multi-asset correlation'i quaternion'larla modelleme yeni

### Eksileri

1. **Yuksek Teknik Bariyer**
   - Quaternion cebrini anlamak zaman alir
   - Hamilton product, GHR calculus ogrenilmeli
   - Debug etmesi zor

2. **Sinirli Finansal Uygulama**
   - Cogu calisma: speech recognition, human motion, wind forecasting
   - Finans icin kanitlanmis basari az

3. **4 Kanal Zorunlulugu**
   - Quaternion 4 boyutlu - verinin 4 kanalli olmasi gerekir
   - Finansal veride bu mapping zorlayici olabilir
   - Ornek: (Open, High, Low, Close) veya (Price, Volume, Volatility, Sentiment)

4. **Benchmark Eksikligi**
   - Finansal datasette quaternion vs real-valued karsilastirmasi yok
   - Kendi baseline'ini olusturman gerekir

### Teknik Limitler
- Gradyan hesabi karmasik (GHR calculus)
- Training instability riski
- Hyperparameter tuning zorlugu
- Limited community support for finance applications

### Tez Icin Onerim
**Potansiyel Arastirma:**
- Quaternion LSTM for multi-asset portfolio prediction
- (Asset1_price, Asset2_price, Correlation, Volume) quaternion encoding
- Quatformer'i stock prediction'a adapte etme
- Comparison: Real-valued vs Quaternion for financial time series

---

## KONU 3: Video Transformers for Joint Price Prediction

### Ne Yapiliyor?
Hisse senedi fiyat grafikleri/candlestick chart'lari goruntu/video olarak isleme:
- Vision Transformer (ViT) ile chart pattern recognition
- CNN-Transformer hybrid modeller
- Multi-asset joint prediction

### Artilari

1. **Cok Yenilikci**
   - Video transformer (ViViT, TimeSformer) finans icin neredeyse hic kullanilmamis
   - "First of its kind" potansiyeli yuksek
   - Yuksek yayinlanabilirlik (novelty)

2. **Guzel Sonuclar Mevcut**
   - [TF-ViTNet](https://www.mdpi.com/2227-7390/13/23/3787) - ViT + LSTM hybrid
   - [ViT for candlestick](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5224805) - Momentum, mean reversion detection
   - CNN-Trans-SPP: %45 hata azaltimi

3. **Pratik Motivasyon**
   - Trader'lar gorsel karar veriyor
   - Chart pattern'lari (head-shoulders, double top) otomatik tespit
   - Multi-timeframe analysis

4. **Transfer Learning Imkani**
   - ImageNet pretrained ViT kullanilabilir
   - Video modelleri (ViViT, VideoMAE) adapte edilebilir

### Eksileri

1. **Acik Kaynak Kod YOK**
   - "Video transformer for finance" icin hazir kod bulunamadi
   - Sifirdan implementasyon gerekli
   - En buyuk dezavantaj

2. **Veri Hazirlama Karmasik**
   - Price data -> Candlestick chart -> Image/Video donusumu
   - Multi-asset icin: synchronized video streams
   - Data pipeline olusturmak cok zaman alir

3. **Hesaplama Maliyeti**
   - Video transformer'lar cok agir (ViViT: milyonlarca parametre)
   - GPU gereksinimleri yuksek
   - Training suresi uzun

4. **"Video" Kavrami Belirsiz**
   - Hocamn kastettigi tam olarak ne?
   - Rolling window charts? Time-lapse? Multi-asset overlay?
   - Tanimlanmasi gerekiyor

5. **Overfitting Riski**
   - Gorsel bilgi finansal bilgiden zengin degil
   - Ayni veri farkli formatta = potential information leak
   - Backtest vs real-world gap

### Teknik Limitler
- Video sequence length vs GPU memory trade-off
- Temporal alignment across multiple assets
- Interpretability: Model neyi goruyor?
- Real-time inference zorlugu

### Tez Icin Onerim
**Eger secilirse:**
- Basit bir ViT + candlestick ile basla (video degil image)
- Multi-asset joint prediction icin: attention-based correlation modeling
- Comparison: Numerical features vs Visual features vs Hybrid

---

## SONUC VE ONERILERIM

### Siralama (En iyiden en kotiye)

#### 1. BEIGE BOOK + LLM (ONERILEN)
**Neden?**
- En hizli tamamlanir (3-4 ay)
- Acik kaynak kod bol
- Veri hazir
- Kanitlanmis metodoloji
- Extension yapma sansi yuksek

**Farklilik icin:**
1. Turkiye TCMB raporlarina uygula (ilk olursun)
2. Yeni LLM'ler benchmark'la (Claude 3.5 Opus, Llama 3.3)
3. District-level sentiment ile portfolio allocation
4. Cross-country comparison (US + EU + TR)

#### 2. QUATERNION NEURAL NETWORKS
**Neden ikinci?**
- Ilginc ve yenilikci
- Kod mevcut ama adapte etmek gerekiyor
- Finansal uygulama eksik = firsat
- Ama ogrenme egrisi dik

**Tahmini sure:** 5-6 ay

#### 3. VIDEO TRANSFORMERS (ONERMLMEZ)
**Neden son?**
- Kod yok, sifirdan yazilacak
- "Video" taniminin netlestirilmesi gerekli
- Hesaplama maliyeti cok yuksek
- Risk/odül orani dusuk

**Tahmini sure:** 7-8+ ay

---

## Hocayla Konusma Icin Sorular

1. **Video transformer** derken tam olarak ne kastediliyor?
   - Rolling window chart video'su mu?
   - Multi-asset overlay mi?
   - 3D volume data mi?

2. Tez **yayina donusmeli mi**? (Evetse Beige Book en guvenli)

3. **Hesaplama kaynaklari** ne durumda?
   - GPU erisimi var mi?
   - Cloud budget var mi?

4. **Zaman kisiti** ne kadar?
   - 3 ay? 6 ay? 1 yil?

5. Turkiye verisi kullanmak **avantaj mi** yoksa ABD verisi mi tercih?

---

## Kaynaklar

### Beige Book
- [oscarsuen/beige-book](https://github.com/oscarsuen/beige-book)
- [FOMCAnalysis-Model](https://github.com/hugodevere/FOMCAnalysis-Model)
- [FinBERT](https://huggingface.co/yiyanghkust/finbert-tone)

### Quaternion
- [Pytorch-Quaternion-Neural-Networks](https://github.com/Orkis-Research/Pytorch-Quaternion-Neural-Networks)
- [Quatformer (KDD 2022)](https://github.com/DAMO-DI-ML/KDD2022-Quatformer)
- [QSTGNN Paper](https://ieeexplore.ieee.org/document/11007492/)

### Vision/Video Transformers
- [TF-ViTNet Paper](https://www.mdpi.com/2227-7390/13/23/3787)
- [ViT Candlestick Paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5224805)
- [MPDTransformer](https://link.springer.com/article/10.1007/s44196-025-00768-w)
