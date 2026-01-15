# Beige Book + LLM: Novel Research Directions

Bu rapor, mevcut PDF'lerdeki calismalardan farklilasmak icin potansiyel novel approach'lari icerir.

---

## MEVCUT CALISMALARIN EKSIKLIKLERI

| Eksiklik | PDF-1 (Georgia Tech) | PDF-2 (Stanford) |
|----------|---------------------|------------------|
| Cross-country | Sadece ABD | Sadece ABD |
| Multi-modal | Sadece text | Sadece text |
| Causal inference | Yok | Yok |
| Uncertainty quantification | Sinirli | Yok |
| Explainability | Sinirli | Yok |
| Temporal dynamics | Yok | Yok |
| Graph-based | Yok | Yok |
| Real-time system | Yok | Yok |

---

## NOVEL APPROACH ONERILERI

---

### 1. CROSS-COUNTRY CENTRAL BANK SENTIMENT ANALYSIS
**Yenilik Seviyesi:** ⭐⭐⭐⭐⭐ Cok Yuksek

**Fikir:** Farkli ulkelerin merkez bankasi raporlarini karsilastirmali analiz et.

**Veri Kaynaklari:**
| Ulke | Kaynak | Frekans | Dil |
|------|--------|---------|-----|
| ABD | Federal Reserve Beige Book | 8x/yil | EN |
| Turkiye | TCMB Enflasyon Raporu | 4x/yil | TR |
| Turkiye | PPK Kararlari | 12x/yil | TR |
| Avrupa | ECB Economic Bulletin | 8x/yil | EN |
| Ingiltere | BoE Monetary Policy Report | 4x/yil | EN |

**Arastirma Sorulari:**
- Farkli ulkelerin sentiment'lari nasil korelasyon gosteriyor?
- ABD sentiment'i gelismekte olan ulke piyasalarini onceden tahmin edebilir mi?
- Cross-border sentiment spillover etkisi var mi?

**Teknik Yaklasim:**
```
Input: [US_Beige_Book, TR_TCMB_Report, ECB_Bulletin]
                    |
          Multilingual LLM (mBERT, XLM-R, GPT-4)
                    |
          Sentiment Scores per Country
                    |
          VAR Model / Granger Causality
                    |
Output: Cross-country recession/inflation prediction
```

**Avantajlar:**
- ✅ Literaturde ilk olur
- ✅ Policy-maker'lar icin degerli
- ✅ Yayinlanabilirlik cok yuksek
- ✅ 3-4 ayda yapilabilir

**Zorluklar:**
- ⚠️ Coklu dil isleme
- ⚠️ Farkli rapor formatlari
- ⚠️ Temporal alignment

**Tahmini Sure:** 3-4 ay

---

### 2. MULTI-MODAL RECESSION FORECASTING
**Yenilik Seviyesi:** ⭐⭐⭐⭐ Yuksek

**Fikir:** Text sentiment + Numerical indicators + Market data fusion

**Arsitektur:**
```
┌─────────────────────────────────────────────────────────┐
│                    INPUT LAYER                          │
├─────────────────┬─────────────────┬─────────────────────┤
│  Beige Book     │  Numerical      │  Market Data        │
│  Text           │  Indicators     │                     │
│                 │  - Yield curve  │  - S&P 500          │
│                 │  - VIX          │  - Treasury yields  │
│                 │  - Unemployment │  - Credit spreads   │
└────────┬────────┴────────┬────────┴──────────┬──────────┘
         │                 │                   │
    ┌────▼────┐      ┌─────▼─────┐      ┌──────▼──────┐
    │  LLM    │      │   LSTM    │      │ Transformer │
    │ Encoder │      │  Encoder  │      │   Encoder   │
    └────┬────┘      └─────┬─────┘      └──────┬──────┘
         │                 │                   │
    ┌────▼────┐      ┌─────▼─────┐      ┌──────▼──────┐
    │  768d   │      │   128d    │      │    256d     │
    │Embedding│      │ Embedding │      │  Embedding  │
    └────┬────┘      └─────┬─────┘      └──────┬──────┘
         │                 │                   │
         └────────────┬────┴───────────────────┘
                      │
              ┌───────▼───────┐
              │Cross-Attention│
              └───────┬───────┘
                      │
              ┌───────▼───────┐
              │ Fusion Layer  │
              └───────┬───────┘
                      │
              ┌───────▼───────┐
              │   Recession   │
              │  Probability  │
              └───────────────┘
```

**Arastirma Sorulari:**
- Text mi numerical mi daha predictive?
- Fusion hangi horizon'da en etkili?
- Attention weights neyi gosteriyor?

**Ablation Study:**
| Model | Text | Numerical | Market | AUC |
|-------|------|-----------|--------|-----|
| A | ✓ | ✗ | ✗ | ? |
| B | ✗ | ✓ | ✗ | ? |
| C | ✗ | ✗ | ✓ | ? |
| D | ✓ | ✓ | ✗ | ? |
| E | ✓ | ✓ | ✓ | ? |

**Avantajlar:**
- ✅ PDF'lerde yok
- ✅ State-of-the-art ML teknikleri
- ✅ Ablation study yapilabilir

**Zorluklar:**
- ⚠️ Karmasik implementasyon
- ⚠️ Modality alignment
- ⚠️ Overfitting riski

**Tahmini Sure:** 5 ay

---

### 3. CAUSAL SENTIMENT ANALYSIS
**Yenilik Seviyesi:** ⭐⭐⭐⭐⭐ Cok Yuksek

**Fikir:** Correlation degil, causation bul. Hangi topic hangi economic outcome'a neden oluyor?

**Metodoloji:**

| Yontem | Aciklama | Zorluk |
|--------|----------|--------|
| Granger Causality | X, Y'yi onceden tahmin ediyor mu? | Kolay |
| Instrumental Variables | External shocks as instruments | Orta |
| Difference-in-Differences | Policy oncesi/sonrasi | Orta |
| Structural VAR | Simultaneous equations | Zor |
| Synthetic Control | Counterfactual analysis | Zor |

**Arastirma Sorulari:**
- Employment sentiment GDP'ye mi neden oluyor yoksa tersi mi?
- Hangi district'ler leading indicator?
- Monetary policy sentiment'i nasil etkiliyor?

**Ornek Sonuc:**
```
Granger Causality Test Results:
┌─────────────────────┬────────────────┬─────────┐
│ From                │ To             │ p-value │
├─────────────────────┼────────────────┼─────────┤
│ Manufacturing Sent. │ → GDP Growth   │ 0.003** │
│ Employment Sent.    │ → Unemployment │ 0.012*  │
│ Real Estate Sent.   │ → Housing Idx  │ 0.001** │
│ GDP Growth          │ → Manuf. Sent. │ 0.342   │
└─────────────────────┴────────────────┴─────────┘
** p<0.01, * p<0.05
```

**Avantajlar:**
- ✅ Ekonomistler icin cok degerli
- ✅ Policy implications
- ✅ Yuksek akademik prestij

**Zorluklar:**
- ⚠️ Ekonometri bilgisi gerekli
- ⚠️ Identification problem
- ⚠️ Confounding variables

**Tahmini Sure:** 6 ay

---

### 4. UNCERTAINTY-AWARE SENTIMENT SCORING
**Yenilik Seviyesi:** ⭐⭐⭐⭐ Orta-Yuksek

**Fikir:** Sadece sentiment degil, sentiment'in uncertainty'sini de olc.

**Mevcut Durum vs Onerilen:**
```
MEVCUT (PDF'lerde):
Input: "Employment has increased slightly..."
Output: Sentiment = 0.7

ONERILEN:
Input: "Employment has increased slightly..."
Output: Sentiment = 0.7 ± 0.15 (95% CI: [0.55, 0.85])
        Model Confidence: 82%
```

**Metodoloji:**
| Yontem | Aciklama | Maliyet |
|--------|----------|---------|
| Monte Carlo Dropout | N forward pass, variance hesapla | Dusuk |
| Ensemble Disagreement | 5 LLM, aralarindaki variance | Orta |
| Bayesian Neural Net | Full posterior | Yuksek |
| Conformal Prediction | Coverage guarantee | Orta |

**Arastirma Sorulari:**
- Yuksek uncertainty donemlerinde tahmin nasil degisiyor?
- Uncertainty kendisi bir recession indicator mi?
- LLM'ler ne zaman "emin degil"?

**Hipotez:**
> Yuksek sentiment uncertainty → ekonomik belirsizlik → artan recession riski

**Avantajlar:**
- ✅ Risk management icin kritik
- ✅ Trustworthy AI trendi
- ✅ PDF'lerde yok

**Zorluklar:**
- ⚠️ Calibration zor
- ⚠️ Computational cost
- ⚠️ Interpretation karmasik

**Tahmini Sure:** 4 ay

---

### 5. EXPLAINABLE RECESSION PREDICTION (XAI)
**Yenilik Seviyesi:** ⭐⭐⭐⭐ Orta-Yuksek

**Fikir:** Black-box model yerine, NEDEN recession tahmin ettigini acikla.

**Ornek Dashboard Ciktisi:**
```
╔════════════════════════════════════════════════════════╗
║           RECESSION PROBABILITY: 78%                   ║
╠════════════════════════════════════════════════════════╣
║                                                        ║
║  TOP CONTRIBUTING FACTORS:                             ║
║  ┌──────────────────────────────────────────────────┐  ║
║  │ 1. Manufacturing sentiment: -0.8    (+25%)       │  ║
║  │ 2. "layoffs" keyword: 15x increase  (+20%)       │  ║
║  │ 3. Real Estate: "declining" freq.   (+18%)       │  ║
║  │ 4. Consumer spending: negative      (+12%)       │  ║
║  └──────────────────────────────────────────────────┘  ║
║                                                        ║
║  KEY SENTENCES:                                        ║
║  • "Several manufacturers reported significant         ║
║     layoffs in response to declining orders"           ║
║  • "Housing market activity has slowed considerably"   ║
║                                                        ║
║  COUNTERFACTUAL:                                       ║
║  "If Manufacturing sentiment were neutral,             ║
║   probability would drop to 52%"                       ║
║                                                        ║
╚════════════════════════════════════════════════════════╝
```

**Metodoloji:**
| Yontem | Ne Acikliyor? | Zorluk |
|--------|---------------|--------|
| SHAP Values | Feature importance | Kolay |
| Attention Viz | Hangi kelimeler onemli | Kolay |
| LIME | Local explanations | Orta |
| Counterfactual | "What if" senaryolar | Orta |
| Concept Bottleneck | Human concepts | Zor |

**Avantajlar:**
- ✅ Regulators icin onemli (AI Act, EU)
- ✅ Policy-makers icin actionable
- ✅ Akademik trend (XAI)

**Zorluklar:**
- ⚠️ Explanation fidelity
- ⚠️ Human evaluation gerekli

**Tahmini Sure:** 4 ay

---

### 6. TEMPORAL REGIME DETECTION
**Yenilik Seviyesi:** ⭐⭐⭐⭐ Yuksek

**Fikir:** Ekonomi farkli "rejimlerde" calisiyor. Sentiment-outcome iliskisi rejime gore degisiyor.

**Rejim Ornekleri:**
```
Timeline: 1970 ──────────────────────────────────────── 2024

         ┌────┐     ┌────┐        ┌──┐    ┌────┐
Recession│    │     │    │        │  │    │    │
         └────┘     └────┘        └──┘    └────┘
           73        80-82         08      20

Regime:  [NORMAL][REC][NORMAL][REC][NORMAL][REC][RECOVERY][NORMAL]
```

**Hipotez:**
> Sentiment, resesyon oncesi rejimlerde daha predictive, normal donemde daha az.

**Metodoloji:**
| Yontem | Aciklama |
|--------|----------|
| Hidden Markov Model | Latent economic states |
| Regime-Switching VAR | State-dependent coefficients |
| Change Point Detection | Structural breaks |
| Mixture of Experts | Different models per regime |

**Arastirma Sorulari:**
- Kac farkli rejim var? (2? 3? 4?)
- Sentiment hangi rejimlerde daha predictive?
- Rejim degisimi onceden tespit edilebilir mi?

**Avantajlar:**
- ✅ Ekonomik teori ile uyumlu
- ✅ Non-stationary data'yi handle eder
- ✅ Novel findings potansiyeli yuksek

**Zorluklar:**
- ⚠️ Model selection (kac rejim?)
- ⚠️ Short recession periods = az veri
- ⚠️ Computational complexity

**Tahmini Sure:** 5 ay

---

### 7. GRAPH-BASED DISTRICT SENTIMENT NETWORK
**Yenilik Seviyesi:** ⭐⭐⭐⭐ Yuksek

**Fikir:** 12 Fed district'i bir network olarak modelle. Sentiment'in yayilimini incele.

**Network Yaklasimi:**
```
                    BOSTON ←──────→ NEW YORK
                      ↑               ↑
                      │               │
                      ↓               ↓
                 CLEVELAND ←──→ PHILADELPHIA
                      ↑               ↑
                      │               │
            ┌─────────┼───────────────┤
            ↓         ↓               ↓
        CHICAGO ←→ RICHMOND ←──→ ATLANTA
            ↑         ↑               ↑
            │         │               │
            ↓         ↓               ↓
      MINNEAPOLIS ←→ ST.LOUIS ←→ KANSAS CITY
            ↑                         ↑
            │                         │
            └────→ DALLAS ←──→ SAN FRANCISCO
```

**Edge Types:**
- Geographic proximity
- Economic similarity
- Trade flow intensity
- Sector overlap

**Arastirma Sorulari:**
- Sentiment hangi district'ten basliyor?
- Geographic proximity mi economic similarity mi onemli?
- Network centrality recession prediction'i iyilestiriyor mu?

**Model:**
```python
# Graph Neural Network
class DistrictGNN(nn.Module):
    def __init__(self):
        self.sentiment_encoder = LLMEncoder()
        self.gnn_layers = [GraphConv(768, 256),
                          GraphConv(256, 64)]
        self.classifier = nn.Linear(64*12, 1)

    def forward(self, text_per_district, adjacency):
        # Node features from sentiment
        node_features = self.sentiment_encoder(text_per_district)
        # Message passing
        for layer in self.gnn_layers:
            node_features = layer(node_features, adjacency)
        # Aggregate and predict
        return self.classifier(node_features.flatten())
```

**Avantajlar:**
- ✅ Spatial dynamics captured
- ✅ PDF-2'nin regional analysis'ini genisletiyor
- ✅ GNN trendy topic

**Zorluklar:**
- ⚠️ Graph construction (hangi edges?)
- ⚠️ 12 node az (overfitting riski)
- ⚠️ Interpretation zor

**Tahmini Sure:** 5 ay

---

### 8. PROMPT ENGINEERING OPTIMIZATION
**Yenilik Seviyesi:** ⭐⭐⭐ Orta

**Fikir:** LLM prompt'larini sistematik olarak optimize et.

**Deney Matrisi:**
| Prompt Type | Example | Expected Effect |
|-------------|---------|-----------------|
| Zero-shot | "Rate sentiment: [TEXT]" | Baseline |
| Few-shot (3) | "Examples: ... Now rate: [TEXT]" | +5-10% |
| Chain-of-Thought | "Let's analyze step by step..." | +3-7% |
| Role-playing | "You are a Fed economist..." | +2-5% |
| Structured Output | "Output JSON: {score, reasoning}" | Consistency |

**Prompt Ornekleri:**
```
# PROMPT A: Direct (Baseline)
"Rate the economic sentiment of this text on a scale of -1 to 1:
[TEXT]"

# PROMPT B: Chain-of-Thought
"Analyze this Federal Reserve Beige Book excerpt:
1. First, identify the main economic topics mentioned
2. For each topic, assess if the tone is positive, negative, or neutral
3. Consider any comparisons to previous periods
4. Provide an overall sentiment score from -1 (very negative) to 1 (very positive)
Text: [TEXT]"

# PROMPT C: Persona-based
"You are a senior Federal Reserve economist with 20 years of experience.
You are reviewing a Beige Book excerpt to assess economic conditions.
Based on your expertise, provide a sentiment score from -1 to 1.
Be especially attentive to employment trends and inflation signals.
Text: [TEXT]"

# PROMPT D: Comparative
"Compare this Beige Book excerpt to a typical expansion period.
Is this text more pessimistic, similar, or more optimistic?
Score: -1 (much more pessimistic) to 1 (much more optimistic)
Text: [TEXT]"
```

**Arastirma Sorulari:**
- Hangi prompt template en iyi accuracy verir?
- Few-shot ornekleri nasil secilmeli? (random vs strategic)
- Model-specific prompt differences var mi?

**Avantajlar:**
- ✅ Hizli deneyler (1-2 hafta)
- ✅ API cost azaltir
- ✅ Practical insights

**Zorluklar:**
- ⚠️ Limited novelty
- ⚠️ Reproducibility issues
- ⚠️ Model-dependent

**Tahmini Sure:** 3 ay

---

### 9. SECTOR-SPECIFIC DEEP DIVE
**Yenilik Seviyesi:** ⭐⭐⭐ Orta

**Fikir:** Tek bir sektore odaklan, derinlemesine analiz yap.

**Sektor Secenekleri:**
| Sektor | Beige Book Coverage | Economic Importance | Data Availability |
|--------|--------------------|--------------------|-------------------|
| Manufacturing | Yuksek | Yuksek | FRED, ISM |
| Real Estate | Yuksek | Yuksek | Case-Shiller |
| Employment | Cok Yuksek | Cok Yuksek | BLS |
| Banking/Finance | Orta | Cok Yuksek | FRED |
| Agriculture | Dusuk | Orta | USDA |

**Ornek: Manufacturing Focus**
```
Research Questions:
1. Manufacturing sentiment → ISM PMI correlation?
2. District-level manufacturing → local employment?
3. Supply chain mentions → inflation predictions?
4. Export mentions → trade balance?

Output Metrics:
- Monthly Manufacturing Sentiment Index
- Confidence intervals per district
- Leading indicator analysis (how many months ahead?)
```

**Avantajlar:**
- ✅ Daha az karmasik
- ✅ Sector specialists icin degerli
- ✅ Targeted policy implications

**Zorluklar:**
- ⚠️ Narrower scope = limited impact
- ⚠️ May miss cross-sector dynamics

**Tahmini Sure:** 3 ay

---

### 10. TURKISH CENTRAL BANK (TCMB) STANDALONE
**Yenilik Seviyesi:** ⭐⭐⭐⭐⭐ Cok Yuksek (Turkiye icin)

**Fikir:** Beige Book metodolojisini sadece TCMB'ye uygula.

**Veri Kaynaklari:**
| Kaynak | Frekans | Baslangic | Dil | Sayfa |
|--------|---------|-----------|-----|-------|
| Enflasyon Raporu | 4x/yil | 2006 | TR | ~80 |
| PPK Kararlari | 12x/yil | 2005 | TR | ~2 |
| Finansal Istikrar | 2x/yil | 2005 | TR | ~100 |
| Baskan Konusmalari | ~20x/yil | 2010 | TR | ~5 |

**Hedef Degiskenler:**
- Enflasyon (TUFE)
- Faiz kararlari (policy rate)
- TL/USD kuru
- BIST100 endeksi

**Turkce NLP Zorlugu:**
```
Turkce Cumle: "Enflasyon beklentileri yukari yonlu guncellendi"

Problem 1: Agglutinative dil (ek yapilar)
  - yukari+yon+lu = upward
  - guncel+le+n+di = was updated

Problem 2: Sinirli Turkce finans modeli
  - FinBERT yok
  - BERTurk var ama finans-specific degil

Cozum:
  - GPT-4 (multilingual)
  - BERTurk + fine-tuning
  - Translation → FinBERT (information loss?)
```

**Avantajlar:**
- ✅ Turkiye icin ilk calisma
- ✅ Yerel akademik etki
- ✅ Policy relevance yuksek
- ✅ Hoca icin cazip olabilir

**Zorluklar:**
- ⚠️ Turkce NLP sinirli
- ⚠️ Daha az veri (2006'dan beri)
- ⚠️ Yuksek volatilite ortami

**Tahmini Sure:** 4 ay

---

## KOMBINASYON ONERILERI

### Kombinasyon A: "Safe but Novel" (ONERILEN)
```
┌─────────────────────────────────────────────────┐
│  Beige Book (PDF metodolojisi)                  │
│            +                                     │
│  TCMB Turkce Extension                          │
│            +                                     │
│  Cross-country Comparison (US vs TR)            │
└─────────────────────────────────────────────────┘
```
- **Sure:** 4 ay
- **Risk:** Dusuk
- **Novelty:** Yuksek
- **Yayın:** A tier journal/conference

### Kombinasyon B: "Technical Excellence"
```
┌─────────────────────────────────────────────────┐
│  Multi-modal Fusion                             │
│            +                                     │
│  Uncertainty Quantification                     │
│            +                                     │
│  Explainability (SHAP)                          │
└─────────────────────────────────────────────────┘
```
- **Sure:** 5-6 ay
- **Risk:** Orta
- **Novelty:** Cok Yuksek
- **Yayın:** Top ML/Finance venue

### Kombinasyon C: "Economics Focus"
```
┌─────────────────────────────────────────────────┐
│  Causal Inference                               │
│            +                                     │
│  Temporal Regime Detection                      │
│            +                                     │
│  Policy Implications                            │
└─────────────────────────────────────────────────┘
```
- **Sure:** 6+ ay
- **Risk:** Yuksek
- **Novelty:** Cok Yuksek
- **Yayın:** Economics journal (AER, JME)

---

## HIZLI KARAR TABLOSU

| # | Yaklasim | Sure | Zorluk | Novelty | Colab? | Oneri |
|---|----------|------|--------|---------|--------|-------|
| 1 | Cross-country | 4 ay | ⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | **1. SECIM** |
| 2 | Multi-modal | 5 ay | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | 2. Secim |
| 3 | Causal | 6 ay | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | Ekonometri biliyorsan |
| 4 | Uncertainty | 4 ay | ⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | 3. Secim |
| 5 | Explainability | 4 ay | ⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | 4. Secim |
| 6 | Regime | 5 ay | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | Ilginc |
| 7 | Graph GNN | 5 ay | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⚠️ | GNN biliyorsan |
| 8 | Prompt Eng. | 3 ay | ⭐⭐ | ⭐⭐⭐ | ✅ | Hizli bitirmek icin |
| 9 | Sector Deep | 3 ay | ⭐⭐ | ⭐⭐⭐ | ✅ | Focused |
| 10 | TCMB Only | 4 ay | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | TR odakli |

---

## FINAL ONERI

Senin durumun icin (3-4 ay, Colab, makale hedefi):

### **CROSS-COUNTRY SENTIMENT ANALYSIS (ABD + TURKIYE)**

```
┌──────────────────────────────────────────────────────────────┐
│                     PROPOSED THESIS                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   US Beige Book          TCMB Reports                       │
│        │                      │                              │
│        ▼                      ▼                              │
│   ┌─────────┐           ┌─────────┐                         │
│   │FinBERT/ │           │ GPT-4/  │                         │
│   │ Mistral │           │ BERTurk │                         │
│   └────┬────┘           └────┬────┘                         │
│        │                     │                               │
│        ▼                     ▼                               │
│   US Sentiment          TR Sentiment                        │
│        │                     │                               │
│        └──────────┬──────────┘                              │
│                   │                                          │
│                   ▼                                          │
│        ┌──────────────────────┐                             │
│        │ Cross-country VAR    │                             │
│        │ Granger Causality    │                             │
│        │ Spillover Analysis   │                             │
│        └──────────────────────┘                             │
│                   │                                          │
│                   ▼                                          │
│        ┌──────────────────────┐                             │
│        │ Output:              │                             │
│        │ - US→TR spillover?   │                             │
│        │ - Leading indicators │                             │
│        │ - Policy insights    │                             │
│        └──────────────────────┘                             │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**Neden bu?**
1. ✅ Turkiye icin ilk calisma (guaranteed novelty)
2. ✅ ABD metodolojisi hazir (PDF'den al)
3. ✅ 3-4 ay yeterli
4. ✅ Colab'da calisir (FinBERT, Mistral, GPT-4 API)
5. ✅ Hem US hem TR dergileri icin uygun

---

## SONRAKI ADIMLAR

Hocayla toplantidan sonra:
1. Konu onayi al
2. TCMB verilerini topla
3. Beige Book scraper'i calistir
4. Baseline model kur
5. Cross-country analiz yap
