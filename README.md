# Comparative Analysis of Machine Learning Models for Emotion Classification in Song Lyrics

**Authors:** Sarem Waheed, M. Hamza Iqbal, M. Mamoon Chishti

## Overview

LyricSense is a research-driven NLP project that evaluates how different model families perform on emotion classification across two linguistically contrasting datasets. The study benchmarks classical machine learning, hybrid embedding-based, and transformer-based large language models, exploring how dataset style, annotation quality, and linguistic complexity affect performance.

The full research report is available in [`report.pdf`](report.pdf).

## Research Goals

- Compare three modeling paradigms under identical conditions.
- Evaluate the effect of dataset difficulty and language style on performance.
- Analyze the impact of figurative vs. literal language on model behavior.
- Provide reproducible baselines for emotion-classification tasks.

## Model Families Tested

### 1. Classical Models

Implemented using TF-IDF features with iterative feature engineering (17-step ablation study):
- Logistic Regression (17 iterative improvements from baseline to maximal config)
- Linear SVM
- XGBoost
- Soft Voting Ensemble (best classical performer)

### 2. Hybrid Models

Transformer embeddings combined with classical classifiers:
- DistilBERT embeddings (CLS pooling)
- DistilRoBERTa embeddings (mean pooling)
- RoBERTa-base embeddings (mean pooling)
- Emotion-tuned embeddings via `j-hartmann/emotion-english-distilroberta-base`

Classifiers used: Logistic Regression, SVM, XGBoost, Ensemble

### 3. Transformer-based LLMs

Fine-tuned using HuggingFace Transformers:
- DistilRoBERTa
- RoBERTa-base
- DeBERTa-v3-base
- RoBERTa-large (best overall on GoEmotions)
- BERT-base (best LLM on Lyrics)

## Repository Structure

```
├── report.pdf                                          # Full research report
├── requirements.txt
├── README.md
│
├── lyrics_dataset/                                     # Song lyrics (figurative language)
│   ├── classical/
│   │   ├── lr_svm_xgboost_ensemble.py                 # Final multi-classifier pipeline
│   │   ├── lr_svm_xgboost_ensemble_confusion_*.png
│   │   └── ablation/                                   # 17-step feature-engineering progression
│   │       ├── 00_baseline_tfidf_lr.ipynb              # Notebook prototype
│   │       ├── 01_baseline_tfidf_lr.py                 # Baseline: TF-IDF + LR + GridSearch
│   │       ├── 02_negation_aware_tfidf.py              # Retain negation words, sublinear TF
│   │       ├── 03_vader_sentiment_features.py          # Add VADER sentiment features
│   │       ├── 04_scaled_sentiment_features.py         # Add StandardScaler for numeric features
│   │       ├── 05_contraction_pos_lemmatization.py     # Contraction expansion, POS-aware lemma
│   │       ├── 06_textblob_length_features.py          # TextBlob sentiment + length features
│   │       ├── 07_randomized_search_url_cleanup.py     # RandomizedSearchCV, URL removal
│   │       ├── 08_word_char_ngram_union.py             # Word + char n-gram FeatureUnion
│   │       ├── 09_column_transformer_refactor.py       # ColumnTransformer, sparse pipeline
│   │       ├── 10_negation_handling_meta_features.py   # Negation bigrams, expanded meta-features
│   │       ├── 11_nrc_emotion_lexicon.py               # NRC emotion lexicon features
│   │       ├── 12_lexical_richness_features.py         # Type-token ratio, lexical diversity
│   │       ├── 13_punctuation_sentiment_categories.py  # Punctuation density, sentiment bins
│   │       ├── 14_sentiment_negation_density.py        # Sentiment strength, negation density
│   │       ├── 15_word_entropy_cross_val.py            # Word entropy, cross-validation
│   │       ├── 16_interaction_features_emoji.py        # Interaction features, emoji replacement
│   │       └── 17_maximal_feature_config.py            # Maximal TF-IDF + all meta-features
│   │
│   ├── hybrid/
│   │   ├── 01_distilbert_cls_ensemble.py               # DistilBERT CLS → LR/SVM/XGB/Ensemble
│   │   ├── 02_distilroberta_mean_xgboost.py            # DistilRoBERTa mean-pool → XGBoost
│   │   └── 03_roberta_mean_xgboost.py                  # RoBERTa mean-pool → XGBoost
│   │
│   └── llm/
│       ├── 01_distilroberta_finetune.py                # DistilRoBERTa fine-tune
│       ├── 02_roberta_base_finetune.py                 # RoBERTa-base fine-tune
│       ├── 03_deberta_v3_finetune.py                   # DeBERTa-v3-base fine-tune
│       ├── 04_roberta_large_finetune.py                # RoBERTa-large fine-tune
│       └── 05_bert_base_finetune.py                    # BERT-base fine-tune
│
└── goemotions_dataset/                                 # GoEmotions (literal language)
    ├── classical/
    │   └── lr_svm_xgboost_ensemble.py                  # LR + SVM + XGBoost + Ensemble
    ├── hybrid/
    │   └── emotion_embeddings_xgboost.py               # Emotion-tuned embeddings → XGBoost
    └── llm/
        └── roberta_large_finetune.py                   # RoBERTa-large fine-tune
```

## Summary of Findings

### Lyrics Dataset (figurative, difficult)

| Model Type              | Best Accuracy |
|-------------------------|---------------|
| Classical (Ensemble)    | 0.8049        |
| Hybrid (RoBERTa + XGB)  | 0.7947        |
| LLM (BERT-base)         | 0.7574        |

### GoEmotions Dataset (literal, clean)

| Model Type                    | Best Accuracy |
|-------------------------------|---------------|
| Classical (Ensemble)          | 0.8870        |
| Hybrid (LLM Embeddings + XGB) | 0.8800        |
| LLM (RoBERTa-large)           | 0.9300        |

### Key Insight

Dataset quality and linguistic style influence accuracy more than model complexity.

## Setup and Installation

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run a model

```bash
# Example: run the final classical ensemble on lyrics
python lyrics_dataset/classical/lr_svm_xgboost_ensemble.py

# Example: run a specific ablation step
python lyrics_dataset/classical/ablation/11_nrc_emotion_lexicon.py

# Example: fine-tune BERT on lyrics
python lyrics_dataset/llm/05_bert_base_finetune.py
```

## Datasets Used

### 1. Annanay/aml_song_lyrics_balanced

- Approximately 12.2k lyric excerpts
- Four emotion classes: anger, calm, happy, sad
- Highly figurative and metaphorical language

### 2. dair-ai/emotion (GoEmotions subset)

- 18k conversational texts
- Six emotion classes: sadness, joy, love, anger, fear, surprise
- Clean labels and literal language

## Included Outputs

- Confusion matrices (per model)
- Training and validation accuracy/loss curves
- Full classification reports
- Final test-set metrics

## Acknowledgments

This project was developed as part of a university NLP research project, exploring practical differences between traditional ML models and modern transformer-based models in emotion classification tasks.
