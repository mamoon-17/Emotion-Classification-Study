# Comparative Analysis of Machine Learning Models for Emotion Classification in Song Lyrics
Author: Sarem Waheed, M. Hamza Iqbal, M. Mamoon Chishti

## Overview
LyricSense is a research-driven NLP project that evaluates how different model families perform on emotion classification across two linguistically contrasting datasets. The study includes classical machine learning models, hybrid embedding-based models, and transformer-based large language models. The project explores how dataset style, annotation quality, and linguistic complexity affect performance.

## Research Goals
- Compare three modeling paradigms under identical conditions.
- Evaluate the effect of dataset difficulty and language style on performance.
- Analyze the impact of figurative vs. literal language on model behavior.
- Provide reproducible baselines for emotion-classification tasks.

## Model Families Tested

### 1. Classical Models
Implemented using TF-IDF features:
- Logistic Regression (17 iterative improvements)
- Linear SVM
- XGBoost
- Soft Voting Ensemble (best classical performer)

### 2. Hybrid Models
Transformer embeddings combined with classical classifiers:
- distilroberta-base embeddings
- roberta-base embeddings
- Emotion-tuned embeddings (j-hartmann series)

Classifiers used: Logistic Regression, SVM, XGBoost

### 3. Transformer-based LLMs
Fine-tuned using HuggingFace Transformers:
- BERT-base
- DistilRoBERTa
- RoBERTa-base
- DeBERTa-v3-base
- RoBERTa-large (best overall on GoEmotions)

## Repository Structure
```text
project-root/
│
├── lyrics_dataset/
│   ├── classical/
│   ├── hybrid/
│   └── llm/
│
├── goemotions_dataset/
│   ├── classical/
│   ├── hybrid/
│   └── llm/
│
├── requirements.txt
└── README.md
```

## Summary of Findings

### Lyrics Dataset (figurative, difficult)
| Model Type                   | Best Accuracy |
|------------------------------|---------------|
| Classical (Ensemble)         | 0.8049        |
| Hybrid (RoBERTa + XGB)       | 0.7947        |
| LLM (BERT-base)              | 0.7574        |

### GoEmotions Dataset (literal, clean)
| Model Type                        | Best Accuracy |
|-----------------------------------|---------------|
| Classical (Ensemble)              | 0.8870        |
| Hybrid (LLM Embeddings + XGB)     | 0.8800        |
| LLM (RoBERTa-large)               | 0.9300        |

### Key Insight
Dataset quality and linguistic style influence accuracy more than model complexity.

## Setup and Installation

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run models

```bash
python LLM1.py
```

(where LLM1 is the script filename)

## Datasets Used
### 1. Annanay/aml_song_lyrics_balanced
- Approximately 12.2k lyric excerpts
- Four emotion classes
- Highly figurative and metaphorical language

### 2. dair-ai/emotion (GoEmotions subset)
- 18k conversational texts
- Six emotion classes
- Clean labels and literal language


## Included Outputs
- Confusion matrices
- Training and validation accuracy curves
- Training and validation loss curves
- Full classification reports
- Final test-set metrics

## Acknowledgments
This project was developed as part of a university NLP research project, exploring practical differences between traditional ML models and modern transformer-based models in emotion classification tasks.
