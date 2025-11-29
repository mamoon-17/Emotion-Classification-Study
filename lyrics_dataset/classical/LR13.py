import os
import time
import re
import random
import numpy as np
import pandas as pd

from datasets import load_dataset

# NLP
import nltk
import contractions
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from textblob import TextBlob
from nrclex import NRCLex

# ML
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from scipy import sparse
import matplotlib.pyplot as plt


def _download_nltk_resources():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('punkt', quiet=True)


def get_wordnet_pos(tag: str):
    tag = tag[0].upper() if tag else 'N'
    return {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}.get(tag, wordnet.NOUN)


def preprocess_text_factory():
    stop_words = set(stopwords.words('english')) - {'not', 'no', 'never'}
    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = contractions.fix(text)
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"[^a-zA-Z!?\'\s]", " ", text)
        text = re.sub(r"\s{2,}", " ", text)
        text = text.lower().strip()

        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)

        negators = {"not", "no", "never"}
        cleaned = []
        i = 0
        n = len(tagged)
        while i < n:
            t, tag = tagged[i]
            if t in negators and (i + 1) < n:
                nxt, nxt_tag = tagged[i + 1]
                if re.match(r"^[a-zA-Z']+$", nxt):
                    lemma_nxt = lemmatizer.lemmatize(nxt, get_wordnet_pos(nxt_tag))
                    cleaned.append(f"not_{lemma_nxt}")
                    i += 2
                    continue
            if t not in stop_words:
                lemma = lemmatizer.lemmatize(t, get_wordnet_pos(tag))
                cleaned.append(lemma)
            i += 1
        return " ".join(cleaned)

    return preprocess_text


def extract_meta_features(text: str) -> pd.Series:
    if not isinstance(text, str):
        text = ""
    blob = TextBlob(text)
    words = text.split()
    letters = re.findall(r'[A-Za-z]', text)
    upper = re.findall(r'[A-Z]', text)
    upper_ratio = (len(upper) / len(letters)) if letters else 0.0
    elongated = len(re.findall(r'(.)\1{2,}', text))
    negations = len(re.findall(r"\b(not|no|never|n't)\b", text, flags=re.IGNORECASE))
    punctuation_density = len(re.findall(r'[!?.,]', text)) / (len(words) + 1)

    # Emotion lexicon
    emotion = NRCLex(text)
    emotions = emotion.raw_emotion_scores
    emotion_features = {k: emotions.get(k, 0) for k in
                        ['joy', 'sadness', 'anger', 'fear', 'trust', 'surprise', 'anticipation', 'disgust']}

    word_count = len(words)
    unique_words = len(set(words))
    type_token_ratio = (unique_words / word_count) if word_count else 0
    avg_word_len = np.mean([len(w) for w in words]) if word_count else 0

    sentiment_val = float(blob.sentiment.polarity)
    sentiment_strength = (
        "strong_positive" if sentiment_val > 0.5 else
        "weak_positive" if sentiment_val > 0 else
        "neutral" if sentiment_val == 0 else
        "weak_negative" if sentiment_val > -0.5 else
        "strong_negative"
    )

    base = {
        'sentiment': sentiment_val,
        'subjectivity': float(blob.sentiment.subjectivity),
        'punctuation_density': punctuation_density,
        'elongated': elongated,
        'negations': negations,
        'uppercase_ratio': upper_ratio,
        'word_count': word_count,
        'unique_words': unique_words,
        'type_token_ratio': type_token_ratio,
        'avg_word_len': avg_word_len
    }
    base.update(emotion_features)
    base.update({f"sent_{sentiment_strength}": 1})
    return pd.Series(base).fillna(0)


def to_sparse_matrix(X):
    return X if sparse.issparse(X) else sparse.csr_matrix(X)


def build_pipeline(class_weights: dict) -> Pipeline:
    text_word = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 4),
        stop_words='english',
        max_df=0.97,
        min_df=2,
        max_features=150000,
        sublinear_tf=True,
        norm='l2'
    )
    text_char = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 6),
        max_features=80000,
        sublinear_tf=True,
        norm='l2'
    )

    preprocess = ColumnTransformer([
        ('word_tfidf', text_word, 'clean_lyrics'),
        ('char_tfidf', text_char, 'lyrics'),
        ('num', Pipeline([
            ('scale', StandardScaler(with_mean=False)),
            ('maxabs', MaxAbsScaler())
        ]), [
            'sentiment', 'subjectivity', 'punctuation_density',
            'elongated', 'negations', 'uppercase_ratio',
            'joy', 'sadness', 'anger', 'fear', 'trust', 'surprise', 'anticipation', 'disgust',
            'word_count', 'unique_words', 'type_token_ratio', 'avg_word_len'
        ])
    ], remainder='drop', sparse_threshold=1.0)

    model = LogisticRegression(
        solver='saga',
        max_iter=7000,
        multi_class='multinomial',
        class_weight=class_weights,
        n_jobs=-1,
        C=5.0,
        random_state=42
    )

    return Pipeline([('preprocess', preprocess), ('logreg', model)])


def main():
    np.random.seed(42)
    print('📥 Loading dataset...')
    dataset = load_dataset("Annanay/aml_song_lyrics_balanced", split="train")
    df = pd.DataFrame(dataset)[['lyrics', 'mood']]

    print('🧹 Preprocessing...')
    _download_nltk_resources()
    preprocess_text = preprocess_text_factory()
    df['clean_lyrics'] = df['lyrics'].apply(preprocess_text)

    print('💡 Extracting features...')
    meta = df['lyrics'].apply(extract_meta_features)
    df = pd.concat([df, meta], axis=1).fillna(0)

    # Handle missing data
    df = df.dropna(subset=['clean_lyrics', 'mood'])

    X = df.drop(columns=['mood'])
    y = df['mood']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))

    pipeline = build_pipeline(class_weights)

    print('🚀 Training Logistic Regression...')
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    t1 = time.time()
    print(f"✅ Training complete in {round((t1 - t0)/60, 2)} min")

    print('\n🔍 Evaluating...')
    y_pred = pipeline.predict(X_test)
    print('Accuracy:', round(accuracy_score(y_test, y_pred), 4))
    print('Macro F1:', round(f1_score(y_test, y_pred, average="macro"), 4))
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title('Improved Logistic Regression (Feature-Enriched)')
    plt.show()


if __name__ == "__main__":
    main()

# ✅ Training complete in 3.86 min

# 🔍 Evaluating...
# Accuracy: 0.7947
# Macro F1: 0.7927

# Classification Report:
#               precision    recall  f1-score   support

#        anger       0.96      1.00      0.98       610
#         calm       0.97      0.97      0.97       610
#        happy       0.62      0.59      0.60       610
#          sad       0.62      0.62      0.62       610

#     accuracy                           0.79      2440
#    macro avg       0.79      0.79      0.79      2440
# weighted avg       0.79      0.79      0.79      2440

