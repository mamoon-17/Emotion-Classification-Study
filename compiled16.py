import os
import re
import time
import numpy as np
import pandas as pd
import nltk
import contractions
import matplotlib.pyplot as plt

from datasets import load_dataset
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from textblob import TextBlob
from nrclex import NRCLex
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.utils.class_weight import compute_class_weight
from scipy import sparse

# ---------------------------------------------------------------------
# 📦 NLTK Setup
# ---------------------------------------------------------------------
def _download_nltk_resources():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('punkt', quiet=True)

def get_wordnet_pos(tag):
    tag = tag[0].upper() if tag else 'N'
    return {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}.get(tag, wordnet.NOUN)

# ---------------------------------------------------------------------
# 🧹 Preprocessing
# ---------------------------------------------------------------------
def preprocess_text_factory():
    stop_words = set(stopwords.words('english')) - {'no', 'not', 'never'}
    lemmatizer = WordNetLemmatizer()
    negators = {"no", "not", "never"}
    emoji_map = {
        ":)": "smile", ":-)": "smile", ":(": "sad", ":-(": "sad",
        ":D": "laugh", ":'(": "cry", "<3": "love"
    }

    def preprocess_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = contractions.fix(text)
        text = re.sub(r"http\S+|www\S+", "", text)
        for emo, word in emoji_map.items():
            text = text.replace(emo, f" {word} ")
        text = re.sub(r"[^a-zA-Z!?'\s]", " ", text)
        text = re.sub(r"\s{2,}", " ", text)
        text = text.lower().strip()

        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        processed = []
        i = 0
        while i < len(tagged):
            word, tag = tagged[i]
            if word in negators and i + 1 < len(tagged):
                next_word, next_tag = tagged[i + 1]
                lemma = lemmatizer.lemmatize(next_word, get_wordnet_pos(next_tag))
                processed.append(f"not_{lemma}")
                i += 2
                continue
            if len(word) > 2 and word not in stop_words:
                lemma = lemmatizer.lemmatize(word, get_wordnet_pos(tag))
                processed.append(lemma)
            i += 1
        return " ".join(processed)

    return preprocess_text

# ---------------------------------------------------------------------
# 💡 Meta-Feature Extraction
# ---------------------------------------------------------------------
def extract_meta_features(text: str) -> pd.Series:
    if not isinstance(text, str):
        text = ""

    blob = TextBlob(text)
    words = text.split()
    num_words = len(words)
    unique = len(set(words))
    avg_len = np.mean([len(w) for w in words]) if num_words else 0
    type_token_ratio = (unique / num_words) if num_words else 0
    puncts = re.findall(r'[!?.,]', text)
    punct_density = len(puncts) / (num_words + 1)
    elongated = len(re.findall(r'(.)\1{2,}', text))
    negations = len(re.findall(r"\b(no|not|never|n't)\b", text))
    upper_ratio = len(re.findall(r'[A-Z]', text)) / (len(re.findall(r'[A-Za-z]', text)) + 1)

    emotion = NRCLex(text)
    emotions = emotion.raw_emotion_scores
    emotion_features = {k: emotions.get(k, 0) for k in
                        ['joy', 'sadness', 'anger', 'fear', 'trust', 'surprise', 'anticipation', 'disgust']}

    sentiment = float(blob.sentiment.polarity)
    subjectivity = float(blob.sentiment.subjectivity)

    entropy = -(np.sum([p * np.log2(p) for p in
                        pd.Series(words).value_counts(normalize=True)])) if num_words > 1 else 0

    # Derived / interaction features
    pos_neg_ratio = (emotion_features['joy'] + emotion_features['trust'] + 1) / (
        emotion_features['sadness'] + emotion_features['anger'] + emotion_features['fear'] + 1)
    sentiment_strength = abs(sentiment) * (sum(emotion_features.values()) + 1)

    return pd.Series({
        'sentiment': sentiment,
        'subjectivity': subjectivity,
        'punctuation_density': punct_density,
        'elongated': elongated,
        'negations': negations,
        'uppercase_ratio': upper_ratio,
        'word_count': np.log1p(num_words),
        'unique_words': np.log1p(unique),
        'type_token_ratio': type_token_ratio,
        'avg_word_len': avg_len,
        'entropy': entropy,
        'pos_neg_ratio': pos_neg_ratio,
        'sentiment_strength': sentiment_strength,
        **emotion_features
    }).fillna(0)

# ---------------------------------------------------------------------
# 🧩 Pipeline Construction
# ---------------------------------------------------------------------
def build_pipeline(class_weights):
    word_tfidf = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.98,
        max_features=250000,
        sublinear_tf=True,
        use_idf=True,
        smooth_idf=True,
        norm='l2'
    )

    char_tfidf = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 6),
        max_features=120000,
        sublinear_tf=True,
        norm='l2'
    )

    numeric_features = [
        'sentiment', 'subjectivity', 'punctuation_density',
        'elongated', 'negations', 'uppercase_ratio',
        'joy', 'sadness', 'anger', 'fear', 'trust',
        'surprise', 'anticipation', 'disgust',
        'word_count', 'unique_words', 'type_token_ratio',
        'avg_word_len', 'entropy', 'pos_neg_ratio', 'sentiment_strength'
    ]

    preprocess = ColumnTransformer([
        ('word_tfidf', word_tfidf, 'clean_lyrics'),
        ('char_tfidf', char_tfidf, 'lyrics'),
        ('numeric', Pipeline([
            ('scaler', StandardScaler(with_mean=False)),
            ('maxabs', MaxAbsScaler())
        ]), numeric_features)
    ], sparse_threshold=1.0)

    logreg = LogisticRegression(
        solver='saga',
        multi_class='multinomial',
        class_weight=class_weights,
        C=10.0,
        penalty='l2',
        max_iter=10000,
        tol=1e-4,
        n_jobs=-1,
        random_state=42
    )

    return Pipeline([
        ('preprocess', preprocess),
        ('logreg', logreg)
    ])

# ---------------------------------------------------------------------
# 🚀 Main
# ---------------------------------------------------------------------
def main():
    np.random.seed(42)
    print('📥 Loading dataset...')
    dataset = load_dataset("Annanay/aml_song_lyrics_balanced", split="train")
    df = pd.DataFrame(dataset)[['lyrics', 'mood']]

    print('🧹 Preprocessing...')
    _download_nltk_resources()
    preprocess_text = preprocess_text_factory()
    df['clean_lyrics'] = df['lyrics'].apply(preprocess_text)

    print('💡 Extracting meta-features...')
    meta = df['lyrics'].apply(extract_meta_features)
    df = pd.concat([df, meta], axis=1).fillna(0)
    df = df.dropna(subset=['clean_lyrics', 'mood'])

    X, y = df.drop(columns=['mood']), df['mood']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))

    pipeline = build_pipeline(class_weights)

    print('🚀 Training Logistic Regression...')
    start = time.time()
    pipeline.fit(X_train, y_train)
    print(f"✅ Training complete in {round((time.time() - start)/60, 2)} min")

    print('\n🔍 5-Fold Cross-validation...')
    cv_acc = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    print("CV Accuracy:", np.round(cv_acc, 4), "→ Mean:", round(cv_acc.mean(), 4))

    print('\n🧠 Evaluating on test set...')
    y_pred = pipeline.predict(X_test)
    print('Accuracy:', round(accuracy_score(y_test, y_pred), 4))
    print('Macro F1:', round(f1_score(y_test, y_pred, average="macro"), 4))
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title('Enhanced Logistic Regression (Optimized Features)')
    plt.show()


if __name__ == "__main__":
    main()

# ✅ Training complete in 30.9 min
# CV Accuracy: [0.7643 0.7729 0.757  0.7581 0.7827] → Mean: 0.767

# 🧠 Evaluating on test set...
# Accuracy: 0.7963
# Macro F1: 0.7943

# Classification Report:
#               precision    recall  f1-score   support

#        anger       0.97      1.00      0.99       610
#         calm       0.95      0.97      0.96       610
#        happy       0.63      0.59      0.61       610
#          sad       0.62      0.62      0.62       610

#     accuracy                           0.80      2440
#    macro avg       0.79      0.80      0.79      2440
# weighted avg       0.79      0.80      0.79      2440