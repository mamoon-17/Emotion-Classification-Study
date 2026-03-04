import os
import time
import re
import numpy as np
import pandas as pd

from datasets import load_dataset

# Text/NLP
import nltk
import contractions
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from textblob import TextBlob

# Modeling
from scipy.stats import loguniform
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
from scipy import sparse


def _download_nltk_resources() -> None:
    """Download required NLTK resources (idempotent)."""
    # Quiet downloads reduce console noise
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    # Newer NLTK versions split punkt resources
    nltk.download('punkt', quiet=True)
    try:
        nltk.download('punkt_tab', quiet=True)
    except Exception:
        # Older NLTK versions won't have this; safe to ignore
        pass
    # Some versions expect the language-specific tagger name
    try:
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    except Exception:
        pass


def get_wordnet_pos(tag: str):
    tag = tag[0].upper() if tag else 'N'
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def preprocess_text_factory():
    stop_words = set(stopwords.words('english')) - {'not', 'no', 'never'}
    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = contractions.fix(text)
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'http\S+|www\S+', '', text)
        # Keep ! and ? since they convey emotion
        text = re.sub(r"[^a-zA-Z!?\'\s]", '', text)
        text = re.sub(r'(.)\1{4,}', r'\1\1', text)  # keep mild repetition
        text = text.lower().strip()

        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        cleaned = [
            lemmatizer.lemmatize(t, get_wordnet_pos(tag))
            for t, tag in tagged if t not in stop_words
        ]
        return ' '.join(cleaned)

    return preprocess_text


def extract_meta_features(text: str) -> pd.Series:
    if not isinstance(text, str):
        text = ""
    blob = TextBlob(text)
    return pd.Series({
        'sentiment': float(blob.sentiment.polarity),
        'exclamations': text.count('!'),
        'questions': text.count('?'),
        'first_person': len(re.findall(r'\b(i|me|my|mine)\b', text, flags=re.IGNORECASE)),
    })


def to_sparse_matrix(X: np.ndarray):
    # Convert dense numeric array to sparse CSR to keep the whole feature space sparse
    if sparse.issparse(X):
        return X
    return sparse.csr_matrix(X)


def build_pipeline(class_weights: dict) -> Pipeline:
    text_word = TfidfVectorizer(
        analyzer='word', ngram_range=(1, 3),
        min_df=2, max_df=0.9, max_features=50000,
        sublinear_tf=True, smooth_idf=True,
        dtype=np.float32,
    )
    text_char = TfidfVectorizer(
        analyzer='char_wb', ngram_range=(3, 5),
        min_df=2, max_df=0.9, max_features=30000,
        sublinear_tf=True, smooth_idf=True,
        dtype=np.float32,
    )

    preprocess = ColumnTransformer(
        transformers=[
            ('word_tfidf', text_word, 'clean_lyrics'),
            ('char_tfidf', text_char, 'clean_lyrics'),
            # numeric features -> sparse -> scale without densifying
            (
                'num',
                Pipeline([
                    ('to_sparse', FunctionTransformer(to_sparse_matrix, accept_sparse=True)),
                    ('scale', MaxAbsScaler()),
                ]),
                ['sentiment', 'exclamations', 'questions', 'first_person'],
            ),
        ],
        remainder='drop',
        sparse_threshold=1.0,  # force sparse output when possible
        n_jobs=None,
    )

    logreg = LogisticRegression(
        max_iter=5000,
        class_weight=class_weights,
        solver='saga',
        multi_class='multinomial',
        C=5.0,
        n_jobs=-1,
        random_state=42,
    )

    pipe = Pipeline([
        ('preprocess', preprocess),
        ('logreg', logreg),
    ])
    return pipe


def main():
    np.random.seed(42)

    print('📥 Loading dataset...')
    fast = os.getenv('FAST', '0') == '1'
    # For smoke tests, you can run with FAST=1 to fetch a subset only
    if fast:
        dataset = load_dataset("Annanay/aml_song_lyrics_balanced", split="train[:2000]")
        df = pd.DataFrame(dataset)[['lyrics', 'mood']]
    else:
        dataset = load_dataset("Annanay/aml_song_lyrics_balanced")
        data = dataset['train']
        df = pd.DataFrame(data)[['lyrics', 'mood']]

    print('🧹 Cleaning lyrics...')
    _download_nltk_resources()
    preprocess_text = preprocess_text_factory()
    df['clean_lyrics'] = df['lyrics'].apply(preprocess_text)

    print('💡 Adding sentiment features...')
    extra_features = df['lyrics'].apply(extract_meta_features)
    df = pd.concat([df, extra_features], axis=1)

    # Prepare modeling frame
    X = df[['clean_lyrics', 'sentiment', 'exclamations', 'questions', 'first_person']]
    y = df['mood']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Class weights
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))

    # Pipeline
    pipeline = build_pipeline(class_weights)

    # Hyperparameter tuning with randomized search
    param_distributions = {
        'preprocess__word_tfidf__ngram_range': [(1, 2), (1, 3)],
        'preprocess__word_tfidf__min_df': [2, 3, 5],
        'preprocess__word_tfidf__max_df': [0.85, 0.9, 0.95],
        'preprocess__word_tfidf__max_features': [30000, 50000, 80000],

        'preprocess__char_tfidf__ngram_range': [(3, 5), (3, 6)],
        'preprocess__char_tfidf__min_df': [2, 3, 5],
        'preprocess__char_tfidf__max_features': [20000, 30000, 50000],

        'logreg__C': loguniform(0.5, 20.0),
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=18 if not fast else 4,
        scoring='f1_macro',
        n_jobs=-1,
        cv=5,
        verbose=2,
        random_state=42,
        refit=True,
    )

    print('🚀 Training Logistic Regression (with tuning)...')
    t0 = time.time()
    search.fit(X_train, y_train)
    t1 = time.time()
    print(f"✅ Training + tuning complete in {round((t1 - t0) / 60, 2)} minutes.")
    print('Best params:', search.best_params_)

    best_model = search.best_estimator_

    # Evaluation
    print('\n🔍 Evaluating on test set...')
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average='macro')
    print('Accuracy:', round(acc, 4))
    print('Macro F1 Score:', round(f1m, 4))
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap='Purples', xticks_rotation=45)
    plt.title('Confusion Matrix - Tuned Logistic Regression (TF-IDF + Meta)')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()