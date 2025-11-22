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

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, top_k_accuracy_score
)
from sklearn.utils.class_weight import compute_class_weight

from xgboost import XGBClassifier

# ---------------------------------------------------------------------
# NLTK Setup
# ---------------------------------------------------------------------
def _download_nltk_resources():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True) # Added to resolve LookupError

def get_wordnet_pos(tag):
    tag = tag[0].upper() if tag else 'N'
    return {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV
    }.get(tag, wordnet.NOUN)


# ---------------------------------------------------------------------
# Preprocessing
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
        # expand contractions
        text = contractions.fix(text)
        # remove URLs
        text = re.sub(r"http\S+|www\S+", "", text)
        # emoji handling
        for emo, word in emoji_map.items():
            text = text.replace(emo, f" {word} ")
        # keep only letters and basic punctuation
        text = re.sub(r"[^a-zA-Z!?'\s]", " ", text)
        text = re.sub(r"\s{2,}", " ", text)
        text = text.lower().strip()

        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        processed = []
        i = 0
        while i < len(tagged):
            word, tag = tagged[i]
            # handle "not good" -> "not_good"
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
# Meta-Feature Extraction
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
    upper_ratio = len(re.findall(r'[A-Z]', text)) / (
        len(re.findall(r'[A-Za-z]', text)) + 1
    )

    emotion = NRCLex(text)
    emotions = emotion.raw_emotion_scores
    emotion_features = {
        k: emotions.get(k, 0) for k in
        ['joy', 'sadness', 'anger', 'fear',
         'trust', 'surprise', 'anticipation', 'disgust']
    }

    sentiment = float(blob.sentiment.polarity)
    subjectivity = float(blob.sentiment.subjectivity)

    entropy = -(np.sum([
        p * np.log2(p) for p in
        pd.Series(words).value_counts(normalize=True)
    ])) if num_words > 1 else 0

    pos_neg_ratio = (emotion_features['joy'] + emotion_features['trust'] + 1) / \
                    (emotion_features['sadness'] + emotion_features['anger']
                     + emotion_features['fear'] + 1)
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
# Shared feature transformer (word TF-IDF + char TF-IDF + numeric)
# ---------------------------------------------------------------------
def build_preprocessor():
    word_tfidf = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 4),
        min_df=2,
        max_df=0.95,
        max_features=300_000,
        sublinear_tf=True,
        use_idf=True,
        smooth_idf=True,
        norm='l2'
    )

    char_tfidf = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 7),
        max_features=150_000,
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

    return preprocess


# ---------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------
def evaluate_model(name, y_true, y_pred, y_score, classes_int, label_names):
    print("\n" + "=" * 70)
    print(f"🏁 Evaluating: {name}")
    print("=" * 70)

    # base metrics
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print(f"\nAccuracy: {acc:.4f}")
    print(f"Exact Match (EM): {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=label_names))

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes_int)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # AUC + Top-k (need scores with shape [n_samples, n_classes])
    y_true_bin = label_binarize(y_true, classes=classes_int)
    try:
        auc_macro = roc_auc_score(
            y_true_bin, y_score, multi_class="ovr", average="macro"
        )
        print(f"Macro AUC (OvR): {auc_macro:.4f}")
    except Exception as e:
        print("⚠️ Could not compute AUC:", e)

    try:
        top2 = top_k_accuracy_score(y_true, y_score, k=2, labels=classes_int)
        top3 = top_k_accuracy_score(y_true, y_score, k=3, labels=classes_int)
        print(f"Top-2 Accuracy: {top2:.4f}")
        print(f"Top-3 Accuracy: {top3:.4f}")
    except Exception as e:
        print("⚠️ Could not compute Top-k accuracy:", e)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    np.random.seed(42)

    # toggles (set False if you want to skip any model)
    RUN_LR = True
    RUN_SVM = True
    RUN_XGB = True
    RUN_ENSEMBLE = True

    print("📥 Loading dataset...")
    dataset = load_dataset("Annanay/aml_song_lyrics_balanced", split="train")
    df = pd.DataFrame(dataset)[['lyrics', 'mood']]

    print("🧹 Preprocessing text...")
    _download_nltk_resources()
    preprocess_text = preprocess_text_factory()
    df['clean_lyrics'] = df['lyrics'].apply(preprocess_text)

    print("💡 Extracting meta-features...")
    meta = df['lyrics'].apply(extract_meta_features)
    df = pd.concat([df, meta], axis=1).fillna(0)
    df = df.dropna(subset=['clean_lyrics', 'mood'])

    # label encoding: string -> int
    y_str = df['mood']
    label_names = sorted(y_str.unique())        # ['anger','calm','happy','sad']
    label2idx = {lab: i for i, lab in enumerate(label_names)}
    y = y_str.map(label2idx).astype(int)
    X = df.drop(columns=['mood'])

    print("\nLabel distribution:")
    print(y_str.value_counts())

    # train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    classes_int = np.arange(len(label_names))

    # class weights (for LR & SVM)
    weights = compute_class_weight('balanced', classes=classes_int, y=y_train)
    class_weights = dict(zip(classes_int, weights))

    # feature transformer (fit ONCE)
    print("\n🔧 Building TF-IDF + meta feature space...")
    preproc = build_preprocessor()

    print("⚙️ Fitting transformer on training data...")
    t0 = time.time()
    X_train_trans = preproc.fit_transform(X_train)
    X_test_trans = preproc.transform(X_test)
    print(f"Transformer fit+transform time: {(time.time() - t0)/60:.2f} min")

    # containers for model scores (for ensemble)
    proba_lr = proba_svm = proba_xgb = None

    # ------------------ Logistic Regression ---------------------------
    if RUN_LR:
        print("\n================ Logistic Regression ================")
        lr_clf = LogisticRegression(
            solver='saga',
            multi_class='multinomial',
            class_weight=class_weights,
            C=10.0,
            penalty='l2',
            max_iter=15000,
            tol=1e-5,
            n_jobs=-1,
            random_state=42
        )

        t1 = time.time()
        lr_clf.fit(X_train_trans, y_train)
        print(f"⏱ LR training time: {(time.time() - t1)/60:.2f} min")

        proba_lr = lr_clf.predict_proba(X_test_trans)
        pred_lr = np.argmax(proba_lr, axis=1)

        evaluate_model("Logistic Regression (TF-IDF + meta)",
                       y_test, pred_lr, proba_lr,
                       classes_int, label_names)

    # ------------------ Calibrated Linear SVM ------------------------
    if RUN_SVM:
        print("\n================ Linear SVM (Calibrated) ================")
        svm_base = LinearSVC(
            C=1.0,
            class_weight=class_weights,
            max_iter=8000,
            random_state=42
        )
        # Calibrate to get probabilities for AUC & Top-k
        svm_clf = CalibratedClassifierCV(svm_base, method='sigmoid', cv=3)

        t1 = time.time()
        svm_clf.fit(X_train_trans, y_train)
        print(f"⏱ SVM+Calib training time: {(time.time() - t1)/60:.2f} min")

        proba_svm = svm_clf.predict_proba(X_test_trans)
        pred_svm = np.argmax(proba_svm, axis=1)

        evaluate_model("Calibrated Linear SVM (TF-IDF + meta)",
                       y_test, pred_svm, proba_svm,
                       classes_int, label_names)

    # ------------------ XGBoost --------------------------------------
    if RUN_XGB:
        print("\n================ XGBoost ================")
        xgb_clf = XGBClassifier(
            objective='multi:softprob',
            num_class=len(classes_int),
            n_estimators=250,        # slightly reduced to avoid OOM
            learning_rate=0.08,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',
            eval_metric='mlogloss',
            n_jobs=-1,
            random_state=42
        )

        t1 = time.time()
        xgb_clf.fit(X_train_trans, y_train)
        print(f"⏱ XGBoost training time: {(time.time() - t1)/60:.2f} min")

        proba_xgb = xgb_clf.predict_proba(X_test_trans)
        pred_xgb = np.argmax(proba_xgb, axis=1)

        evaluate_model("XGBoost (TF-IDF + meta)",
                       y_test, pred_xgb, proba_xgb,
                       classes_int, label_names)

    # ------------------ Soft-Voting Ensemble -------------------------
    if RUN_ENSEMBLE:
        print("\n================ Ensemble (LR + SVM + XGB) ================")
        scores = []
        weights = []

        if proba_lr is not None:
            scores.append(proba_lr)
            weights.append(0.4)   # LR slightly higher weight
        if proba_svm is not None:
            scores.append(proba_svm)
            weights.append(0.3)
        if proba_xgb is not None:
            scores.append(proba_xgb)
            weights.append(0.3)

        if len(scores) == 0:
            print("⚠️ No base model probabilities available; skipping ensemble.")
        else:
            weights = np.array(weights).reshape(-1, 1)
            stacked = np.stack(scores, axis=0)       # [n_models, n_samples, n_classes]
            # weighted average over models
            proba_ens = np.sum(stacked * weights[:, None, :], axis=0) / weights.sum()
            pred_ens = np.argmax(proba_ens, axis=1)

            evaluate_model("Ensemble (LR + SVM + XGB)",
                           y_test, pred_ens, proba_ens,
                           classes_int, label_names)


if __name__ == "__main__":
    main()

# ================ Logistic Regression ================
# /usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_logistic.py:1247: FutureWarning: 'multi_class' was deprecated in version 1.5 and will be removed in 1.7. From then on, it will always use 'multinomial'. Leave it to its default value to avoid this warning.
#   warnings.warn(
# ⏱ LR training time: 159.79 min

# ======================================================================
# 🏁 Evaluating: Logistic Regression (TF-IDF + meta)
# ======================================================================

# Accuracy: 0.7980
# Exact Match (EM): 0.7980
# Macro F1: 0.7967

# Classification report:
#               precision    recall  f1-score   support

#        anger       0.98      1.00      0.99       610
#         calm       0.97      0.97      0.97       610
#        happy       0.63      0.59      0.61       610
#          sad       0.61      0.63      0.62       610

#     accuracy                           0.80      2440
#    macro avg       0.80      0.80      0.80      2440
# weighted avg       0.80      0.80      0.80      2440


# Macro AUC (OvR): 0.9334
# Top-2 Accuracy: 0.9668
# Top-3 Accuracy: 0.9959

# ================ Linear SVM (Calibrated) ================
# ⏱ SVM+Calib training time: 1.55 min

# ======================================================================
# 🏁 Evaluating: Calibrated Linear SVM (TF-IDF + meta)
# ======================================================================

# Accuracy: 0.8000
# Exact Match (EM): 0.8000
# Macro F1: 0.7991

# Classification report:
#               precision    recall  f1-score   support

#        anger       0.98      1.00      0.99       610
#         calm       0.98      0.97      0.97       610
#        happy       0.62      0.58      0.60       610
#          sad       0.62      0.65      0.63       610

#     accuracy                           0.80      2440
#    macro avg       0.80      0.80      0.80      2440
# weighted avg       0.80      0.80      0.80      2440


# Macro AUC (OvR): 0.9308
# Top-2 Accuracy: 0.9762
# Top-3 Accuracy: 0.9996

# ================ XGBoost ================
# ⏱ XGBoost training time: 105.50 min

# ======================================================================
# 🏁 Evaluating: XGBoost (TF-IDF + meta)
# ======================================================================

# Accuracy: 0.7865
# Exact Match (EM): 0.7865
# Macro F1: 0.7877

# Classification report:
#               precision    recall  f1-score   support

#        anger       1.00      1.00      1.00       610
#         calm       0.98      0.95      0.96       610
#        happy       0.60      0.58      0.59       610
#          sad       0.58      0.62      0.60       610

#     accuracy                           0.79      2440
#    macro avg       0.79      0.79      0.79      2440
# weighted avg       0.79      0.79      0.79      2440


# Macro AUC (OvR): 0.9294
# Top-2 Accuracy: 0.9848
# Top-3 Accuracy: 0.9992

# ================ Ensemble (LR + SVM + XGB) ================

# ======================================================================
# 🏁 Evaluating: Ensemble (LR + SVM + XGB)
# ======================================================================

# Accuracy: 0.8049
# Exact Match (EM): 0.8049
# Macro F1: 0.8050

# Classification report:
#               precision    recall  f1-score   support

#        anger       0.99      1.00      0.99       610
#         calm       0.99      0.97      0.98       610
#        happy       0.63      0.60      0.61       610
#          sad       0.62      0.66      0.64       610

#     accuracy                           0.80      2440
#    macro avg       0.81      0.80      0.81      2440
# weighted avg       0.81      0.80      0.81      2440


# Macro AUC (OvR): 0.9368
# Top-2 Accuracy: 0.9807
# Top-3 Accuracy: 0.9984
