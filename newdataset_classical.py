# If you're on Colab, run this once in a separate cell:
# !pip install -q datasets transformers scikit-learn xgboost matplotlib

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, top_k_accuracy_score
)

from xgboost import XGBClassifier

# ============================================================
# Config
# ============================================================

SEED = 42

RUN_LR = True
RUN_SVM = True
RUN_XGB = True
RUN_ENSEMBLE = True

# ============================================================
# Utils
# ============================================================

def set_seeds(seed: int = 42):
    np.random.seed(seed)


def evaluate_model(name, y_true, y_pred, y_score, classes_int, label_names):
    print("\n" + "=" * 70)
    print(f"🏁 Evaluating: {name}")
    print("=" * 70)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print(f"\nAccuracy: {acc:.4f}")
    print(f"Exact Match (EM): {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}\n")

    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=label_names))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes_int)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(f"{name} – Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # AUC + Top-k
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


# ============================================================
# Main
# ============================================================

def main():
    set_seeds(SEED)

    # ------------------ Load dataset ------------------
    print("📥 Loading dataset: dair-ai/emotion")
    emotions = load_dataset("dair-ai/emotion")   # train / validation / test
    # train: 16k, val: 2k, test: 2k  (text, label):contentReference[oaicite:0]{index=0}

    # Use train + validation for training; keep test for final evaluation
    train_df = emotions["train"].to_pandas()
    val_df   = emotions["validation"].to_pandas()
    test_df  = emotions["test"].to_pandas()

    train_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)

    print("Train size:", len(train_df), "Test size:", len(test_df))

    # Label names (in fixed order)
    label_names = emotions["train"].features["label"].names
    print("\nLabel names:", label_names)

    # y are already ints 0..5; no need to re-encode
    X_train_text = train_df["text"].tolist()
    y_train = train_df["label"].astype(int).values

    X_test_text = test_df["text"].tolist()
    y_test = test_df["label"].astype(int).values

    classes_int = np.arange(len(label_names))

    # ------------------ TF-IDF features ------------------
    print("\n🔧 Building TF-IDF feature space...")
    tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),        # unigrams + bigrams + trigrams
        min_df=2,
        max_df=0.95,
        max_features=80_000,
        sublinear_tf=True,
        use_idf=True,
        smooth_idf=True,
        norm="l2"
    )

    t0 = time.time()
    X_train = tfidf.fit_transform(X_train_text)
    X_test  = tfidf.transform(X_test_text)
    print(f"TF-IDF shapes: train {X_train.shape}, test {X_test.shape}")
    print(f"TF-IDF fit+transform time: {(time.time() - t0)/60:.2f} min")

    # Class weights (just in case there is slight imbalance)
    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=classes_int,
        y=y_train
    )
    class_weights = {cls: w for cls, w in zip(classes_int, class_weights_array)}
    print("\nClass weights:", class_weights)

    proba_lr = proba_svm = proba_xgb = None

    # ------------------ Logistic Regression ------------------
    if RUN_LR:
        from sklearn.exceptions import ConvergenceWarning
        import warnings
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        print("\n================ Logistic Regression ================")
        lr_clf = LogisticRegression(
            solver="saga",
            multi_class="multinomial",
            C=6.0,
            penalty="l2",
            max_iter=5000,
            n_jobs=-1,
            class_weight=class_weights,
            random_state=SEED
        )

        t1 = time.time()
        lr_clf.fit(X_train, y_train)
        print(f"⏱ LR training time: {(time.time() - t1)/60:.2f} min")

        proba_lr = lr_clf.predict_proba(X_test)
        pred_lr = np.argmax(proba_lr, axis=1)

        evaluate_model(
            "Logistic Regression (TF-IDF)",
            y_test, pred_lr, proba_lr,
            classes_int, label_names
        )

    # ------------------ Calibrated Linear SVM ------------------
    if RUN_SVM:
        print("\n================ Linear SVM (Calibrated) ================")
        svm_base = LinearSVC(
            C=1.0,
            class_weight=class_weights,
            max_iter=8000,
            random_state=SEED
        )
        svm_clf = CalibratedClassifierCV(
            svm_base, method="sigmoid", cv=3
        )

        t1 = time.time()
        svm_clf.fit(X_train, y_train)
        print(f"⏱ SVM+Calib training time: {(time.time() - t1)/60:.2f} min")

        proba_svm = svm_clf.predict_proba(X_test)
        pred_svm = np.argmax(proba_svm, axis=1)

        evaluate_model(
            "Calibrated Linear SVM (TF-IDF)",
            y_test, pred_svm, proba_svm,
            classes_int, label_names
        )

    # ------------------ XGBoost ------------------
    if RUN_XGB:
        print("\n================ XGBoost ================")
        xgb_clf = XGBClassifier(
            objective="multi:softprob",
            num_class=len(classes_int),

            # Reasonable defaults for this dataset
            n_estimators=400,
            learning_rate=0.05,
            max_depth=7,
            min_child_weight=2,
            subsample=0.9,
            colsample_bytree=0.9,
            gamma=0.1,
            reg_lambda=1.0,
            reg_alpha=0.0,

            tree_method="hist",
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=SEED
        )

        t1 = time.time()
        xgb_clf.fit(X_train, y_train)
        print(f"⏱ XGBoost training time: {(time.time() - t1)/60:.2f} min")

        proba_xgb = xgb_clf.predict_proba(X_test)
        pred_xgb = np.argmax(proba_xgb, axis=1)

        evaluate_model(
            "XGBoost (TF-IDF)",
            y_test, pred_xgb, proba_xgb,
            classes_int, label_names
        )

    # ------------------ Soft-Voting Ensemble ------------------
    if RUN_ENSEMBLE:
        print("\n================ Ensemble (LR + SVM + XGB) ================")
        scores = []
        weights = []

        if proba_lr is not None:
            scores.append(proba_lr)
            weights.append(0.4)
        if proba_svm is not None:
            scores.append(proba_svm)
            weights.append(0.3)
        if proba_xgb is not None:
            scores.append(proba_xgb)
            weights.append(0.3)

        if not scores:
            print("⚠️ No base model probabilities available; skipping ensemble.")
        else:
            weights = np.array(weights).reshape(-1, 1)
            stacked = np.stack(scores, axis=0)   # [n_models, n_samples, n_classes]
            proba_ens = np.sum(stacked * weights[:, None, :], axis=0) / weights.sum()
            pred_ens = np.argmax(proba_ens, axis=1)

            evaluate_model(
                "Ensemble (LR + SVM + XGB, TF-IDF)",
                y_test, pred_ens, proba_ens,
                classes_int, label_names
            )


if __name__ == "__main__":
    main()

# ================ Logistic Regression ================
# /usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_logistic.py:1247: FutureWarning: 'multi_class' was deprecated in version 1.5 and will be removed in 1.7. From then on, it will always use 'multinomial'. Leave it to its default value to avoid this warning.
#   warnings.warn(
# ⏱ LR training time: 2.59 min

# ======================================================================
# 🏁 Evaluating: Logistic Regression (TF-IDF)
# ======================================================================

# Accuracy: 0.8765
# Exact Match (EM): 0.8765
# Macro F1: 0.8330

# Classification report:
#               precision    recall  f1-score   support

#      sadness       0.93      0.90      0.91       581
#          joy       0.91      0.89      0.90       695
#         love       0.72      0.84      0.77       159
#        anger       0.86      0.88      0.87       275
#         fear       0.86      0.82      0.84       224
#     surprise       0.67      0.73      0.70        66

#     accuracy                           0.88      2000
#    macro avg       0.82      0.84      0.83      2000
# weighted avg       0.88      0.88      0.88      2000


# Macro AUC (OvR): 0.9844
# Top-2 Accuracy: 0.9805
# Top-3 Accuracy: 0.9950

# ================ Linear SVM (Calibrated) ================
# ⏱ SVM+Calib training time: 0.02 min

# ======================================================================
# 🏁 Evaluating: Calibrated Linear SVM (TF-IDF)
# ======================================================================

# Accuracy: 0.8805
# Exact Match (EM): 0.8805
# Macro F1: 0.8285

# Classification report:
#               precision    recall  f1-score   support

#      sadness       0.93      0.91      0.92       581
#          joy       0.90      0.91      0.91       695
#         love       0.74      0.77      0.76       159
#        anger       0.87      0.88      0.88       275
#         fear       0.87      0.84      0.86       224
#     surprise       0.68      0.64      0.66        66

#     accuracy                           0.88      2000
#    macro avg       0.83      0.83      0.83      2000
# weighted avg       0.88      0.88      0.88      2000


# Macro AUC (OvR): 0.9887
# Top-2 Accuracy: 0.9825
# Top-3 Accuracy: 0.9960

# ================ XGBoost ================
# ⏱ XGBoost training time: 7.92 min

# ======================================================================
# 🏁 Evaluating: XGBoost (TF-IDF)
# ======================================================================

# Accuracy: 0.8810
# Exact Match (EM): 0.8810
# Macro F1: 0.8386

# Classification report:
#               precision    recall  f1-score   support

#      sadness       0.97      0.90      0.93       581
#          joy       0.87      0.91      0.89       695
#         love       0.74      0.79      0.76       159
#        anger       0.91      0.86      0.88       275
#         fear       0.87      0.84      0.86       224
#     surprise       0.64      0.79      0.71        66

#     accuracy                           0.88      2000
#    macro avg       0.83      0.85      0.84      2000
# weighted avg       0.89      0.88      0.88      2000


# Macro AUC (OvR): 0.9850
# Top-2 Accuracy: 0.9705
# Top-3 Accuracy: 0.9875

# ================ Ensemble (LR + SVM + XGB) ================

# ======================================================================
# 🏁 Evaluating: Ensemble (LR + SVM + XGB, TF-IDF)
# ======================================================================

# Accuracy: 0.8870
# Exact Match (EM): 0.8870
# Macro F1: 0.8406

# Classification report:
#               precision    recall  f1-score   support

#      sadness       0.94      0.92      0.93       581
#          joy       0.91      0.91      0.91       695
#         love       0.73      0.81      0.76       159
#        anger       0.89      0.89      0.89       275
#         fear       0.88      0.85      0.86       224
#     surprise       0.64      0.74      0.69        66

#     accuracy                           0.89      2000
#    macro avg       0.83      0.85      0.84      2000
# weighted avg       0.89      0.89      0.89      2000


# Macro AUC (OvR): 0.9897
# Top-2 Accuracy: 0.9875
# Top-3 Accuracy: 0.9960