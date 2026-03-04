import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, top_k_accuracy_score
)

from xgboost import XGBClassifier

# ============================================================
# Config
# ============================================================

MODEL_NAME = "distilbert-base-uncased"  # good tradeoff: quality + speed
MAX_LENGTH = 192                        # enough for most lyrics
EMB_BATCH_SIZE = 16                     # for embedding with GPU
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
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

# Verified reproducibility settings for hybrid models
def embed_texts(texts, tokenizer, model, device, batch_size=16, max_length=192):
    """
    Convert a list of texts -> matrix of embeddings using CLS token.
    Returns: np.ndarray of shape [n_samples, hidden_size]
    """
    all_embs = []
    model.eval()

#Refactor evaluation helpers and tidy metric logging
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)

            outputs = model(**enc)
            # DistilBERT: use CLS token = first token
            hidden = outputs.last_hidden_state  # [B, T, H]
            cls_emb = hidden[:, 0, :]          # [B, H]
            all_embs.append(cls_emb.cpu().numpy())

    return np.concatenate(all_embs, axis=0)


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
    device = get_device()
    print("Using device:", device)

    # ------------------ Load dataset ------------------
    print("📥 Loading dataset: Annanay/aml_song_lyrics_balanced")
    dataset = load_dataset("Annanay/aml_song_lyrics_balanced", split="train")
    df = pd.DataFrame(dataset)[["lyrics", "mood"]]

    print("Dataset shape:", df.shape)
    print("\nLabel distribution:")
    print(df["mood"].value_counts())

    # ------------------ Label encoding ------------------
    y_str = df["mood"]
    label_names = sorted(y_str.unique())   # ['anger','calm','happy','sad']
    label2idx = {lab: i for i, lab in enumerate(label_names)}
    y = y_str.map(label2idx).astype(int).values
    X_text = df["lyrics"].tolist()

    classes_int = np.arange(len(label_names))

    # ------------------ Train / Test split ------------------
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y
    )

    print("\nTrain size:", len(X_train_text), "Test size:", len(X_test_text))

    # ------------------ Load LLM (DistilBERT) ------------------
    print(f"\n🔤 Loading tokenizer & model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModel.from_pretrained(MODEL_NAME).to(device)

    # ------------------ Build embeddings ------------------
    print("\n💾 Computing embeddings for TRAIN...")
    t0 = time.time()
    X_train_emb = embed_texts(
        X_train_text, tokenizer, base_model, device,
        batch_size=EMB_BATCH_SIZE, max_length=MAX_LENGTH
    )
    print(f"Train embeddings shape: {X_train_emb.shape}")
    print(f"Time: {(time.time() - t0)/60:.2f} min")

    print("\n💾 Computing embeddings for TEST...")
    t0 = time.time()
    X_test_emb = embed_texts(
        X_test_text, tokenizer, base_model, device,
        batch_size=EMB_BATCH_SIZE, max_length=MAX_LENGTH
    )
    print(f"Test embeddings shape: {X_test_emb.shape}")
    print(f"Time: {(time.time() - t0)/60:.2f} min")

    # ------------------ Scale features ------------------
    print("\n🔧 Scaling features (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_emb)
    X_test_scaled = scaler.transform(X_test_emb)

    # Containers for ensemble
    proba_lr = proba_svm = proba_xgb = None

    # ------------------ Logistic Regression ------------------
    if RUN_LR:
        print("\n================ Logistic Regression (Hybrid) ================")
        lr_clf = LogisticRegression(
            solver="saga",
            multi_class="multinomial",
            C=6.0,
            penalty="l2",
            max_iter=5000,
            n_jobs=-1,
            class_weight="balanced",
            random_state=SEED
        )

        t1 = time.time()
        lr_clf.fit(X_train_scaled, y_train)
        print(f"⏱ LR training time: {(time.time() - t1)/60:.2f} min")

        proba_lr = lr_clf.predict_proba(X_test_scaled)
        pred_lr = np.argmax(proba_lr, axis=1)

        evaluate_model(
            "Hybrid LR (DistilBERT embeddings)",
            y_test, pred_lr, proba_lr,
            classes_int, label_names
        )

    # ------------------ Calibrated Linear SVM ------------------
    if RUN_SVM:
        print("\n================ Linear SVM (Calibrated, Hybrid) ================")
        svm_base = LinearSVC(
            C=1.5,
            class_weight="balanced",
            max_iter=8000,
            random_state=SEED
        )

        svm_clf = CalibratedClassifierCV(
            svm_base, method="sigmoid", cv=3
        )

        t1 = time.time()
        svm_clf.fit(X_train_scaled, y_train)
        print(f"⏱ SVM+Calib training time: {(time.time() - t1)/60:.2f} min")

        proba_svm = svm_clf.predict_proba(X_test_scaled)
        pred_svm = np.argmax(proba_svm, axis=1)

        evaluate_model(
            "Hybrid Calibrated Linear SVM (DistilBERT embeddings)",
            y_test, pred_svm, proba_svm,
            classes_int, label_names
        )

    # ------------------ XGBoost ------------------
    if RUN_XGB:
        print("\n================ XGBoost (Hybrid) ================")
        xgb_clf = XGBClassifier(
            objective="multi:softprob",
            num_class=len(classes_int),
            n_estimators=250,
            learning_rate=0.08,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            eval_metric="mlogloss",
            n_jobs=-1,
            random_state=SEED
        )

        t1 = time.time()
        xgb_clf.fit(X_train_scaled, y_train)
        print(f"⏱ XGBoost training time: {(time.time() - t1)/60:.2f} min")

        proba_xgb = xgb_clf.predict_proba(X_test_scaled)
        pred_xgb = np.argmax(proba_xgb, axis=1)

        evaluate_model(
            "Hybrid XGBoost (DistilBERT embeddings)",
            y_test, pred_xgb, proba_xgb,
            classes_int, label_names
        )

    # ------------------ Soft-Voting Ensemble ------------------
    if RUN_ENSEMBLE:
        print("\n================ Ensemble (LR + SVM + XGB, Hybrid) ================")
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

        if len(scores) == 0:
            print("⚠️ No base model probabilities available; skipping ensemble.")
        else:
            weights = np.array(weights).reshape(-1, 1)
            stacked = np.stack(scores, axis=0)  # [n_models, n_samples, n_classes]
            proba_ens = np.sum(stacked * weights[:, None, :], axis=0) / weights.sum()
            pred_ens = np.argmax(proba_ens, axis=1)

            evaluate_model(
                "Hybrid Ensemble (LR + SVM + XGB, DistilBERT embeddings)",
                y_test, pred_ens, proba_ens,
                classes_int, label_names
            )


if __name__ == "__main__":
    main()

# ================ Logistic Regression (Hybrid) ================
# /usr/local/lib/python3.12/dist-packages/sklearn/linear_model/_logistic.py:1247: FutureWarning: 'multi_class' was deprecated in version 1.5 and will be removed in 1.7. From then on, it will always use 'multinomial'. Leave it to its default value to avoid this warning.
#   warnings.warn(
# ⏱ LR training time: 3.87 min

# ======================================================================
# 🏁 Evaluating: Hybrid LR (DistilBERT embeddings)
# ======================================================================

# Accuracy: 0.4635
# Exact Match (EM): 0.4635
# Macro F1: 0.4553

# Classification report:
#               precision    recall  f1-score   support

#        anger       0.59      0.71      0.65       610
#         calm       0.44      0.46      0.45       610
#        happy       0.41      0.37      0.39       610
#          sad       0.37      0.30      0.33       610

#     accuracy                           0.46      2440
#    macro avg       0.45      0.46      0.46      2440
# weighted avg       0.45      0.46      0.46      2440


# Macro AUC (OvR): 0.7087
# Top-2 Accuracy: 0.7299
# Top-3 Accuracy: 0.9033

# ================ Linear SVM (Calibrated, Hybrid) ================
# ⏱ SVM+Calib training time: 1.38 min

# ======================================================================
# 🏁 Evaluating: Hybrid Calibrated Linear SVM (DistilBERT embeddings)
# ======================================================================

# Accuracy: 0.4639
# Exact Match (EM): 0.4639
# Macro F1: 0.4486

# Classification report:
#               precision    recall  f1-score   support

#        anger       0.56      0.73      0.63       610
#         calm       0.44      0.52      0.48       610
#        happy       0.40      0.37      0.39       610
#          sad       0.39      0.24      0.30       610

#     accuracy                           0.46      2440
#    macro avg       0.45      0.46      0.45      2440
# weighted avg       0.45      0.46      0.45      2440


# Macro AUC (OvR): 0.7150
# Top-2 Accuracy: 0.7291
# Top-3 Accuracy: 0.9061

# ================ XGBoost (Hybrid) ================
# ⏱ XGBoost training time: 3.80 min

# ======================================================================
# 🏁 Evaluating: Hybrid XGBoost (DistilBERT embeddings)
# ======================================================================

# Accuracy: 0.7939
# Exact Match (EM): 0.7939
# Macro F1: 0.7906

# Classification report:
#               precision    recall  f1-score   support

#        anger       0.95      1.00      0.97       610
#         calm       0.96      0.97      0.97       610
#        happy       0.62      0.59      0.61       610
#          sad       0.63      0.61      0.62       610

#     accuracy                           0.79      2440
#    macro avg       0.79      0.79      0.79      2440
# weighted avg       0.79      0.79      0.79      2440


# Macro AUC (OvR): 0.9393
# Top-2 Accuracy: 0.9549
# Top-3 Accuracy: 0.9963

# ================ Ensemble (LR + SVM + XGB, Hybrid) ================

# ======================================================================
# 🏁 Evaluating: Hybrid Ensemble (LR + SVM + XGB, DistilBERT embeddings)
# ======================================================================

# Accuracy: 0.6816
# Exact Match (EM): 0.6816
# Macro F1: 0.6614

# Classification report:
#               precision    recall  f1-score   support

#        anger       0.75      0.98      0.85       610
#         calm       0.73      0.87      0.79       610
#        happy       0.57      0.48      0.52       610
#          sad       0.59      0.40      0.48       610

#     accuracy                           0.68      2440
#    macro avg       0.66      0.68      0.66      2440
# weighted avg       0.66      0.68      0.66      2440


# Macro AUC (OvR): 0.8709
# Top-2 Accuracy: 0.8754
# Top-3 Accuracy: 0.9807