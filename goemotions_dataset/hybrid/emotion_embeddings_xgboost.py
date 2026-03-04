import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, top_k_accuracy_score
)
from sklearn.preprocessing import label_binarize

from xgboost import XGBClassifier

# ============================================================
# Config
# ============================================================

# Emotion-specific model (DistilRoBERTa fine-tuned on emotions)
EMB_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

SEED           = 42
EMB_BATCH_SIZE = 32   # reduce if you hit RAM issues


# ============================================================
# Utils
# ============================================================

def set_seeds(seed: int = 42):
    import torch, random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    print("📥 Loading dataset: dair-ai/emotion (GoEmotions 6-class)")
    dataset = load_dataset("dair-ai/emotion")

    df_train = pd.DataFrame(dataset["train"])[["text", "label"]]
    df_test  = pd.DataFrame(dataset["test"])[["text", "label"]]

    label_names = dataset["train"].features["label"].names
    print("\nLabel names:", label_names)

    y_train = df_train["label"].values
    y_test  = df_test["label"].values
    X_train_text = df_train["text"].tolist()
    X_test_text  = df_test["text"].tolist()

    classes_int = np.arange(len(label_names))

    print("\nTrain size:", len(X_train_text), "Test size:", len(X_test_text))

    # ------------------ Load emotion-tuned encoder ------------------
    print(f"\n🔤 Loading SentenceTransformer model: {EMB_MODEL_NAME}")
    st_model = SentenceTransformer(EMB_MODEL_NAME)

    # ------------------ Build embeddings ------------------
    print("\n💾 Computing embeddings for TRAIN...")
    t0 = time.time()
    X_train_emb = st_model.encode(
        X_train_text,
        batch_size=EMB_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    print(f"Train embeddings shape: {X_train_emb.shape}")
    print(f"Time: {(time.time() - t0)/60:.2f} min")

    print("\n💾 Computing embeddings for TEST...")
    t0 = time.time()
    X_test_emb = st_model.encode(
        X_test_text,
        batch_size=EMB_BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    print(f"Test embeddings shape: {X_test_emb.shape}")
    print(f"Time: {(time.time() - t0)/60:.2f} min")

    # ------------------ XGBoost classifier ------------------
    print("\n================ XGBoost (Hybrid – emotion embeddings) ================")
    xgb_clf = XGBClassifier(
        objective="multi:softprob",
        num_class=len(classes_int),

        # slightly conservative, regularised setup
        n_estimators=300,
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
    xgb_clf.fit(X_train_emb, y_train)
    print(f"⏱ XGBoost training time: {(time.time() - t1)/60:.2f} min")

    proba_xgb = xgb_clf.predict_proba(X_test_emb)
    pred_xgb = np.argmax(proba_xgb, axis=1)

    evaluate_model(
        "Hybrid XGBoost (emotion-tuned DistilRoBERTa embeddings, GoEmotions)",
        y_test, pred_xgb, proba_xgb,
        classes_int, label_names
    )


if __name__ == "__main__":
    main()

# ======================================================================
# 🏁 Evaluating: Hybrid XGBoost (emotion-tuned DistilRoBERTa embeddings, GoEmotions)
# ======================================================================

# Accuracy: 0.8800
# Exact Match (EM): 0.8800
# Macro F1: 0.8233

# Classification report:
#               precision    recall  f1-score   support

#      sadness       0.92      0.93      0.92       581
#          joy       0.87      0.93      0.90       695
#         love       0.78      0.50      0.61       159
#        anger       0.89      0.89      0.89       275
#         fear       0.89      0.90      0.89       224
#     surprise       0.75      0.70      0.72        66

#     accuracy                           0.88      2000
#    macro avg       0.85      0.81      0.82      2000
# weighted avg       0.88      0.88      0.88      2000


# Macro AUC (OvR): 0.9842
# Top-2 Accuracy: 0.9770
# Top-3 Accuracy: 0.9900