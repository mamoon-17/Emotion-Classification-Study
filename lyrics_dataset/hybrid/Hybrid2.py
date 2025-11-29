
# If you're on Colab, run this once in a separate cell:
# !pip install -q transformers datasets scikit-learn xgboost matplotlib

import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, top_k_accuracy_score
)

from xgboost import XGBClassifier


# ============================================================
# Config
# ============================================================

# You can switch to "distilbert-base-uncased" if DistilRoBERTa causes issues
MODEL_NAME = "distilroberta-base"      # better embeddings than DistilBERT in many cases
MAX_LENGTH = 192                       # enough for most lyrics
EMB_BATCH_SIZE = 16                    # increase if GPU memory allows
SEED = 42


# ============================================================
# Utils
# ============================================================

def set_seeds(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def embed_texts(texts, tokenizer, model, device, batch_size=16, max_length=192):
    """
    Convert a list of texts -> matrix of embeddings using
    mean pooling over the last hidden layer (masked by attention).

    Returns: np.ndarray of shape [n_samples, hidden_size]
    """
    all_embs = []
    model.eval()

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
            hidden = outputs.last_hidden_state          # [B, T, H]

            # attention_mask: 1 for real tokens, 0 for padding
            mask = enc["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            masked_hidden = hidden * mask               # zero out paddings

            # mean pooling
            summed = masked_hidden.sum(dim=1)           # [B, H]
            counts = mask.sum(dim=1).clamp(min=1e-9)    # [B, H]
            mean_pooled = summed / counts               # [B, H]

            all_embs.append(mean_pooled.cpu().numpy())

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

    # ------------------ Load encoder ------------------
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

    # ------------------ XGBoost classifier ------------------
    print("\n================ XGBoost (Hybrid LLM + Classical) ================")
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
        "Hybrid XGBoost (Distil* embeddings)",
        y_test, pred_xgb, proba_xgb,
        classes_int, label_names
    )


if __name__ == "__main__":
    main()

# ======================================================================
# 🏁 Evaluating: Hybrid XGBoost (Distil* embeddings)
# ======================================================================

# Accuracy: 0.7902
# Exact Match (EM): 0.7902
# Macro F1: 0.7877

# Classification report:
#               precision    recall  f1-score   support

#        anger       0.96      1.00      0.98       610
#         calm       0.96      0.97      0.96       610
#        happy       0.61      0.60      0.60       610
#          sad       0.61      0.60      0.60       610

#     accuracy                           0.79      2440
#    macro avg       0.79      0.79      0.79      2440
# weighted avg       0.79      0.79      0.79      2440


# Macro AUC (OvR): 0.9412
# Top-2 Accuracy: 0.9566
# Top-3 Accuracy: 0.9996