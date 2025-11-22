# If you're on Colab, run this once in a separate cell:
# !pip install -q transformers datasets scikit-learn matplotlib

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    set_seed,
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    top_k_accuracy_score
)
from sklearn.preprocessing import label_binarize

# =========================================================
# Config
# =========================================================
MODEL_NAME      = "bert-base-uncased"
MAX_LENGTH      = 256          # 256 works well on T4 GPU
TRAIN_BATCH     = 16
EVAL_BATCH      = 32
EPOCHS          = 5
LR              = 2e-5
WARMUP_RATIO    = 0.06
WEIGHT_DECAY    = 0.01
SEED            = 42
GRAD_ACCUM      = 1            # increase if GPU memory is tight

# =========================================================
# Utils
# =========================================================
def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)

def compute_all_metrics(name, y_true, probs, label_list):
    """
    y_true: 1D numpy array of true class ids
    probs : [n_samples, n_classes] softmax probabilities
    """
    y_pred = probs.argmax(axis=1)
    classes_int = np.arange(len(label_list))

    print("\n" + "=" * 70)
    print(f"🏁 Evaluation – {name}")
    print("=" * 70)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print(f"\nAccuracy: {acc:.4f}")
    print(f"Exact Match (EM): {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}\n")

    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=label_list))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes_int)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_list)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(f"{name} – Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # Macro AUC (OvR)
    y_true_bin = label_binarize(y_true, classes=classes_int)
    try:
        auc_macro = roc_auc_score(
            y_true_bin, probs, multi_class="ovr", average="macro"
        )
        print(f"Macro AUC (OvR): {auc_macro:.4f}")
    except Exception as e:
        print("⚠️ Could not compute AUC:", e)

    # Top-k accuracy
    try:
        top2 = top_k_accuracy_score(y_true, probs, k=2, labels=classes_int)
        top3 = top_k_accuracy_score(y_true, probs, k=3, labels=classes_int)
        print(f"Top-2 Accuracy: {top2:.4f}")
        print(f"Top-3 Accuracy: {top3:.4f}")
    except Exception as e:
        print("⚠️ Could not compute Top-k accuracy:", e)

    return acc, macro_f1

# =========================================================
# Main
# =========================================================
def main():
    set_all_seeds(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ----------------------------------------
    # 1. Load dataset
    # ----------------------------------------
    print("\n📥 Loading dataset: Annanay/aml_song_lyrics_balanced ...")
    raw_dataset = load_dataset("Annanay/aml_song_lyrics_balanced", split="train")
    df = pd.DataFrame(raw_dataset)[["lyrics", "mood"]]

    print("Dataset shape:", df.shape)
    print("\nLabel distribution:")
    print(df["mood"].value_counts())

    # Label mapping
    label_list = sorted(df["mood"].unique())   # ['anger','calm','happy','sad']
    label2id = {lab: i for i, lab in enumerate(label_list)}
    id2label = {i: lab for lab, i in label2id.items()}
    print("\nLabel mapping:", label2id)

    # Train / test split
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=df["mood"]
    )
    print("\nTrain size:", len(train_df), "Test size:", len(test_df))

    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    test_dataset  = Dataset.from_pandas(test_df.reset_index(drop=True))

    # ----------------------------------------
    # 2. Tokenizer
    # ----------------------------------------
    print("\n🔤 Loading tokenizer:", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(batch):
        texts = batch["lyrics"]
        labels = [label2id[m] for m in batch["mood"]]

        enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )
        enc["labels"] = labels
        return enc

    print("🧹 Tokenizing...")
    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    test_tokenized  = test_dataset.map(tokenize_function, batched=True)

    keep_cols = ["input_ids", "attention_mask", "labels"]
    train_tokenized = train_tokenized.remove_columns(
        [c for c in train_tokenized.column_names if c not in keep_cols]
    )
    test_tokenized = test_tokenized.remove_columns(
        [c for c in test_tokenized.column_names if c not in keep_cols]
    )

    train_tokenized.set_format("torch")
    test_tokenized.set_format("torch")

    train_loader = DataLoader(
        train_tokenized,
        batch_size=TRAIN_BATCH,
        shuffle=True
    )
    test_loader = DataLoader(
        test_tokenized,
        batch_size=EVAL_BATCH,
        shuffle=False
    )

    # ----------------------------------------
    # 3. Model + optimizer + scheduler
    # ----------------------------------------
    print("\n🧠 Loading model:", MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    num_update_steps_per_epoch = max(1, len(train_loader) // GRAD_ACCUM)
    total_steps = num_update_steps_per_epoch * EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)

    print(f"Total optimizer steps: {total_steps}, Warmup steps: {warmup_steps}")

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss()  # dataset is balanced; no class weights

    best_f1 = 0.0
    best_state = None
    global_step = 0

    # ----------------------------------------
    # 4. Training loop
    # ----------------------------------------
    for epoch in range(1, EPOCHS + 1):
        print(f"\n🚀 Epoch {epoch}/{EPOCHS}")
        model.train()
        running_loss = 0.0

        optimizer.zero_grad()
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            logits = outputs.logits
            loss = loss_fn(logits, batch["labels"])
            loss = loss / GRAD_ACCUM
            loss.backward()

            running_loss += loss.item()

            if step % GRAD_ACCUM == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 50 == 0:
                    print(f"  Step {global_step} - loss: {running_loss:.4f}")
                    running_loss = 0.0

        # ---- evaluation at end of epoch ----
        model.eval()
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

                all_probs.append(probs)
                all_labels.append(batch["labels"].cpu().numpy())

        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        acc, macro_f1 = compute_all_metrics(
            name=f"BERT (epoch {epoch})",
            y_true=all_labels,
            y_score=all_probs,
            classes_int=np.arange(len(label_list)),
            label_names=label_list
        ) if False else (None, None)  # placeholder to show structure
        # We’ll compute metrics once more cleanly below,
        # but you can uncomment the above block if you want per-epoch metrics.

        # Simple best-F1 tracking (recompute quickly):
        y_pred_tmp = all_probs.argmax(axis=1)
        macro_f1_epoch = f1_score(all_labels, y_pred_tmp, average="macro")
        print(f"Epoch {epoch} Macro F1: {macro_f1_epoch:.4f}")

        if macro_f1_epoch > best_f1:
            best_f1 = macro_f1_epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"✨ New best model – Macro F1 = {best_f1:.4f}")

    # ----------------------------------------
    # 5. Final evaluation with best model
    # ----------------------------------------
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    print("\n🧪 Final evaluation on test set (best epoch)...")
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(batch["labels"].cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # This call prints *all* TA-required metrics + confusion matrix
    compute_all_metrics(
        name="BERT-base (TF only)",
        y_true=all_labels,
        y_score=all_probs,
        classes_int=np.arange(len(label_list)),
        label_names=label_list
    )

    # Optionally save model & tokenizer
    save_dir = "./bert_lyrics_mood"
    print(f"\n💾 Saving fine-tuned model to {save_dir}")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


# Slight tweak so compute_all_metrics fits with above call signature
def compute_all_metrics(name, y_true, y_score, classes_int, label_names):
    y_pred = y_score.argmax(axis=1)

    print("\n" + "=" * 70)
    print(f"🏁 Evaluation – {name}")
    print("=" * 70)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print(f"\nAccuracy: {acc:.4f}")
    print(f"Exact Match (EM): {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}\n")

    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=label_names))

    cm = confusion_matrix(y_true, y_pred, labels=classes_int)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(f"{name} – Confusion Matrix")
    plt.tight_layout()
    plt.show()

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

    return acc, macro_f1


if __name__ == "__main__":
    main()

# ======================================================================
# 🏁 Evaluation – BERT-base (TF only)
# ======================================================================

# Accuracy: 0.7574
# Exact Match (EM): 0.7574
# Macro F1: 0.7479

# Classification report:
#               precision    recall  f1-score   support

#        anger       0.93      0.99      0.96       610
#         calm       0.84      0.96      0.90       610
#        happy       0.61      0.57      0.59       610
#          sad       0.59      0.51      0.55       610

#     accuracy                           0.76      2440
#    macro avg       0.74      0.76      0.75      2440
# weighted avg       0.74      0.76      0.75      2440

# Macro AUC (OvR): 0.9246
# Top-2 Accuracy: 0.9303
# Top-3 Accuracy: 0.9902