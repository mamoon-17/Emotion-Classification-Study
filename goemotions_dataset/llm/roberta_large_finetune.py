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
# Config (strong LLM setup)
# =========================================================
MODEL_NAME      = "roberta-large"   # stronger than base; reduce to roberta-base if OOM
MAX_LENGTH      = 96                # GoEmotions texts are short
TRAIN_BATCH     = 8                 # keep small for roberta-large
EVAL_BATCH      = 32
EPOCHS          = 6                 # upper bound; early stopping will cut off earlier
LR              = 1e-5
WARMUP_RATIO    = 0.10
WEIGHT_DECAY    = 0.01
SEED            = 42
GRAD_ACCUM      = 2                 # effective batch size ~16
EARLY_STOP_PATIENCE = 2             # stop if val F1 doesn't improve for N epochs

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


def eval_on_loader(model, dataloader, device, num_labels, loss_fn=None):
    model.eval()
    all_probs = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(batch["labels"].cpu().numpy())

            if loss_fn is not None:
                loss = loss_fn(logits, batch["labels"])
                total_loss += loss.item()
                n_batches += 1

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    y_pred = all_probs.argmax(axis=1)
    macro_f1 = f1_score(all_labels, y_pred, average="macro")
    avg_loss = total_loss / n_batches if n_batches > 0 else None

    return all_probs, all_labels, macro_f1, avg_loss


# =========================================================
# Main
# =========================================================
def main():
    set_all_seeds(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ----------------------------------------
    # 1. Load GoEmotions dataset
    # ----------------------------------------
    print("\n📥 Loading dataset: dair-ai/emotion (GoEmotions 6-class) ...")
    raw_dataset = load_dataset("dair-ai/emotion")
    train_ds = raw_dataset["train"]
    val_ds   = raw_dataset["validation"]
    test_ds  = raw_dataset["test"]

    # Convert to DataFrames for convenience
    train_df = pd.DataFrame(train_ds)
    val_df   = pd.DataFrame(val_ds)
    test_df  = pd.DataFrame(test_ds)

    print("Train shape:", train_df.shape)
    print("Val shape:", val_df.shape)
    print("Test shape:", test_df.shape)

    # Label mapping from dataset feature
    label_list = train_ds.features["label"].names  # ['sadness','joy','love','anger','fear','surprise']
    label2id = {lab: i for i, lab in enumerate(label_list)}
    id2label = {i: lab for lab, i in label2id.items()}
    print("\nLabel mapping:", label2id)

    # ----------------------------------------
    # 2. Tokenizer
    # ----------------------------------------
    print("\n🔤 Loading tokenizer:", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(batch):
        texts = batch["text"]
        # labels are already ints 0..5 in the dataset
        labels = batch["label"]

        enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )
        enc["labels"] = labels
        return enc

    print("🧹 Tokenizing...")
    train_tokenized = train_ds.map(tokenize_function, batched=True)
    val_tokenized   = val_ds.map(tokenize_function, batched=True)
    test_tokenized  = test_ds.map(tokenize_function, batched=True)

    keep_cols = ["input_ids", "attention_mask", "labels"]
    train_tokenized = train_tokenized.remove_columns(
        [c for c in train_tokenized.column_names if c not in keep_cols]
    )
    val_tokenized = val_tokenized.remove_columns(
        [c for c in val_tokenized.column_names if c not in keep_cols]
    )
    test_tokenized = test_tokenized.remove_columns(
        [c for c in test_tokenized.column_names if c not in keep_cols]
    )

    train_tokenized.set_format("torch")
    val_tokenized.set_format("torch")
    test_tokenized.set_format("torch")

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_tokenized, batch_size=TRAIN_BATCH, shuffle=True)
    val_loader   = DataLoader(val_tokenized,   batch_size=EVAL_BATCH,  shuffle=False)
    test_loader  = DataLoader(test_tokenized,  batch_size=EVAL_BATCH,  shuffle=False)

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

    # ---- class weights for loss (to help love/surprise) ----
    train_labels = np.array(train_df["label"].values)
    class_counts = np.bincount(train_labels, minlength=len(label_list))
    class_freq   = class_counts / class_counts.sum()
    inv_freq     = 1.0 / np.maximum(class_freq, 1e-8)
    class_weights = inv_freq / inv_freq.mean()
    print("\nClass counts:", class_counts)
    print("Class weights (CE):", class_weights)

    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_t)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    num_update_steps_per_epoch = max(1, len(train_loader) // GRAD_ACCUM)
    total_steps = num_update_steps_per_epoch * EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)

    print(f"\nTotal optimizer steps: {total_steps}, Warmup steps: {warmup_steps}")

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    best_val_f1 = 0.0
    best_state = None
    global_step = 0
    patience_counter = 0

    # ----------------------------------------
    # 4. Training loop with early stopping
    # ----------------------------------------
    for epoch in range(1, EPOCHS + 1):
        print(f"\n🚀 Epoch {epoch}/{EPOCHS}")
        model.train()
        running_loss = 0.0
        steps_in_window = 0

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
            steps_in_window += 1

            if step % GRAD_ACCUM == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # log avg loss over last ~50 update steps
                if global_step % 50 == 0:
                    avg_loss = running_loss / max(1, steps_in_window)
                    print(f"  Step {global_step} - avg loss: {avg_loss:.4f}")
                    running_loss = 0.0
                    steps_in_window = 0

        # ---- validation at end of epoch ----
        val_probs, val_labels, val_f1, val_loss = eval_on_loader(
            model, val_loader, device, len(label_list), loss_fn=loss_fn
        )
        print(f"Epoch {epoch} – Val Macro F1: {val_f1:.4f}, Val Loss: {val_loss:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"✨ New best model – Val Macro F1 = {best_val_f1:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience = {patience_counter}/{EARLY_STOP_PATIENCE}")
            if patience_counter >= EARLY_STOP_PATIENCE:
                print("⏹ Early stopping triggered.")
                break

    # ----------------------------------------
    # 5. Final evaluation on test set
    # ----------------------------------------
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    print("\n🧪 Final evaluation on test set (best validation model)...")
    test_probs, test_labels, _, _ = eval_on_loader(
        model, test_loader, device, len(label_list), loss_fn=loss_fn
    )

    classes_int = np.arange(len(label_list))
    compute_all_metrics(
        name=f"{MODEL_NAME} (GoEmotions 6-class, TF only, tuned)",
        y_true=test_labels,
        y_score=test_probs,
        classes_int=classes_int,
        label_names=label_list
    )

    # Optional: save model & tokenizer
    save_dir = "./roberta_large_goemotions"
    print(f"\n💾 Saving fine-tuned model to {save_dir}")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    main()

# ======================================================================
# 🏁 Evaluation – roberta-large (GoEmotions 6-class, TF only, tuned)
# ======================================================================

# Accuracy: 0.9300
# Exact Match (EM): 0.9300
# Macro F1: 0.8946

# Classification report:
#               precision    recall  f1-score   support

#      sadness       0.97      0.96      0.96       581
#          joy       0.97      0.93      0.95       695
#         love       0.79      0.93      0.85       159
#        anger       0.93      0.92      0.92       275
#         fear       0.90      0.88      0.89       224
#     surprise       0.71      0.88      0.78        66

#     accuracy                           0.93      2000
#    macro avg       0.88      0.92      0.89      2000
# weighted avg       0.93      0.93      0.93      2000


# Macro AUC (OvR): 0.9964
# Top-2 Accuracy: 0.9965
# Top-3 Accuracy: 0.9985