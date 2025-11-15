import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    set_seed,
)

# ============================================================
# Config - tuned for better accuracy
# ============================================================

# Stronger encoder than distilroberta-base
MODEL_NAME = "roberta-base"     # if you get OOM, switch back to "distilroberta-base"

MAX_LENGTH = 384                # more of each lyric is visible to the model
BATCH_SIZE = 16                 # if OOM, change this to 8
EPOCHS = 6                      # more training than before (3 -> 6)
LR = 1e-5                       # lower LR = more stable fine-tuning
WARMUP_RATIO = 0.05             # warmup steps as fraction of total
SEED = 42


def set_all_seeds(seed: int = 42):
    """Make results (roughly) reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


# ============================================================
# Main
# ============================================================

def main():
    set_all_seeds(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------
    print("📥 Loading dataset: Annanay/aml_song_lyrics_balanced ...")
    raw_dataset = load_dataset("Annanay/aml_song_lyrics_balanced", split="train")

    df = pd.DataFrame(raw_dataset)[["lyrics", "mood"]]
    print("Dataset shape:", df.shape)
    print(df["mood"].value_counts())

    # Map mood labels to IDs
    label_list = sorted(df["mood"].unique())         # ['anger','calm','happy','sad']
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}
    print("Label mapping:", label2id)

    # Stratified split: same idea as your LR baseline
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=df["mood"]
    )
    print("Train size:", len(train_df), "Test size:", len(test_df))

    train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))

    # --------------------------------------------------------
    # 2. Tokenizer & preprocessing
    # --------------------------------------------------------
    print("🔤 Loading tokenizer:", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(batch):
        """Tokenize lyrics and attach label IDs."""
        texts = batch["lyrics"]
        labels = [label2id[m] for m in batch["mood"]]

        enc = tokenizer(
            texts,
            padding="max_length",       # pad to fixed length for efficiency
            truncation=True,
            max_length=MAX_LENGTH
        )
        enc["labels"] = labels
        return enc

    print("🧹 Tokenizing...")
    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    test_tokenized = test_dataset.map(tokenize_function, batched=True)

    # Keep only tensors the model needs
    keep_cols = ["input_ids", "attention_mask", "labels"]
    train_tokenized = train_tokenized.remove_columns(
        [c for c in train_tokenized.column_names if c not in keep_cols]
    )
    test_tokenized = test_tokenized.remove_columns(
        [c for c in test_tokenized.column_names if c not in keep_cols]
    )

    train_tokenized.set_format("torch")
    test_tokenized.set_format("torch")

    # Dataloaders
    train_loader = DataLoader(
        train_tokenized,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_tokenized,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # --------------------------------------------------------
    # 3. Model, optimizer, scheduler
    # --------------------------------------------------------
    print("🧠 Loading model:", MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)
    print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}")

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # dataset is perfectly balanced, so basic CE loss is fine
    loss_fn = nn.CrossEntropyLoss()

    best_f1 = 0.0
    best_state_dict = None

    # --------------------------------------------------------
    # 4. Training loop
    # --------------------------------------------------------
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        print(f"\n🚀 Epoch {epoch}/{EPOCHS}")
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            logits = outputs.logits
            loss = loss_fn(logits, batch["labels"])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            if step % 100 == 0 or step == 1:
                avg_loss = total_loss / step
                print(f"  Step {step}/{len(train_loader)} - loss: {avg_loss:.4f}")

        avg_epoch_loss = total_loss / len(train_loader)
        print(f"✅ Epoch {epoch} finished. Avg loss: {avg_epoch_loss:.4f}")

        # ----------------------------------------------------
        # 5. Evaluation after each epoch
        # ----------------------------------------------------
        acc, f1 = evaluate_model(model, test_loader, device)
        print(f"📊 Val Accuracy: {acc:.4f} | Val Macro F1: {f1:.4f}")

        # Keep best model based on macro F1
        if f1 > best_f1:
            best_f1 = f1
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"✨ New best model found! Macro F1 = {best_f1:.4f}")

    # Load best weights if we found a better epoch
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        model.to(device)

    # Final evaluation + classification report
    print("\n🧪 Final evaluation on test set (best epoch)...")
    acc, f1, y_true, y_pred = evaluate_model(
        model, test_loader, device, return_preds=True
    )
    print(f"\n📊 Final Test Accuracy: {acc:.4f}")
    print(f"📊 Final Test Macro F1: {f1:.4f}")

    print("\n📄 Classification Report (LLM - RoBERTa):")
    print(classification_report(y_true, y_pred, target_names=label_list))

    # Save model
    save_path = "./llm_lyrics_mood_roberta"
    print(f"\n💾 Saving fine-tuned model to {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # Quick demo
    example_text = "I feel so alone, crying every night, nothing seems to matter anymore."
    pred_label, probs = predict_mood(example_text, save_path, id2label)
    print("\n🔍 Example inference:")
    print("Text:", example_text)
    print("Predicted mood:", pred_label)
    print("Probabilities:", probs)


# ============================================================
# Evaluation helper
# ============================================================

def evaluate_model(model, dataloader, device, return_preds=False):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    if return_preds:
        return acc, f1, np.array(all_labels), np.array(all_preds)
    return acc, f1


# ============================================================
# Inference helper
# ============================================================

def predict_mood(text: str, model_path: str, id2label: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        pred_id = int(np.argmax(probs))
        return id2label[pred_id], probs


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
