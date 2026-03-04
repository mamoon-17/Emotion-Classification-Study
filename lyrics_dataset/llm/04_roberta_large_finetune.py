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
# Config - RoBERTa-Large + Grad Accumulation
# ============================================================

MODEL_NAME = "roberta-large"      # very strong encoder
MAX_LENGTH = 320                  # a bit shorter to fit in VRAM
BATCH_SIZE = 2                    # micro-batch size
GRAD_ACCUM_STEPS = 8              # 2 * 8 = effective batch size 16
EPOCHS = 6
LR = 1e-5
WARMUP_RATIO = 0.05
SEED = 42


def set_all_seeds(seed: int = 42):
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

    label_list = sorted(df["mood"].unique())         # ['anger','calm','happy','sad']
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}
    print("Label mapping:", label2id)

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
    test_tokenized = test_dataset.map(tokenize_function, batched=True)

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
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_tokenized,
        batch_size=16,  # eval can be larger batch
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

    # total_steps is in terms of optimizer steps, not micro-batches
    num_update_steps_per_epoch = len(train_loader) // GRAD_ACCUM_STEPS
    total_steps = num_update_steps_per_epoch * EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)
    print(f"Total optimizer steps: {total_steps}, Warmup steps: {warmup_steps}")

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_f1 = 0.0
    best_state_dict = None

    # --------------------------------------------------------
    # 4. Training loop with gradient accumulation
    # --------------------------------------------------------
    global_step = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        running_loss = 0.0
        steps_in_epoch = 0

        print(f"\n🚀 Epoch {epoch}/{EPOCHS}")
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            logits = outputs.logits
            loss = loss_fn(logits, batch["labels"])
            loss = loss / GRAD_ACCUM_STEPS   # scale loss

            loss.backward()
            running_loss += loss.item()
            steps_in_epoch += 1

            if step % GRAD_ACCUM_STEPS == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                avg_loss = running_loss
                total_loss += running_loss
                running_loss = 0.0

                if global_step % 50 == 0:
                    print(f"  OptStep {global_step}/{total_steps} - loss: {avg_loss:.4f}")

        avg_epoch_loss = total_loss / max(1, (steps_in_epoch / GRAD_ACCUM_STEPS))
        print(f"✅ Epoch {epoch} finished. Avg loss: {avg_epoch_loss:.4f}")

        acc, f1 = evaluate_model(model, test_loader, device)
        print(f"📊 Val/Test Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"✨ New best model found! Macro F1 = {best_f1:.4f}")

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        model.to(device)

    print("\n🧪 Final evaluation on test set (best epoch)...")
    acc, f1, y_true, y_pred = evaluate_model(
        model, test_loader, device, return_preds=True
    )
    print(f"\n📊 Final Test Accuracy: {acc:.4f}")
    print(f"📊 Final Test Macro F1: {f1:.4f}")

    print("\n📄 Classification Report (LLM - RoBERTa-Large):")
    print(classification_report(y_true, y_pred, target_names=label_list))

    save_path = "./llm_lyrics_mood_roberta_large"
    print(f"\n💾 Saving fine-tuned model to {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

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


if __name__ == "__main__":
    main()

# Using device: cuda
# 📥 Loading dataset: Annanay/aml_song_lyrics_balanced ...
# /usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
# The secret HF_TOKEN does not exist in your Colab secrets.
# To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
# You will be able to reuse this secret in all of your notebooks.
# Please note that authentication is recommended but still optional to access public models or datasets.
#   warnings.warn(
# README.md: 100%
#  25.0/25.0 [00:00<00:00, 2.38kB/s]
# training.csv: 100%
#  29.6M/29.6M [00:02<00:00, 25.2MB/s]
# test.csv: 
#  4.54M/? [00:00<00:00, 95.5MB/s]
# Generating train split: 100%
#  12196/12196 [00:00<00:00, 26887.72 examples/s]
# Generating test split: 100%
#  2539/2539 [00:00<00:00, 32834.88 examples/s]
# Dataset shape: (12196, 2)
# mood
# anger    3049
# sad      3049
# happy    3049
# calm     3049
# Name: count, dtype: int64
# Label mapping: {'anger': 0, 'calm': 1, 'happy': 2, 'sad': 3}
# Train size: 9756 Test size: 2440
# 🔤 Loading tokenizer: roberta-large
# tokenizer_config.json: 100%
#  25.0/25.0 [00:00<00:00, 2.51kB/s]
# config.json: 100%
#  482/482 [00:00<00:00, 55.9kB/s]
# vocab.json: 100%
#  899k/899k [00:00<00:00, 24.7MB/s]
# merges.txt: 100%
#  456k/456k [00:00<00:00, 2.80MB/s]
# tokenizer.json: 100%
#  1.36M/1.36M [00:00<00:00, 4.14MB/s]
# 🧹 Tokenizing...
# Map: 100%
#  9756/9756 [00:16<00:00, 524.58 examples/s]
# Map: 100%
#  2440/2440 [00:03<00:00, 709.80 examples/s]
# 🧠 Loading model: roberta-large
# model.safetensors: 100%
#  1.42G/1.42G [00:15<00:00, 244MB/s]
# Some weights of RobertaForSequenceClassification were not initialized from the model checkpoint at roberta-large and are newly initialized: ['classifier.dense.bias', 'classifier.dense.weight', 'classifier.out_proj.bias', 'classifier.out_proj.weight']
# You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
# Total optimizer steps: 3654, Warmup steps: 182

# 🚀 Epoch 1/6
#   OptStep 50/3654 - loss: 1.3698
#   OptStep 100/3654 - loss: 1.5228
#   OptStep 150/3654 - loss: 1.4315
#   OptStep 200/3654 - loss: 1.3005
#   OptStep 250/3654 - loss: 1.1778
#   OptStep 300/3654 - loss: 1.2273
#   OptStep 350/3654 - loss: 1.1676
#   OptStep 400/3654 - loss: 1.3530
#   OptStep 450/3654 - loss: 1.0022
#   OptStep 500/3654 - loss: 1.2675
#   OptStep 550/3654 - loss: 1.1530
#   OptStep 600/3654 - loss: 1.0444
# ✅ Epoch 1 finished. Avg loss: 1.3265
# 📊 Val/Test Accuracy: 0.4873 | Macro F1: 0.4809
# ✨ New best model found! Macro F1 = 0.4809

# 🚀 Epoch 2/6
#   OptStep 650/3654 - loss: 1.2677
#   OptStep 700/3654 - loss: 1.5552
#   OptStep 750/3654 - loss: 1.0895
#   OptStep 800/3654 - loss: 0.9564
#   OptStep 850/3654 - loss: 1.3816
#   OptStep 900/3654 - loss: 0.9616
#   OptStep 950/3654 - loss: 1.0506
#   OptStep 1000/3654 - loss: 0.9041
#   OptStep 1050/3654 - loss: 1.0278
#   OptStep 1100/3654 - loss: 1.0520
#   OptStep 1150/3654 - loss: 0.8496
#   OptStep 1200/3654 - loss: 1.0168
# ✅ Epoch 2 finished. Avg loss: 1.0909
# 📊 Val/Test Accuracy: 0.5922 | Macro F1: 0.5692
# ✨ New best model found! Macro F1 = 0.5692

# 🚀 Epoch 3/6
#   OptStep 1250/3654 - loss: 1.0218
#   OptStep 1300/3654 - loss: 0.7124
#   OptStep 1350/3654 - loss: 1.1994
#   OptStep 1400/3654 - loss: 0.9564
#   OptStep 1450/3654 - loss: 0.8014
#   OptStep 1500/3654 - loss: 1.0875
#   OptStep 1550/3654 - loss: 0.8864
#   OptStep 1600/3654 - loss: 0.6563
#   OptStep 1650/3654 - loss: 0.9668
#   OptStep 1700/3654 - loss: 0.8754
#   OptStep 1750/3654 - loss: 0.6338
#   OptStep 1800/3654 - loss: 0.8465
# ✅ Epoch 3 finished. Avg loss: 0.8417
# 📊 Val/Test Accuracy: 0.6869 | Macro F1: 0.6767
# ✨ New best model found! Macro F1 = 0.6767

# 🚀 Epoch 4/6
#   OptStep 1850/3654 - loss: 0.7703
#   OptStep 1900/3654 - loss: 0.5279
#   OptStep 1950/3654 - loss: 0.7888
#   OptStep 2000/3654 - loss: 0.7425
#   OptStep 2050/3654 - loss: 0.7388
#   OptStep 2100/3654 - loss: 0.6035
#   OptStep 2150/3654 - loss: 0.7116
#   OptStep 2200/3654 - loss: 0.4512
#   OptStep 2250/3654 - loss: 0.6254
#   OptStep 2300/3654 - loss: 0.8265
#   OptStep 2350/3654 - loss: 0.7701
#   OptStep 2400/3654 - loss: 0.6939
# ✅ Epoch 4 finished. Avg loss: 0.6481
# 📊 Val/Test Accuracy: 0.7275 | Macro F1: 0.7100
# ✨ New best model found! Macro F1 = 0.7100

# 🚀 Epoch 5/6
#   OptStep 2450/3654 - loss: 0.4959
#   OptStep 2500/3654 - loss: 0.4758
#   OptStep 2550/3654 - loss: 0.5021
#   OptStep 2600/3654 - loss: 0.4841
#   OptStep 2650/3654 - loss: 0.6338
#   OptStep 2700/3654 - loss: 0.5570
#   OptStep 2750/3654 - loss: 0.5532
#   OptStep 2800/3654 - loss: 0.5626
#   OptStep 2850/3654 - loss: 0.6014
#   OptStep 2900/3654 - loss: 0.4672
#   OptStep 2950/3654 - loss: 0.5586
#   OptStep 3000/3654 - loss: 0.4868
#   OptStep 3050/3654 - loss: 0.3736
# ✅ Epoch 5 finished. Avg loss: 0.5447
# 📊 Val/Test Accuracy: 0.7455 | Macro F1: 0.7313
# ✨ New best model found! Macro F1 = 0.7313

# 🚀 Epoch 6/6
#   OptStep 3100/3654 - loss: 0.5036
#   OptStep 3150/3654 - loss: 0.4252
#   OptStep 3200/3654 - loss: 0.6451
#   OptStep 3250/3654 - loss: 0.4045
#   OptStep 3300/3654 - loss: 0.4676
#   OptStep 3350/3654 - loss: 0.5553
#   OptStep 3400/3654 - loss: 0.3817
#   OptStep 3450/3654 - loss: 0.4716
#   OptStep 3500/3654 - loss: 0.6046
#   OptStep 3550/3654 - loss: 0.7389
#   OptStep 3600/3654 - loss: 0.4699
#   OptStep 3650/3654 - loss: 0.4988
# ✅ Epoch 6 finished. Avg loss: 0.4808
# 📊 Val/Test Accuracy: 0.7500 | Macro F1: 0.7374
# ✨ New best model found! Macro F1 = 0.7374

# 🧪 Final evaluation on test set (best epoch)...

# 📊 Final Test Accuracy: 0.7500
# 📊 Final Test Macro F1: 0.7374

# 📄 Classification Report (LLM - RoBERTa-Large):
#               precision    recall  f1-score   support

#        anger       0.88      1.00      0.93       610
#         calm       0.82      0.95      0.88       610
#        happy       0.62      0.51      0.56       610
#          sad       0.61      0.54      0.57       610

#     accuracy                           0.75      2440
#    macro avg       0.73      0.75      0.74      2440
# weighted avg       0.73      0.75      0.74      2440


# 💾 Saving fine-tuned model to ./llm_lyrics_mood_roberta_large

# 🔍 Example inference:
# Text: I feel so alone, crying every night, nothing seems to matter anymore.
# Predicted mood: anger
# Probabilities: [0.7801424  0.02558424 0.08475222 0.10952104]