# train_albert_or_bert.py

import os
import json
import argparse
import numpy as np
from collections import Counter

import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import evaluate

print(f"Using Transformers version: {transformers.__version__}")

# --------------------
# Data loader
# --------------------
def load_style_detection_data(data_dir, truth_dir):
    """
    Load paragraph-level texts and paragraph-authors labels from PAN-style files.
    Expects:
      data_dir/problem-XXXX.txt
      truth_dir/truth-problem-XXXX.json
    The truth JSON should contain "paragraph-authors" (list of ints).
    Returns lists: texts, labels
    """
    all_texts = []
    all_labels = []
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not os.path.isdir(truth_dir):
        raise FileNotFoundError(f"Truth directory not found: {truth_dir}")

    truth_files = [f for f in os.listdir(truth_dir) if f.endswith('.json')]
    truth_files.sort()

    for truth_filename in truth_files:
        # Try to robustly extract problem id
        parts = truth_filename.replace(".json", "").split("-")
        # e.g., truth-problem-00001
        problem_id = parts[-1]
        text_filename = f"problem-{problem_id}.txt"
        truth_path = os.path.join(truth_dir, truth_filename)
        text_path = os.path.join(data_dir, text_filename)

        if not os.path.exists(text_path):
            print(f"Warning: Missing text file for {truth_filename} -> expected {text_filename}. Skipping.")
            continue

        with open(truth_path, 'r', encoding='utf-8') as f:
            truth_data = json.load(f)
            # expected key: "paragraph-authors"
            if "paragraph-authors" not in truth_data:
                print(f"Warning: 'paragraph-authors' not found in {truth_filename}. Skipping.")
                continue
            labels = [int(author_id) - 1 for author_id in truth_data["paragraph-authors"]]

        with open(text_path, 'r', encoding='utf-8') as f:
            # assume paragraphs are separated by newline(s); keep non-empty lines
            texts = [line.strip() for line in f if line.strip()]

        if len(texts) != len(labels):
            print(f"Warning: Mismatch in {text_filename}: paragraphs={len(texts)} labels={len(labels)}. Skipping.")
            continue

        all_texts.extend(texts)
        all_labels.extend(labels)

    return all_texts, all_labels


# --------------------
# Main
# --------------------
def main():
    parser = argparse.ArgumentParser(description="Train a style-detection classifier (ALBERT or BERT).")
    parser.add_argument("--model_name", type=str, default="albert", choices=["albert", "bert"],
                        help="Model architecture to use: 'albert' or 'bert'")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing problem-*.txt files")
    parser.add_argument("--truth_dir", type=str, default="./truth", help="Directory containing truth-*.json files")
    parser.add_argument("--debug", action="store_true", help="Use small subset for quick debugging (1000 samples)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Per-device batch size")
    args = parser.parse_args()

    print(f"🚀 Running with model: {args.model_name.upper()}")
    model_checkpoint = "albert-base-v2" if args.model_name == "albert" else "bert-base-uncased"
    output_dir = f"./models/{args.model_name}_style_detector"
    logging_dir = f"./logs/{args.model_name}"

    # Load data
    texts, labels = load_style_detection_data(args.data_dir, args.truth_dir)
    if not texts:
        print("❌ No data loaded. Check data_dir & truth_dir contents.")
        return

    print(f"🧠 Loaded {len(texts)} paragraphs.")

    # Build Dataset
    ds = Dataset.from_dict({"text": texts, "labels": labels})

    # Optional debug subset
    if args.debug:
        n_debug = min(1000, len(ds))
        ds = ds.shuffle(seed=42).select(range(n_debug))
        print(f"🔧 Debug mode ON: using {len(ds)} samples for quick run.")

    # Determine number of labels automatically (max label + 1)
    unique_labels = sorted(set(labels))
    num_labels = max(unique_labels) + 1
    print(f"🔢 Detected label ids: {unique_labels} -> num_labels = {num_labels}")

    # Tokenizer & tokenization
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    def tokenize_fn(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    # Split into train/validation/test (80/10/10)
    split = tokenized.train_test_split(test_size=0.2, seed=42)
    train_ds = split["train"]
    test_temp = split["test"]
    val_test_split = test_temp.train_test_split(test_size=0.5, seed=42)
    val_ds = val_test_split["train"]
    test_ds = val_test_split["test"]

    print("📊 Dataset split sizes:")
    print(f"   Training:   {len(train_ds)} samples")
    print(f"   Validation: {len(val_ds)} samples")
    print(f"   Test:       {len(test_ds)} samples")

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)

    # Metrics
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_metric.compute(predictions=preds, references=labels)["accuracy"],
            "f1": f1_metric.compute(predictions=preds, references=labels, average="weighted")["f1"]
        }

    # TrainingArguments (kept simple to avoid compatibility issues)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        save_strategy="epoch",
        logging_dir=logging_dir,
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train
    trainer.train()

    # Final evaluation on test set
    print("\n🧪 Evaluating on the held-out test set...")
    test_results = trainer.evaluate(eval_dataset=test_ds)
    print("📊 Final test results:", test_results)

    # Save
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Trained model saved to: {output_dir}")

    # Predictions on test set -> count unique authors and show distribution
    print("\n🔍 Running predictions on test set to detect unique authors...")
    preds_output = trainer.predict(test_ds)
    preds = np.argmax(preds_output.predictions, axis=-1)

    author_counts = Counter(preds)
    print("\n🧩 Detected author style distribution (predicted):")
    for author_id, count in sorted(author_counts.items()):
        print(f"   Author {author_id}: {count} paragraphs")

    unique_authors = len(author_counts)
    print(f"\nTotal unique authors detected: {unique_authors}")

if __name__ == "__main__":
    main()
