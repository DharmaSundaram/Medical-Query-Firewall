# train_and_eval.py
import os
import random
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import joblib
import warnings

# Paths
DATA_CSV = Path("data/sample_data.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "classifier.joblib"
EVAL_PATH = MODEL_DIR / "eval_report.json"

# 1) Build or load dataset
def build_synthetic_dataset(n_per_class=120):
    """
    Create a synthetic dataset using templates per label.
    Returns a DataFrame with columns 'text' and 'label'.
    """
    templates = {
        "general_info": [
            "What are the symptoms of {cond}?",
            "What is the normal range for {measure}?",
            "Where can I find a clinic for {cond}?",
            "Tell me about {cond} causes and treatment.",
            "How to prevent {cond}?"
        ],
        "prescription_request": [
            "Can I get {med} without prescription?",
            "Give me the dosage for {med}.",
            "How much {med} should I take for {cond}?",
            "I need {med}, where to buy without a prescription?",
            "What is the proper dose of {med} for an adult?"
        ],
        "procedural": [
            "How to inject {med} at home?",
            "How to suture a wound by myself?",
            "How to start an IV at home?",
            "Step by step to do {procedure} at home",
            "How to perform {procedure} on my child?"
        ],
        "self_harm": [
            "I want to hurt myself",
            "How to kill myself?",
            "I don't want to live anymore",
            "Tell me ways to self harm",
            "How to overdose safely?"
        ]
    }

    # small lists for placeholders
    conditions = ["flu", "diabetes", "high blood pressure", "migraine", "urinary infection"]
    measures = ["blood pressure", "heart rate", "cholesterol level"]
    meds = ["amoxicillin", "oxycontin", "ibuprofen", "insulin", "azithromycin"]
    procedures = ["inject insulin", "administer IV", "suture", "intubate"]
    
    rows = []
    for label, tpls in templates.items():
        for _ in range(n_per_class):
            tpl = random.choice(tpls)
            text = tpl.format(
                cond=random.choice(conditions),
                measure=random.choice(measures),
                med=random.choice(meds),
                procedure=random.choice(procedures)
            )
            # small paraphrase: add few filler tokens randomly
            if random.random() < 0.25:
                text = text + " please help"
            if random.random() < 0.15:
                text = " ".join(random.sample(text.split(), len(text.split())))
            rows.append({"text": text, "label": label})
    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

if DATA_CSV.exists():
    print(f"Loading dataset from {DATA_CSV}")
    df = pd.read_csv(DATA_CSV)
else:
    print("No CSV found — building synthetic dataset")
    df = build_synthetic_dataset(n_per_class=120)
    os.makedirs(DATA_CSV.parent, exist_ok=True)
    df.to_csv(DATA_CSV, index=False)
    print(f"Synthetic dataset saved to {DATA_CSV}")

print("Dataset class distribution:")
print(df['label'].value_counts())

# 2) Train / test split (stratified with safety checks)
X = df['text'].values
y = df['label'].values

# ensure we have enough examples for stratified split:
from collections import Counter
counts = Counter(y)
n_classes = len(counts)
n_samples = len(y)

print(f"[INFO] dataset samples={n_samples}, classes={n_classes}, class_counts={dict(counts)}")

# choose a target test fraction
target_frac = 0.20  # try 20% test size

# convert to absolute number of test samples (at least 1)
test_n = max(1, int(round(n_samples * target_frac)))

# if using stratify, test set must have at least one sample per class
if test_n < n_classes:
    # If dataset is too small -> regenerate a larger synthetic dataset
    print(f"[WARN] computed test_n={test_n} < n_classes={n_classes}. Regenerating synthetic dataset with more examples...")
    df = build_synthetic_dataset(n_per_class=120)
    df.to_csv(DATA_CSV, index=False)
    X = df['text'].values
    y = df['label'].values
    counts = Counter(y)
    n_classes = len(counts)
    n_samples = len(y)
    print(f"[INFO] new dataset samples={n_samples}, classes={n_classes}, class_counts={dict(counts)}")
    # recompute test_n
    test_n = max(1, int(round(n_samples * target_frac)))

# Now use stratified split if every class has at least 2 samples
if min(counts.values()) >= 2:
    stratify = y
else:
    stratify = None
    print("[WARN] Some classes have < 2 examples; proceeding without stratify (not recommended for final eval).")

# If test_n is still less than n_classes (very unlikely after regeneration), force a proportion
if test_n < n_classes:
    # pick a fraction that guarantees at least n_classes examples in test
    frac = float(n_classes + 1) / float(n_samples)
    print(f"[WARN] Forcing test fraction to {frac:.2f} to ensure one sample per class.")
    test_size = frac
else:
    # use a fractional test_size for train_test_split (safer than integer)
    test_size = float(test_n) / float(n_samples)

print(f"[INFO] Using test_size={test_size} (test_n approximately {int(round(test_size * n_samples))}) stratify={'yes' if stratify is not None else 'no'}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=stratify
)