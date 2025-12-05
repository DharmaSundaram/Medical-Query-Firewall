# app/utils.py
import re
import json
import joblib
from pathlib import Path
from typing import Dict, Any, List

# Load rules once
RULES_PATH = Path("rules/rules.json")
with open(RULES_PATH, "r", encoding="utf-8") as f:
    RULES = json.load(f)

# Load classifier pipeline (tfidf + clf) which must implement predict_proba
MODEL_PATH = Path("models/classifier.joblib")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Classifier not found at {MODEL_PATH}. Run train_and_eval.py first.")
clf = joblib.load(MODEL_PATH)

# Simple PII regexes (MVP): email, phone (10 digits), Aadhaar-like (12 digits)
PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9.\-+_]+@[a-zA-Z0-9.\-+_]+\.[a-zA-Z]+"),
    "phone": re.compile(r"\b\d{10}\b"),
    "aadhaar": re.compile(r"\b\d{12}\b")
}

def mask_pii(text: str) -> Dict[str, Any]:
    """Mask detected PII and return masked text + list of pii types."""
    pii_found = []
    masked = text
    for name, patt in PII_PATTERNS.items():
        if patt.search(masked):
            masked = patt.sub(f"[{name.upper()}]", masked)
            pii_found.append(name)
    return {"masked_text": masked, "pii": pii_found}

def match_rules(text: str) -> List[Dict[str, Any]]:
    """Return list of rules that match the input (case-insensitive)."""
    matches = []
    for r in RULES:
        patt = re.compile(r["pattern"], re.IGNORECASE)
        if patt.search(text):
            matches.append(r)
    return matches

def classify_text(text: str) -> Dict[str, Any]:
    """Return predicted label and a risk score (mapped from probabilities)."""
    proba = clf.predict_proba([text])[0]
    labels = list(clf.classes_)
    top_idx = int(proba.argmax())
    label = labels[top_idx]
    score = float(proba[top_idx])
    proba_dict = dict(zip(labels, map(float, proba)))
    return {"label": label, "score": score, "proba": proba_dict}

# -------------------------
# Class-specific thresholds
# -------------------------
# Block thresholds: if probability for this class >= value => immediate BLOCK (if not overridden by rule)
# Warn thresholds: if probability >= warn but < block => WARN
# Tune these thresholds based on eval_report.json (precision/recall tradeoff)
CLASS_THRESHOLDS = {
    "self_harm": 0.40,            # lower blockade for self-harm (safety-critical)
    "prescription_request": 0.60,
    "procedural": 0.55
}
WARN_THRESHOLDS = {
    "self_harm": 0.25,
    "prescription_request": 0.40,
    "procedural": 0.35
}

def decision_aggregator(text: str) -> Dict[str, Any]:
    """
    Combines rules + classifier probabilities -> decision.
    Priority order:
      1) Rules with severity HIGH and action BLOCK -> BLOCK (immediate)
      2) Class-specific probability thresholds:
         - If any hazardous class probability >= block threshold -> BLOCK
         - Else if any hazardous class probability >= warn threshold -> WARN
      3) Fallback to top-class probability:
         - If top prob >= 0.85 -> BLOCK
         - If 0.5 <= top prob < 0.85 -> WARN
         - Else -> ALLOW
    Returns dict with 'decision', 'matched_rules', and 'classifier' (with proba dict).
    """
    # 1) Rule matching (rules always have highest priority)
    r_matches = match_rules(text)
    if any(r["severity"].upper() == "HIGH" and r["action"].upper() == "BLOCK" for r in r_matches):
        clf_res = classify_text(text)  # include classifier info for explainability
        return {"decision": "BLOCK", "matched_rules": r_matches, "classifier": clf_res}

    # 2) Classifier probabilities
    clf_res = classify_text(text)
    proba = clf_res.get("proba", {})
    # Evaluate hazardous classes
    block_hits = []
    warn_hits = []
    for cls_name, blk_thr in CLASS_THRESHOLDS.items():
        prob = proba.get(cls_name, 0.0)
        if prob >= blk_thr:
            block_hits.append({"class": cls_name, "prob": prob, "threshold": blk_thr})
        else:
            warn_thr = WARN_THRESHOLDS.get(cls_name, blk_thr * 0.6)  # fallback if warn not provided
            if prob >= warn_thr:
                warn_hits.append({"class": cls_name, "prob": prob, "threshold": warn_thr})

    if block_hits:
        # If rules also matched (non-high severity), include them in explanation
        return {"decision": "BLOCK", "matched_rules": r_matches, "classifier": clf_res, "block_hits": block_hits}
    if warn_hits:
        return {"decision": "WARN", "matched_rules": r_matches, "classifier": clf_res, "warn_hits": warn_hits}

    # 3) Fallback to previous top-label logic if no class-specific thresholds triggered
    top_score = clf_res["score"]
    if top_score >= 0.85:
        return {"decision": "BLOCK", "matched_rules": r_matches, "classifier": clf_res}
    elif top_score >= 0.5:
        return {"decision": "WARN", "matched_rules": r_matches, "classifier": clf_res}
    else:
        return {"decision": "ALLOW", "matched_rules": r_matches, "classifier": clf_res}

# Mock LLM (MVP) - replace with real LLM/inference API in production
KB = {
    "What are symptoms of flu?": "Common flu symptoms include fever, cough, sore throat, muscle aches. Seek medical care if breathing difficulty occurs.",
    "What is a normal blood pressure?": "Normal blood pressure is around 120/80 mmHg, but targets depend on age and comorbidities."
}

def pass_through_llm(text: str) -> str:
    """Return helpful general answer. In real system, call sanitized LLM API here."""
    for k in KB:
        if k.lower() in text.lower():
            return KB[k]
    return "I can provide general medical information, but not personalized prescriptions. Consult a licensed provider for treatment decisions."
