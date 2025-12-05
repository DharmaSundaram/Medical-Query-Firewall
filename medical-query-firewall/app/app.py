# app/app.py
from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import json
from datetime import datetime
from collections import Counter
import logging
from logging.handlers import RotatingFileHandler

# Local imports
from .utils import mask_pii, decision_aggregator, pass_through_llm, RULES_PATH
from .audit_db import init_db, insert_audit, fetch_audits, fetch_audit_by_id, set_reviewer_decision

# base paths
BASE = Path(__file__).resolve().parent.parent

# --- Logging setup (rotating) ---
LOG_DIR = BASE / "logs"
LOG_DIR.mkdir(exist_ok=True)
handler = RotatingFileHandler(LOG_DIR / "server.log", maxBytes=2_000_000, backupCount=3)
logging.basicConfig(level=logging.INFO, handlers=[handler], format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# --- App setup ---
app = FastAPI(title="Medical Query Firewall (MVP)")

# Serve static UI
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")

# Initialize DB
init_db()
logger.info("Audit DB initialized")

# Simple in-memory metrics
METRICS = Counter()

# Simple admin key for demo (change this before presentation)
ADMIN_KEY = "hackathon-demo-key-CHANGE_ME"

def check_admin(x_api_key: str = Header(None)):
    if x_api_key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# Middleware to collect simple request count metric
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    METRICS['requests'] += 1
    response = await call_next(request)
    return response

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = BASE / "static" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))

@app.get("/health")
async def health():
    return {"status": "ok", "requests": METRICS['requests']}

@app.post("/api/chat")
async def chat_endpoint(req: Request):
    body = await req.json()
    text = body.get("text", "")
    session_id = body.get("session_id", "anon")
    user_meta = body.get("meta", {})

    METRICS['last_request'] = datetime.utcnow().isoformat()

    # Step 1: Preprocess & mask PII
    pii_res = mask_pii(text)
    masked = pii_res["masked_text"]

    # Step 2: Decision aggregation
    decision = decision_aggregator(masked)

    # Build explainability info
    explain = {
        "masked_text": masked,
        "pii_detected": pii_res["pii"],
        "matched_rules": [ { "id": r.get("id"), "explanation": r.get("explanation"), "action": r.get("action") } for r in decision.get("matched_rules", []) ],
        "classifier": decision.get("classifier")
    }

    # Insert audit record (persist)
    audit_record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "session_id": session_id,
        "raw_text": text,
        "masked_text": masked,
        "pii": pii_res["pii"],
        "decision": decision.get("decision"),
        "classifier": decision.get("classifier"),
        "matched_rules": [r.get("id") for r in decision.get("matched_rules", [])],
        "block_hits": decision.get("block_hits", []),
        "warn_hits": decision.get("warn_hits", [])
    }
    try:
        audit_id = insert_audit(audit_record)
        explain["audit_id"] = audit_id
    except Exception as e:
        logger.exception("Failed to insert audit: %s", e)
        explain["audit_error"] = str(e)

    # Return responses depending on decision
    if decision["decision"] == "BLOCK":
        METRICS['blocked'] += 1
        safe_response = "Sorry, I can’t assist with that. This request appears unsafe. Please consult a licensed healthcare professional or ask for general information."
        logger.info("BLOCK: %s (audit_id=%s)", masked, explain.get("audit_id"))
        return JSONResponse({
            "decision": "BLOCK",
            "safe_response": safe_response,
            "explain": explain
        })
    elif decision["decision"] == "WARN":
        METRICS['warned'] += 1
        llm_response = pass_through_llm(masked)
        logger.info("WARN: %s (audit_id=%s)", masked, explain.get("audit_id"))
        return JSONResponse({
            "decision": "WARN",
            "llm_response": llm_response,
            "warning": "This query appears risky. Please consult a professional for prescriptions/procedures.",
            "explain": explain
        })
    else:  # ALLOW
        METRICS['allowed'] += 1
        llm_response = pass_through_llm(masked)
        logger.info("ALLOW: %s (audit_id=%s)", masked, explain.get("audit_id"))
        return JSONResponse({
            "decision": "ALLOW",
            "llm_response": llm_response,
            "explain": explain
        })

# ----------------------------
# Admin endpoints (protected)
# ----------------------------

@app.get("/admin/audit", dependencies=[Depends(check_admin)])
async def admin_get_audits(limit: int = 200):
    rows = fetch_audits(limit=limit)
    # parse JSON string fields for readability
    for r in rows:
        for key in ("pii", "classifier_json", "matched_rules", "block_hits", "warn_hits"):
            if r.get(key):
                try:
                    r[key] = json.loads(r[key])
                except Exception:
                    pass
    return {"count": len(rows), "audits": rows}

@app.get("/admin/audit/{aid}", dependencies=[Depends(check_admin)])
async def admin_get_audit(aid: int):
    r = fetch_audit_by_id(aid)
    if not r:
        raise HTTPException(status_code=404, detail="Not found")
    for key in ("pii", "classifier_json", "matched_rules", "block_hits", "warn_hits"):
        if r.get(key):
            try:
                r[key] = json.loads(r[key])
            except Exception:
                pass
    return r

@app.get("/admin/rules", dependencies=[Depends(check_admin)])
async def admin_get_rules():
    with open(RULES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

@app.post("/admin/rules", dependencies=[Depends(check_admin)])
async def admin_update_rules(new_rules: dict):
    if not isinstance(new_rules, list):
        raise HTTPException(status_code=400, detail="rules must be a list")
    with open(RULES_PATH, "w", encoding="utf-8") as f:
        json.dump(new_rules, f, indent=2)
    # try to update in-memory RULES variable in utils (best-effort)
    try:
        from . import utils as _utils
        _utils.RULES = new_rules
    except Exception:
        logger.warning("Could not reload RULES in-memory; server restart may be required.")
    return {"status": "ok", "rules_count": len(new_rules)}

@app.get("/admin/review", dependencies=[Depends(check_admin)])
async def admin_review_queue(limit: int = 200):
    rows = fetch_audits(limit=limit)
    warn_items = []
    for r in rows:
        if r.get("warn_hits"):
            try:
                wh = json.loads(r["warn_hits"])
                if wh:
                    warn_items.append(r)
            except Exception:
                pass
    return {"count": len(warn_items), "warn_items": warn_items}

@app.post("/admin/review/{aid}", dependencies=[Depends(check_admin)])
async def admin_review_decision(aid: int, action: str):
    # action: "allow" / "block" / "ignore"
    if action not in ("allow", "block", "ignore"):
        raise HTTPException(status_code=400, detail="invalid action")
    set_reviewer_decision(aid, action)
    logger.info("Reviewer decision set: audit_id=%s action=%s", aid, action)
    return {"status": "ok", "audit_id": aid, "action": action}

@app.get("/metrics", dependencies=[Depends(check_admin)])
async def metrics():
    return {
        "requests": METRICS.get('requests', 0),
        "allowed": METRICS.get('allowed', 0),
        "blocked": METRICS.get('blocked', 0),
        "warned": METRICS.get('warned', 0),
        "last_request": METRICS.get('last_request')
    }
