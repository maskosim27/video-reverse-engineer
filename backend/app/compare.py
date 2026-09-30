"""§42 Iterative Recreation: bandingkan video ASLI vs video HASIL GENERATE.

Alur: user upload generated video seperti project biasa -> compare(original_id,
generated_id) -> AI menilai 6 similarity (0-100, estimated) + improved prompt.
Setiap compare disimpan sebagai recreation; parent_id merangkai iterasi.

Interface:
  compare(original_id, generated_id, provider, parent_id="") -> dict recreation
  list_recreations(original_id) -> [...]
"""
import json
import re
import uuid

from . import ai, db, skills

COMPARE_SYSTEM = """You compare an ORIGINAL video analysis against a GENERATED (AI-recreated) video analysis.
HARD RULES: scores are ESTIMATES from observable evidence only. Use
estimated/inferred/likely. If a dimension cannot be judged, use null with reason.
Never claim knowledge of any original prompt.
Output ONLY a JSON object (no markdown fences), exactly:
{"visual_similarity": 0-100|null, "motion_similarity": 0-100|null,
 "composition_similarity": 0-100|null, "color_similarity": 0-100|null,
 "camera_similarity": 0-100|null, "character_similarity": 0-100|null,
 "score_notes": {"visual_similarity": "one-line reason", ...},
 "differences": ["concrete visible difference 1", ...],
 "improved_prompt": "revised generation prompt in English addressing the differences",
 "notes": "any uncertainty"}"""

DIMENSIONS = ["visual_similarity", "motion_similarity", "composition_similarity",
              "color_similarity", "camera_similarity", "character_similarity"]


def _summarize(job: dict) -> str:
    parts = [f"METADATA: {json.dumps(job.get('meta'))}",
             f"SCENES: {json.dumps(job.get('scenes'))}"]
    an = job.get("analysis") or {}
    if an:
        parts.append(f"MOTION: {json.dumps(an.get('motion'))}")
        parts.append(f"COLOR: {json.dumps(an.get('color'))}")
        parts.append(f"LIGHTING: {json.dumps(an.get('lighting'))}")
        parts.append(f"EDITING: {json.dumps(an.get('editing'))}")
    if job.get("transcript"):
        parts.append("TRANSCRIPT: " + " ".join(s["text"] for s in job["transcript"][:40]))
    if job.get("report_md"):
        parts.append("AI REPORT (dipotong):\n" + job["report_md"][:6000])
    return "\n\n".join(parts)


def compare(original_id: str, generated_id: str, provider: str,
            parent_id: str = "") -> dict:
    orig = db.get_job(original_id)
    gen = db.get_job(generated_id)
    if not orig or orig["status"] != "completed":
        raise ValueError("original_id belum completed")
    if not gen or gen["status"] != "completed":
        raise ValueError("generated_id belum completed")
    if provider not in ai.PROVIDERS or provider == "mock":
        raise ValueError("compare butuh provider AI sungguhan")
    import main as _m
    from . import config as _c
    cfg = _m.effective_ai(provider)
    raw = db.get_setting(f"model_{provider}") or ""
    model = raw or ai.MODEL_PRESETS.get(provider, {}).get("standard", "") or _c.AI_MODEL
    user_text = ("=== ORIGINAL VIDEO ANALYSIS ===\n" + _summarize(orig) +
                 "\n\n=== GENERATED VIDEO ANALYSIS ===\n" + _summarize(gen))
    raw = ai.PROVIDERS[provider].chat(
        api_key=cfg["api_key"], model=model,
        system=COMPARE_SYSTEM,
        user_parts=[{"type": "text", "text": user_text}],
        temperature=0.2, max_tokens=cfg["max_tokens"], base_url=cfg["base_url"])
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        scores = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        scores = {}
    if not isinstance(scores, dict) or "improved_prompt" not in scores:
        scores = {"raw": raw[:4000], "improved_prompt": "",
                  "notes": "AI tidak mengembalikan JSON valid — lihat 'raw'."}
    rec = {"id": uuid.uuid4().hex[:12], "original_id": original_id,
           "generated_id": generated_id, "parent_id": parent_id,
           "created_at": db.now(), "scores": scores,
           "improved_prompt": scores.get("improved_prompt", ""),
           "report_md": raw[:20000]}
    db.create_recreation(rec)
    return rec


def list_recreations(original_id: str) -> list:
    rows = db.list_recreations(original_id)
    out = []
    for r in rows:
        d = dict(r)
        d["scores"] = json.loads(d["scores"]) if d["scores"] else {}
        out.append(d)
    return out
