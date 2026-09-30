"""Multi-AI: analisis yang sama dijalankan di beberapa provider, lalu consensus.

- run_multi(job_id, providers, mode, judge): background thread; memakai data
  tersimpan (frames/meta/transcript/ocr/analysis) tanpa memproses ulang video.
  Hasil -> job.multi = {"providers": {pid: {status, report|error}},
                        "consensus": {status, report|error, judge}}.
- Consensus: judge provider menggabungkan laporan -> AGREEMENTS / DISAGREEMENTS /
  MERGED best-estimate, tetap dengan aturan §43 (tidak mengklaim prompt asli).
"""
import logging
import threading
from pathlib import Path

from . import ai, config, db, skills

log = logging.getLogger("vre.multi")

CONSENSUS_SYSTEM = """You merge multiple independent video analyses into one best-estimate report.
HARD RULES: never claim to know the original prompt — every reconstruction is
"estimated from observable characteristics". Use estimated/inferred/reconstructed/likely.
Separate [Observed] / [Inferred] / [Speculation]. For anything unknowable write
exactly: "Unknown / Cannot determine from available evidence."
Output Markdown:
# AI CONSENSUS
## Agreements (klaim yang didukung >=2 analisis)
## Disagreements (klaim bertentangan + mana yang bukti paling kuat)
## Merged Best-Estimate (laporan gabungan, tiap klaim berlabel confidence High/Medium/Low)
## Recommended Prompt (rekonstruksi gabungan, English)"""


def _provider_cfg(provider_id: str) -> dict:
    from . import config as _c  # hindari circular di level modul
    import main as _m
    return _m.effective_ai(provider_id)


def run_multi(job_id: str, providers: list, mode: str, judge: str = ""):
    job = db.get_job(job_id)
    if not job or job["status"] != "completed":
        return
    frame_dir = config.FRAME_DIR / job_id
    text, parts = ai.build_messages(job["meta"], job["scenes"], job["frames"],
                                    frame_dir, mode, job.get("transcript"),
                                    job.get("ocr"), job.get("analysis"))
    system = skills.build_system_prompt()
    results: dict = {"providers": {}, "consensus": {"status": "pending"}}
    db.update_job(job_id, multi=results)
    ok_reports = {}
    for pid in providers:
        if pid not in ai.PROVIDERS or pid == "mock":
            results["providers"][pid] = {"status": "failed",
                                         "error": "provider tidak dikenal/mock"}
            continue
        try:
            cfg = _provider_cfg(pid)
            raw = db.get_setting(f"model_{pid}") or ""
            model = raw or ai.MODEL_PRESETS.get(pid, {}).get(mode, "") or config.AI_MODEL
            rep = ai.PROVIDERS[pid].chat(
                api_key=cfg["api_key"], model=model, system=system,
                user_parts=parts, temperature=cfg["temperature"],
                max_tokens=cfg["max_tokens"], base_url=cfg["base_url"])
            results["providers"][pid] = {"status": "completed", "report": rep}
            ok_reports[pid] = rep
        except Exception as e:
            log.warning("multi %s gagal: %s", pid, e)
            results["providers"][pid] = {"status": "failed", "error": str(e)[:300]}
        db.update_job(job_id, multi=results)
    # consensus dari judge (default: provider pertama yang sukses)
    judge = judge or next(iter(ok_reports), "")
    if judge and ok_reports:
        try:
            cfg = _provider_cfg(judge)
            raw = db.get_setting(f"model_{judge}") or ""
            jmodel = raw or ai.MODEL_PRESETS.get(judge, {}).get(mode, "") or config.AI_MODEL
            merged_in = "\n\n".join(
                f"--- ANALYSIS BY {p} ---\n{r[:12000]}" for p, r in ok_reports.items())
            out = ai.PROVIDERS[judge].chat(
                api_key=cfg["api_key"], model=jmodel,
                system=CONSENSUS_SYSTEM,
                user_parts=[{"type": "text",
                             "text": "Gabungkan analisis-analisis berikut:\n\n" + merged_in}],
                temperature=0.2, max_tokens=cfg["max_tokens"],
                base_url=cfg["base_url"])
            results["consensus"] = {"status": "completed", "report": out,
                                    "judge": judge,
                                    "sources": list(ok_reports)}
        except Exception as e:
            results["consensus"] = {"status": "failed", "error": str(e)[:300],
                                    "judge": judge}
    else:
        results["consensus"] = {"status": "failed",
                                "error": "tidak ada analisis sukses untuk digabung"}
    db.update_job(job_id, multi=results)


def launch(job_id: str, providers: list, mode: str, judge: str = ""):
    t = threading.Thread(target=run_multi, args=(job_id, providers, mode, judge),
                         daemon=True)
    t.start()
    return {"ok": True, "job_id": job_id, "providers": providers}
