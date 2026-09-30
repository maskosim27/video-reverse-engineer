"""FastAPI app. Serves the single-file frontend + JSON API.

API:
  POST /api/video/upload            (multipart: file, name, mode, force)
  POST /api/video/analyze           {url, name?, mode?, force?} -> via yt-dlp per platform
  GET  /api/video/{id}              job summary + meta/scenes/frames/transcript/ocr/estimate
  GET  /api/video/{id}/analysis     report markdown
  GET  /api/video/{id}/file         original video
  GET  /api/video/{id}/frames/{f}   frame jpg
  GET  /api/video/{id}/export?format=md|txt|json|zip
  GET  /api/projects  DELETE /api/projects/{id}
  GET  /api/ai/providers  POST /api/ai/test
  GET/POST /api/settings            (per-provider key, encrypted, never exposed whole)
"""
import io
import json
import logging
import shutil
import uuid
import zipfile
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import ai, config, db, jobs, pipeline, sources, video

log = logging.getLogger("vre")

# --- encrypted settings -------------------------------------------------
if config.SECRET_KEY:
    _fernet = Fernet(config.SECRET_KEY.encode())
else:
    _fernet = Fernet(Fernet.generate_key())
    log.warning("SECRET_KEY kosong — kunci enkripsi ephemeral, API key tersimpan tidak "
                "bertahan setelah restart. Isi SECRET_KEY di .env.")


def _enc(v: str) -> str:
    return _fernet.encrypt(v.encode()).decode() if v else ""


def _dec(v: str) -> str:
    if not v:
        return ""
    try:
        return _fernet.decrypt(v.encode()).decode()
    except InvalidToken:
        return ""


def _env_key(provider: str) -> str:
    return {"openrouter": config.OPENROUTER_API_KEY,
            "ninerouter": config.NINEROUTER_API_KEY}.get(provider, "")


def _env_base(provider: str) -> str:
    return {"openrouter": config.OPENROUTER_BASE_URL,
            "ninerouter": config.NINEROUTER_BASE_URL}.get(provider, "")


def effective_ai(provider: str) -> dict:
    """DB settings override .env. Key & base_url stay server-side."""
    s = lambda k: db.get_setting(k)  # noqa: E731
    base_url = _dec(s(f"base_url_{provider}") or "") or _env_base(provider)
    return {
        "provider": provider,
        "model": s(f"model_{provider}") or config.AI_MODEL,
        "model_raw": s(f"model_{provider}") or "",  # kosong -> preset otomatis per mode
        "api_key": _dec(s(f"api_key_{provider}") or "") or _env_key(provider),
        "base_url": base_url,
        "temperature": float(s("temperature") or config.AI_TEMPERATURE),
        "max_tokens": int(s("max_tokens") or config.AI_MAX_TOKENS),
    }


def _masked(key: str) -> str:
    return ("*" * 8 + key[-4:]) if len(key) > 4 else ("*" * len(key))


# --- app ----------------------------------------------------------------
app = FastAPI(title="Video Reverse Engineering Analyzer")


@app.on_event("startup")
def _startup():
    db.init()
    db.reset_stale()


class AnalyzeURL(BaseModel):
    url: str
    name: str = ""
    mode: str = "standard"
    force: bool = False


class AITest(BaseModel):
    provider: str = "openrouter"
    api_key: str = ""
    model: str = ""
    base_url: str = ""


class SettingsIn(BaseModel):
    provider: str = "openrouter"
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.2
    max_tokens: int = 8000
    custom_system_prompt: str = ""
    pricing_per_1m: str = ""


def _launch(job_id: str, mode: str, provider: str, url: str | None = None,
            source_id: str | None = None, force: bool = False):
    cfg = effective_ai(provider)
    if cfg["provider"] not in ai.PROVIDERS:
        raise HTTPException(400, f"Provider '{provider}' tidak dikenal")
    if cfg["provider"] not in ("mock",) and not cfg["api_key"]:
        raise HTTPException(400,
                            f"API key {ai.PROVIDERS[provider].label} kosong — isi di Settings")
    if cfg["provider"] == "custom" and not cfg["base_url"]:
        raise HTTPException(400, "Base URL kosong — isi di Settings")
    jobs.submit(pipeline.run, job_id, cfg["provider"], cfg["model_raw"], cfg["api_key"],
                cfg["base_url"], cfg["temperature"], cfg["max_tokens"], mode,
                url, source_id, force)
    return {"job_id": job_id, "status": "queued"}


@app.post("/api/video/upload")
async def upload(file: UploadFile = File(...), name: str = Form(""),
                 mode: str = Form("standard"), force: bool = Form(False),
                 provider: str = Form("openrouter")):
    if not file.filename:
        raise HTTPException(400, "File kosong")
    job_id = uuid.uuid4().hex[:12]
    vdir = config.VIDEO_DIR / job_id
    vdir.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename).suffix.lower() or ".mp4"
    vpath = vdir / f"original{ext}"
    with open(vpath, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    db.create_job({"id": job_id, "name": name or file.filename, "source_type": "local",
                   "video_path": str(vpath)})
    return _launch(job_id, mode, provider, force=force)


@app.post("/api/video/analyze")
def analyze_url(body: AnalyzeURL, provider: str = "openrouter"):
    src = sources.detect_platform(body.url)
    if src is None or src.id == "local":
        raise HTTPException(400, "URL tidak dikenali — gunakan tab Upload sebagai fallback")
    job_id = uuid.uuid4().hex[:12]
    db.create_job({"id": job_id, "name": body.name or f"{src.label} video",
                   "source_type": src.id, "source_url": body.url, "status": "queued",
                   "video_path": ""})
    # download berjalan di background (status: downloading); gagalnya satu
    # platform terisolasi di sources.fetch -> job failed dengan pesan jelas
    return _launch(job_id, body.mode, provider, url=body.url,
                   source_id=src.id, force=body.force)


@app.get("/api/video/{job_id}")
def get_video(job_id: str):
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job tidak ditemukan")
    job.pop("video_path", None)
    return job


@app.get("/api/video/{job_id}/analysis")
def get_analysis(job_id: str):
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job tidak ditemukan")
    return {"report_md": job.get("report_md")}


@app.get("/api/video/{job_id}/file")
def get_file(job_id: str):
    job = db.get_job(job_id)
    if not job or not job.get("video_path"):
        raise HTTPException(404, "Video tidak ditemukan")
    return FileResponse(job["video_path"])


@app.get("/api/video/{job_id}/frames/{fname}")
def get_frame(job_id: str, fname: str):
    fp = config.FRAME_DIR / job_id / Path(fname).name
    if not fp.is_file():
        raise HTTPException(404, "Frame tidak ditemukan")
    return FileResponse(fp, media_type="image/jpeg")


def _transcript_txt(transcript) -> str:
    if not transcript:
        return ""
    return "\n".join(f"[{s['start']:.1f}-{s['end']:.1f}] (scene {s.get('scene_id')}) "
                     f"{s['text']}" for s in transcript)


@app.get("/api/video/{job_id}/export")
def export(job_id: str, format: str = "md"):
    job = db.get_job(job_id)
    if not job or job["status"] != "completed":
        raise HTTPException(400, "Analisis belum selesai")
    name = "".join(c if c.isalnum() or c in "-_" else "_" for c in job["name"])[:40]
    report = job.get("report_md") or ""
    if format == "txt":
        return StreamingResponse(io.BytesIO(report.encode()), media_type="text/plain",
                                 headers={"Content-Disposition":
                                          f"attachment; filename={name}.txt"})
    if format == "json":
        payload = {k: job.get(k) for k in
                   ("id", "name", "source_type", "source_url", "sha256", "meta",
                    "scenes", "frames", "transcript", "ocr", "analysis", "estimate",
                    "provider", "model", "mode")}
        payload["report_md"] = report
        return JSONResponse(payload)
    if format == "zip":
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(job["video_path"], "video.mp4")
            z.writestr("analysis.md", report)
            z.writestr("transcript.txt", _transcript_txt(job.get("transcript")))
            z.writestr("ocr.json", json.dumps(job.get("ocr"), indent=2, ensure_ascii=False))
            z.writestr("engine_analysis.json",
                       json.dumps(job.get("analysis"), indent=2, ensure_ascii=False))
            z.writestr("metadata.json", json.dumps(job.get("meta"), indent=2))
            z.writestr("scenes.json", json.dumps(job.get("scenes"), indent=2))
            fdir = config.FRAME_DIR / job_id
            if fdir.is_dir():
                for fp in sorted(fdir.glob("*.jpg")):
                    z.write(fp, f"frames/{fp.name}")
        buf.seek(0)
        return StreamingResponse(buf, media_type="application/zip",
                                 headers={"Content-Disposition":
                                          f"attachment; filename={name}.zip"})
    return StreamingResponse(io.BytesIO(report.encode()), media_type="text/markdown",
                             headers={"Content-Disposition":
                                      f"attachment; filename={name}.md"})


@app.get("/api/projects")
def projects():
    return db.list_jobs()


@app.delete("/api/projects/{job_id}")
def delete_project(job_id: str):
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job tidak ditemukan")
    db.delete_job(job_id)
    shutil.rmtree(config.VIDEO_DIR / job_id, ignore_errors=True)
    shutil.rmtree(config.FRAME_DIR / job_id, ignore_errors=True)
    return {"ok": True}


@app.get("/api/ai/providers")
def providers():
    return [{"id": p.id, "label": p.label} for p in ai.PROVIDERS.values()]


@app.post("/api/ai/test")
def ai_test(body: AITest):
    p = ai.PROVIDERS.get(body.provider)
    if not p:
        raise HTTPException(400, "Provider tidak dikenal")
    if body.provider == "mock":
        return {"ok": True, "detail": "Mock selalu OK"}
    cfg = effective_ai(body.provider)
    return p.test(api_key=body.api_key or cfg["api_key"],
                  model=body.model or cfg["model"],
                  base_url=body.base_url or cfg["base_url"])


@app.get("/api/settings")
def get_settings(provider: str = "openrouter"):
    cfg = effective_ai(provider)
    return {"provider": cfg["provider"], "model": cfg["model"],
            "api_key_masked": _masked(cfg["api_key"]), "base_url": cfg["base_url"],
            "temperature": cfg["temperature"], "max_tokens": cfg["max_tokens"],
            "custom_system_prompt": db.get_setting("custom_system_prompt") or "",
            "pricing_per_1m": db.get_setting("pricing_per_1m") or ""}


@app.post("/api/settings")
def save_settings(body: SettingsIn):
    p = body.provider
    db.set_setting(f"model_{p}", body.model)
    if body.api_key and not set(body.api_key) <= set("*"):
        db.set_setting(f"api_key_{p}", _enc(body.api_key))
    if body.base_url:
        db.set_setting(f"base_url_{p}", _enc(body.base_url))
    db.set_setting("temperature", str(body.temperature))
    db.set_setting("max_tokens", str(body.max_tokens))
    db.set_setting("custom_system_prompt", body.custom_system_prompt or "")
    db.set_setting("pricing_per_1m", body.pricing_per_1m or "")
    return {"ok": True}


# --- skills ---------------------------------------------------------------
from app import skills as skill_mod  # noqa: E402
from app import multi as multi_mod  # noqa: E402
from app import compare as compare_mod  # noqa: E402


class SkillImport(BaseModel):
    url: str


class SkillEnable(BaseModel):
    enabled: bool


class SkillRollback(BaseModel):
    version: str


@app.get("/api/skills")
def skills_list():
    return skill_mod.list_skills()


@app.post("/api/skills/import")
def skills_import(body: SkillImport):
    try:
        return skill_mod.import_from_url(body.url)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e))


@app.get("/api/skills/{name}")
def skills_get(name: str):
    try:
        return skill_mod.get_skill(name)
    except (KeyError, ValueError) as e:
        raise HTTPException(404, str(e))


@app.post("/api/skills/{name}/enable")
def skills_enable(name: str, body: SkillEnable):
    try:
        return skill_mod.set_enabled(name, body.enabled)
    except (KeyError, ValueError) as e:
        raise HTTPException(404, str(e))


@app.post("/api/skills/{name}/rollback")
def skills_rollback(name: str, body: SkillRollback):
    try:
        return skill_mod.rollback(name, body.version)
    except (KeyError, ValueError) as e:
        raise HTTPException(400, str(e))


@app.delete("/api/skills/{name}")
def skills_delete(name: str):
    try:
        skill_mod.delete_skill(name)
        return {"ok": True}
    except (KeyError, ValueError) as e:
        raise HTTPException(404, str(e))


# --- advanced AI: multi-AI, consensus, compare ------------------------------
@app.get("/api/ai/presets")
def ai_presets():
    return ai.MODEL_PRESETS


class MultiIn(BaseModel):
    providers: list[str] = []
    mode: str = "standard"
    judge: str = ""


@app.post("/api/video/{job_id}/multi")
def video_multi(job_id: str, body: MultiIn):
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job tidak ditemukan")
    if job["status"] != "completed":
        raise HTTPException(400, "Job belum completed")
    provs = [p for p in body.providers if p in ai.PROVIDERS and p != "mock"]
    if not provs:
        raise HTTPException(400, "Pilih minimal 1 provider AI (bukan mock)")
    return multi_mod.launch(job_id, provs, body.mode, body.judge)


class CompareIn(BaseModel):
    generated_id: str
    provider: str = "openrouter"
    parent_id: str = ""


@app.post("/api/video/{job_id}/compare")
def video_compare(job_id: str, body: CompareIn):
    try:
        return compare_mod.compare(job_id, body.generated_id, body.provider,
                                   body.parent_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(502, str(e))


@app.get("/api/video/{job_id}/recreations")
def video_recreations(job_id: str):
    return compare_mod.list_recreations(job_id)


app.mount("/", StaticFiles(directory=str(Path(__file__).parent / "static"),
                           html=True), name="static")
