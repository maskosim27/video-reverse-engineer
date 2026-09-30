"""Ingest pipeline:
URL? -> downloading -> probe -> scenes -> frames -> transcribing -> ocr -> AI -> report.

STT/OCR graceful: gagal/tidak tersedia -> None (kolom tetap ada, laporan jujur
menandai dilewati). Satu stage gagal tidak membatalkan stage lain.
"""
import logging
import shutil
import traceback
from pathlib import Path

from . import ai, analysis as analysis_mod, config, db, ocr as ocr_mod, skills, sources, stt, video

log = logging.getLogger("vre")


def _progress(job_id, status, pct):
    db.update_job(job_id, status=status, progress=pct)


def _sync_transcript(transcript, scenes):
    """Bubuhkan scene_id pada tiap segmen transcript (sinkronisasi scene)."""
    for s in transcript:
        mid = (s["start"] + s["end"]) / 2
        s["scene_id"] = next(
            (sc["scene_id"] for sc in scenes if sc["start"] <= mid <= sc["end"]),
            scenes[-1]["scene_id"])
    return transcript


def run(job_id: str, provider_id: str, model: str, api_key: str, base_url: str,
        temperature: float, max_tokens: int, mode: str,
        url: str | None = None, source_id: str | None = None,
        force: bool = False):
    try:
        # automatic model selection: preset per provider+mode bila user kosong;
        # fallback ke default global bila tidak ada preset (custom provider).
        if not model:
            model = ai.MODEL_PRESETS.get(provider_id, {}).get(mode, "") or config.AI_MODEL
        job = db.get_job(job_id)
        vdir = config.VIDEO_DIR / job_id

        if url:
            _progress(job_id, "downloading", 0.05)
            src = sources.by_id(source_id)
            if src is None:
                raise sources.SourceError(f"Source '{source_id}' tidak dikenal")
            try:
                vpath = Path(src.fetch(url, vdir))
            except sources.SourceError as e:
                # kegagalan satu platform terisolasi di sini — pipeline lain aman
                raise RuntimeError(f"{e} Coba tab Upload sebagai fallback.")
            db.update_job(job_id, video_path=str(vpath))
        else:
            vpath = Path(job["video_path"])

        sha = video.sha256_file(vpath)
        if not force:
            hit = db.find_cached(sha)
            src_fd = config.FRAME_DIR / hit["id"] if hit else None
            if hit and hit["id"] != job_id and src_fd and src_fd.exists():
                # cache hit: salin frame agar tiap job punya direktori sendiri
                # (endpoint frames & multi-AI membaca FRAME_DIR/job_id)
                try:
                    shutil.copytree(src_fd, config.FRAME_DIR / job_id,
                                    dirs_exist_ok=True)
                except Exception as e:
                    log.warning("gagal salin frame cache: %s", e)
                db.update_job(job_id, status="completed", progress=1.0, sha256=sha,
                              meta=hit["meta"], scenes=hit["scenes"], frames=hit["frames"],
                              estimate=hit["estimate"], transcript=hit["transcript"],
                              ocr=hit["ocr"], analysis=hit["analysis"],
                              multi=hit["multi"],
                              report_md=hit["report_md"],
                              provider=hit["provider"], model=hit["model"],
                              mode=hit["mode"], finished_at=db.now())
                return

        _progress(job_id, "processing", 0.15)
        meta = video.probe(vpath)
        meta["sha256"] = sha
        db.update_job(job_id, meta=meta, sha256=sha)

        _progress(job_id, "processing", 0.3)
        scenes = video.detect_scenes(vpath, meta["duration"])
        db.update_job(job_id, scenes=scenes)

        _progress(job_id, "extracting_frames", 0.45)
        frame_dir = config.FRAME_DIR / job_id
        stamps = video.sample_timestamps(meta, scenes, mode)
        frames = video.extract_frames(vpath, stamps, frame_dir)
        for f in frames:
            f["scene_id"] = next(
                (s["scene_id"] for s in scenes if s["start"] <= f["timestamp"] <= s["end"]),
                scenes[-1]["scene_id"])
        db.update_job(job_id, frames=frames)

        _progress(job_id, "transcribing", 0.6)
        transcript = None
        if meta.get("has_audio"):
            wav = vdir / "audio.wav"
            try:
                if stt.extract_audio(vpath, wav):
                    transcript = _sync_transcript(
                        stt.transcribe(wav, config.WHISPER_MODEL), scenes)
                else:
                    transcript = []  # audio track kosong
            except Exception as e:
                # STT gagal (dep hilang, model gagal diunduh, dsb) -> dilewati,
                # analisis frame tetap jalan. Jujur dicatat di laporan.
                log.warning("STT dilewati: %s", e)
                transcript = None
            finally:
                wav.unlink(missing_ok=True)
        else:
            transcript = []
        db.update_job(job_id, transcript=transcript)

        _progress(job_id, "ocr", 0.7)
        try:
            ocr = ocr_mod.ocr_frames(frames, frame_dir, lang=config.OCR_LANG)
        except Exception as e:
            log.warning("OCR dilewati: %s", e)
            ocr = None  # tesseract tidak tersedia -> dilewati
        db.update_job(job_id, ocr=ocr)

        _progress(job_id, "analyzing", 0.78)
        try:
            analysis = analysis_mod.analyze(vpath, frames, scenes, meta, frame_dir)
        except Exception as e:
            log.warning("Engine analysis dilewati: %s", e)
            analysis = None
        db.update_job(job_id, analysis=analysis)

        text, parts = ai.build_messages(meta, scenes, frames, frame_dir, mode,
                                        transcript, ocr, analysis)
        pricing = {"per_1m_input": db.get_setting("pricing_per_1m") or ""}
        db.update_job(job_id, estimate=ai.estimate_cost(len(frames), len(text), pricing))

        _progress(job_id, "ai_analysis", 0.85)
        if provider_id == "mock":
            report = ai.mock_report(meta, scenes, frames, transcript, ocr, analysis)
        else:
            provider = ai.PROVIDERS[provider_id]
            report = provider.chat(api_key=api_key, model=model,
                                   system=skills.build_system_prompt(),
                                   user_parts=parts, temperature=temperature,
                                   max_tokens=max_tokens, base_url=base_url)
        db.update_job(job_id, status="completed", progress=1.0, report_md=report,
                      finished_at=db.now(), provider=provider_id, model=model, mode=mode)
    except Exception as e:
        db.update_job(job_id, status="failed", progress=0,
                      error=f"{type(e).__name__}: {e}", finished_at=db.now())
        traceback.print_exc()


def selftest():
    scenes = [{"scene_id": 1, "start": 0.0, "end": 3.7, "duration": 3.7},
              {"scene_id": 2, "start": 3.7, "end": 12.0, "duration": 8.3}]
    tr = _sync_transcript([{"start": 1.0, "end": 2.0, "text": "halo"},
                           {"start": 5.0, "end": 6.0, "text": "dunia"}], scenes)
    assert [s["scene_id"] for s in tr] == [1, 2], tr
    print("pipeline selftest ok")


if __name__ == "__main__":
    selftest()
