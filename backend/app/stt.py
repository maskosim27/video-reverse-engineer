"""Speech-to-text lokal via faster-whisper. Graceful: tanpa audio -> None,
tanpa dep -> STTError (pipeline skip dengan catatan, bukan crash)."""
import subprocess
from pathlib import Path


class STTError(Exception):
    pass


def extract_audio(video_path: Path, out_wav: Path) -> Path | None:
    r = subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", str(out_wav)],
        capture_output=True, timeout=600)
    if r.returncode != 0 or not out_wav.exists() or out_wav.stat().st_size == 0:
        return None  # tidak ada track audio
    return out_wav


def transcribe(wav: Path, model_name: str) -> list:
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise STTError("faster-whisper belum terinstall: pip install faster-whisper")
    # ponytail: model diunduh otomatis saat pertama dipakai; int8 CPU cukup
    try:
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
    except Exception as e:
        raise STTError(f"Gagal memuat model Whisper '{model_name}': {e}")
    segments, _ = model.transcribe(str(wav), beam_size=5)
    return [{"start": round(s.start, 2), "end": round(s.end, 2),
             "text": s.text.strip()}
            for s in segments if s.text.strip()]
