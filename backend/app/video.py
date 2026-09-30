"""Local video processing. FFmpeg only — no OpenCV, no extra CV deps.

- probe: ffprobe metadata
- detect_scenes: ffmpeg select='gt(scene,T)' + showinfo, parsed from stderr
- extract_frames: seek + single-frame jpg, pre-scaled for the vision model
"""
import hashlib
import json
import re
import subprocess
from pathlib import Path

SCENE_THRESHOLD = 0.35
FRAME_WIDTH = 960  # ponytail: cukup untuk analisis vision, hemat token


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fps(rate: str) -> float:
    try:
        n, d = rate.split("/")
        return float(n) / float(d) if float(d) else 0.0
    except Exception:
        try:
            return float(rate)
        except Exception:
            return 0.0


def probe(path: Path) -> dict:
    p = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, check=True)
    j = json.loads(p.stdout)
    v = next(s for s in j["streams"] if s.get("codec_type") == "video")
    a = next((s for s in j["streams"] if s.get("codec_type") == "audio"), None)
    fmt = j.get("format", {})
    duration = float(fmt.get("duration") or v.get("duration") or 0)
    w, h = int(v.get("width", 0)), int(v.get("height", 0))
    return {
        "duration": round(duration, 2),
        "fps": round(_fps(v.get("r_frame_rate", "0")), 2),
        "width": w, "height": h,
        "aspect_ratio": f"{w}:{h}",
        "orientation": "portrait" if h > w else "landscape" if w > h else "square",
        "video_codec": v.get("codec_name"),
        "audio_codec": a.get("codec_name") if a else None,
        "has_audio": a is not None,
        "bitrate_kbps": round(int(fmt.get("bit_rate", 0)) / 1000) if fmt.get("bit_rate") else None,
        "size_bytes": int(fmt.get("size", 0) or 0),
    }


def detect_scenes(path: Path, duration: float, threshold: float = SCENE_THRESHOLD) -> list:
    """Returns [{scene_id, start, end, duration}]. Falls back to one scene."""
    try:
        p = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", str(path), "-filter:v",
             # format=rgb24: scene SAD jadi sensitif terhadap perubahan chroma juga
             # (cut merah->hijau dengan luma sama tetap terdeteksi)
             f"format=rgb24,select='gt(scene,{threshold})',showinfo",
             "-vsync", "vfr", "-f", "null", "-"],
            capture_output=True, text=True, timeout=600)
        cuts = [0.0]
        for m in re.finditer(r"pts_time:([\d.]+)", p.stderr):
            t = float(m.group(1))
            if t - cuts[-1] > 0.5:  # dedupe clustered detections
                cuts.append(t)
    except Exception:
        cuts = [0.0]
    scenes = []
    bounds = cuts + [duration]
    for i in range(len(cuts)):
        s, e = round(cuts[i], 2), round(min(bounds[i + 1], duration), 2)
        if e - s < 0.2:
            continue
        scenes.append({"scene_id": i + 1, "start": s, "end": e,
                       "duration": round(e - s, 2)})
    return scenes or [{"scene_id": 1, "start": 0.0, "end": round(duration, 2),
                       "duration": round(duration, 2)}]


def extract_frames(path: Path, timestamps: list, outdir: Path) -> list:
    outdir.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, t in enumerate(timestamps):
        fp = outdir / f"frame_{i:03d}_{t:.2f}s.jpg"
        subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-ss", str(max(t, 0)), "-i", str(path),
             "-frames:v", "1", "-q:v", "4",
             "-vf", f"scale={FRAME_WIDTH}:-1", str(fp)],
            check=True, timeout=120)
        frames.append({"index": i, "timestamp": round(t, 2), "file": fp.name})
    return frames


MODE_CAPS = {"quick": 5, "standard": 10, "deep": 20, "prompt": 12}


def sample_timestamps(meta: dict, scenes: list, mode: str) -> list:
    """Opening + scene cuts + scene middles + ending, deduped, capped by mode."""
    cap = MODE_CAPS.get(mode, 10)
    dur = meta["duration"]
    ts = [0.1]
    for sc in scenes:
        for t in (sc["start"] + 0.2, (sc["start"] + sc["end"]) / 2):
            if 0 < t < dur and all(abs(t - x) > 0.4 for x in ts):
                ts.append(round(t, 2))
    if dur > 0.5:
        ts.append(round(dur - 0.2, 2))
    return sorted(ts)[:cap]
