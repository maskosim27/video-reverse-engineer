"""Video sources. Uniform interface; each platform fetches via yt-dlp.

Modularity = failure isolation: every fetch() is its own subprocess call and
raises SourceError carrying the platform label. One platform failing (removed
video, login wall, rate limit) never touches the others or the pipeline —
the caller decides (400 + "coba Upload" fallback).
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path


class SourceError(Exception):
    pass


def _yt_dlp_cmd() -> list:
    exe = shutil.which("yt-dlp")
    if exe:
        return [exe]
    try:
        import yt_dlp  # noqa: F401 — fallback: modul di venv yang sama
        return [sys.executable, "-m", "yt_dlp"]
    except ImportError:
        raise SourceError("yt-dlp tidak terinstall (pip install yt-dlp)")


def _yt_dlp(url: str, dest_dir: Path, label: str) -> str:
    dest_dir.mkdir(parents=True, exist_ok=True)
    before = set(dest_dir.iterdir())
    cmd = _yt_dlp_cmd() + ["--no-playlist", "--no-warnings",
                           "-f", "bv*[height<=1080]+ba/b[height<=1080]/b",
                           "--merge-output-format", "mp4",
                           "-o", str(dest_dir / "%(id)s.%(ext)s"), url]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        raise SourceError(f"Download {label} timeout (15 menit)")
    if p.returncode != 0:
        err = ((p.stderr or "") + (p.stdout or ""))[-400:].strip()
        raise SourceError(f"Download {label} gagal: {err or 'unknown error'}")
    new = [f for f in set(dest_dir.iterdir()) - before
           if f.suffix.lower() in (".mp4", ".webm", ".mkv", ".mov")]
    if not new:
        raise SourceError(f"Download {label} gagal: file video tidak ditemukan")
    return str(max(new, key=lambda f: f.stat().st_mtime))


class VideoSource:
    id = ""
    label = ""
    pattern = None

    @classmethod
    def detect(cls, url: str) -> bool:
        return bool(cls.pattern and cls.pattern.search(url or ""))

    def fetch(self, url: str, dest_dir) -> str:
        """Download URL -> path file video. Tiap platform isolated di sini."""
        return _yt_dlp(url, Path(dest_dir), self.label)


class TikTokSource(VideoSource):
    id, label = "tiktok", "TikTok"
    pattern = re.compile(r"tiktok\.com", re.I)


class YouTubeSource(VideoSource):
    id, label = "youtube", "YouTube / Shorts"
    pattern = re.compile(r"(youtube\.com|youtu\.be)", re.I)


class FacebookSource(VideoSource):
    id, label = "facebook", "Facebook"
    pattern = re.compile(r"(facebook\.com|fb\.watch)", re.I)


class XSource(VideoSource):
    id, label = "x", "X / Twitter"
    pattern = re.compile(r"(twitter\.com|x\.com)", re.I)


class ThreadsSource(VideoSource):
    id, label = "threads", "Threads"
    pattern = re.compile(r"threads\.(com|net)", re.I)


class LocalUploadSource(VideoSource):
    id, label = "local", "Local Upload"

    def fetch(self, url: str, dest_dir) -> str:
        raise SourceError("Local Upload tidak memakai URL")


SOURCES = [TikTokSource(), YouTubeSource(), FacebookSource(), XSource(),
           ThreadsSource(), LocalUploadSource()]


def detect_platform(url: str):
    for s in SOURCES:
        if s.detect(url):
            return s
    return None


def by_id(source_id: str):
    return next((s for s in SOURCES if s.id == source_id), None)
