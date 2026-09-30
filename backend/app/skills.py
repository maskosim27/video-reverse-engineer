"""Skill system: SKILL.md sebagai TEKS pengetahuan untuk AI.

SANDBOX MODEL: skill TIDAK PERNAH dieksekusi. Isi SKILL.md (setelah frontmatter)
hanya disuntik sebagai konteks tambahan ke system prompt AI. Tidak ada eval,
tidak ada exec, tidak ada subprocess, tidak ada import dari direktori skill.
Modul ini hanya: download teks -> parse frontmatter -> simpan file -> baca teks.

Interface:
  parse_skill_md(text)      -> {name, description, version, body}
  import_from_url(url)      -> meta dict (GitHub blob/raw URL)
  list_skills()             -> [meta...]
  get_skill(name)           -> {meta, content}
  set_enabled(name, bool)
  delete_skill(name)
  rollback(name, version)
  build_system_prompt()     -> ai.SYSTEM + custom prompt + enabled skills
"""
import json
import re
import shutil
import urllib.request
from datetime import datetime, timezone

from . import ai, config, db

SKILL_DIR = config.DB_PATH.parent / "skills"  # ./data/skills
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
MAX_VERSIONS = 10


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sdir(name: str):
    if not NAME_RE.match(name):
        raise ValueError(f"Nama skill tidak valid: {name!r} (a-z 0-9 _ -)")
    return SKILL_DIR / name


def parse_skill_md(text: str) -> dict:
    """Parser frontmatter YAML-lite: --- key: value --- lalu body markdown."""
    meta = {"name": "", "description": "", "version": "1.0.0", "body": text.strip()}
    m = re.match(r"\s*---\s*\n(.*?)\n---\s*\n(.*)$", text, re.S)
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                k, v = k.strip().lower(), v.strip().strip("'\"")
                if k in ("name", "description", "version"):
                    meta[k] = v
        meta["body"] = m.group(2).strip()
    if not meta["name"]:
        raise ValueError("SKILL.md butuh frontmatter 'name:'")
    if not NAME_RE.match(meta["name"]):
        raise ValueError(f"Nama skill tidak valid: {meta['name']!r}")
    return meta


def _github_to_raw(url: str) -> str:
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)", url)
    if m:
        user, repo, branch, path = m.groups()
        return f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}"
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+)/raw/([^/]+)/(.+)", url)
    if m:
        user, repo, branch, path = m.groups()
        return f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}"
    return url


def import_from_url(url: str) -> dict:
    raw = _github_to_raw(url.strip())
    if not (raw.startswith("https://") or
            raw.startswith("http://127.0.0.1") or raw.startswith("http://localhost")):
        raise ValueError("URL harus https:// (http hanya untuk localhost test)")
    req = urllib.request.Request(raw, headers={"User-Agent": "video-reverse-engineer"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            text = r.read().decode("utf-8", "replace")
    except Exception as e:
        raise RuntimeError(f"Gagal download SKILL.md: {e}")
    if len(text) > 500_000:
        raise ValueError("SKILL.md terlalu besar (>500KB)")
    parsed = parse_skill_md(text)
    name = parsed["name"]
    sdir = _sdir(name)
    sdir.mkdir(parents=True, exist_ok=True)
    meta_path = sdir / "meta.json"
    meta = {"name": name, "description": parsed["description"],
            "version": parsed["version"], "enabled": True,
            "source_url": url, "imported_at": _now(), "history": []}
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        meta["enabled"] = old.get("enabled", True)
        meta["history"] = old.get("history", [])
        old_md = sdir / "SKILL.md"
        if old_md.exists() and old_md.read_text() != text:
            vdir = sdir / "versions"
            vdir.mkdir(exist_ok=True)
            stamp = _now().replace(":", "-")
            (vdir / f"{old.get('version', '0')}_{stamp}.md").write_text(old_md.read_text())
            meta["history"].append({"version": old.get("version"),
                                    "imported_at": old.get("imported_at"),
                                    "saved_at": _now()})
            meta["history"] = meta["history"][-MAX_VERSIONS:]
            for f in sorted(vdir.glob("*.md"))[:-MAX_VERSIONS]:
                f.unlink()
    (sdir / "SKILL.md").write_text(text)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


def list_skills() -> list:
    out = []
    if not SKILL_DIR.exists():
        return out
    for sdir in sorted(SKILL_DIR.iterdir()):
        mp = sdir / "meta.json"
        if sdir.is_dir() and mp.exists():
            out.append(json.loads(mp.read_text()))
    return out


def get_skill(name: str) -> dict:
    sdir = _sdir(name)
    mp = sdir / "meta.json"
    if not mp.exists():
        raise KeyError(f"Skill tidak ditemukan: {name}")
    return {"meta": json.loads(mp.read_text()),
            "content": (sdir / "SKILL.md").read_text()}


def set_enabled(name: str, enabled: bool) -> dict:
    sdir = _sdir(name)
    mp = sdir / "meta.json"
    if not mp.exists():
        raise KeyError(f"Skill tidak ditemukan: {name}")
    meta = json.loads(mp.read_text())
    meta["enabled"] = bool(enabled)
    mp.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


def delete_skill(name: str):
    sdir = _sdir(name)
    if not sdir.exists():
        raise KeyError(f"Skill tidak ditemukan: {name}")
    shutil.rmtree(sdir)


def rollback(name: str, version_file: str) -> dict:
    """Kembalikan SKILL.md ke versi tersimpan. version_file = nama file di versions/."""
    sdir = _sdir(name)
    if ".." in version_file or "/" in version_file:
        raise ValueError("Nama versi tidak valid")
    src = sdir / "versions" / version_file
    if not src.exists():
        raise KeyError(f"Versi tidak ditemukan: {version_file}")
    text = src.read_text()
    parsed = parse_skill_md(text)
    (sdir / "SKILL.md").write_text(text)
    mp = sdir / "meta.json"
    meta = json.loads(mp.read_text())
    meta["history"].append({"version": meta.get("version"),
                            "imported_at": meta.get("imported_at"),
                            "saved_at": _now(), "note": "rollback"})
    meta["version"] = parsed["version"]
    meta["description"] = parsed["description"]
    meta["imported_at"] = _now()
    mp.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


def build_system_prompt() -> str:
    """SYSTEM + custom prompt user + skill yang enabled (teks saja)."""
    parts = [ai.SYSTEM]
    custom = (db.get_setting("custom_system_prompt") or "").strip()
    if custom:
        parts.append("\n\n--- CUSTOM SYSTEM PROMPT (user) ---\n" + custom)
    for s in list_skills():
        if not s.get("enabled"):
            continue
        try:
            body = (_sdir(s["name"]) / "SKILL.md").read_text()
            parsed = parse_skill_md(body)
            parts.append(f"\n\n--- SKILL: {s['name']} v{s.get('version', '?')} "
                         f"(teks pengetahuan, bukan instruksi eksekusi) ---\n"
                         f"{parsed['body'][:8000]}")
        except Exception:
            continue
    return "\n".join(parts)


def selftest():
    md = "---\nname: test-skill\ndescription: uji\nversion: 2.1.0\n---\n\n# Hello\nbody teks"
    p = parse_skill_md(md)
    assert p["name"] == "test-skill" and p["version"] == "2.1.0" and "Hello" in p["body"]
    assert _github_to_raw(
        "https://github.com/u/r/blob/main/skills/x/SKILL.md") == \
        "https://raw.githubusercontent.com/u/r/main/skills/x/SKILL.md"
    try:
        parse_skill_md("no frontmatter")
        raise AssertionError("harus gagal tanpa name")
    except ValueError:
        pass
    try:
        _sdir("../evil")
        raise AssertionError("path traversal harus ditolak")
    except ValueError:
        pass
    print("skills selftest ok")


if __name__ == "__main__":
    selftest()
