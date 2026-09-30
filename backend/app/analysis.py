"""Reverse-engineering engine: fitur TERUKUR dari frame (bukan tebakan AI).

Setiap fungsi mengembalikan {"value":..., "confidence": "High/Medium/Low",
"method": "..."} — engine jujur soal cara ukur, AI yang menafsir visual.
Pillow-only (tanpa OpenCV), bekerja pada frame yang sudah diekstrak.
"""
from PIL import Image, ImageStat


def _gray_small(frame_dir, fname, w=160, h=90):
    img = Image.open(frame_dir / fname).convert("L").resize((w, h))
    return img


def _sad(a: Image.Image, b: Image.Image, dx=0, dy=0) -> float:
    """Sum of absolute differences, b digeser (dx,dy). Pure python."""
    pa, pb = a.load(), b.load()
    w, h = a.size
    tot, n = 0, 0
    for y in range(max(0, -dy), min(h, h - dy)):
        for x in range(max(0, -dx), min(w, w - dx)):
            tot += abs(pa[x, y] - pb[x + dx, y + dy])
            n += 1
    return tot / max(n, 1)


def _pan_tilt(a: Image.Image, b: Image.Image, max_shift=24, step=6):
    """Cari geseran global (dx,dy) yang meminimalkan SAD -> arah pan/tilt kamera."""
    base = _sad(a, b)
    best, best_sad = (0, 0), base
    for dy in range(-max_shift, max_shift + 1, step):
        for dx in range(-max_shift, max_shift + 1, step):
            if dx == 0 and dy == 0:
                continue
            s = _sad(a, b, dx, dy)
            if s < best_sad:
                best, best_sad = (dx, dy), s
    reduction = (base - best_sad) / base if base > 0 else 0
    return best, reduction


def _block_energy(a: Image.Image, b: Image.Image, grid=3):
    """Energi gerak per blok 3x3 -> bedakan gerak kamera (merata) vs subjek (lokal).
    Juga rasio tepi/tengah untuk deteksi zoom."""
    pa, pb = a.load(), b.load()
    w, h = a.size
    bw, bh = w // grid, h // grid
    means = []
    for gy in range(grid):
        for gx in range(grid):
            tot, n = 0, 0
            for y in range(gy * bh, (gy + 1) * bh):
                for x in range(gx * bw, (gx + 1) * bw):
                    tot += abs(pa[x, y] - pb[x, y])
                    n += 1
            means.append(tot / n)
    m = sum(means) / len(means)
    var = sum((x - m) ** 2 for x in means) / len(means)
    uniformity = 1 - (var ** 0.5 / (m + 1e-6))
    center = means[4]
    edge = sum(means[:4] + means[5:]) / 8
    return m / 255.0, max(0.0, min(1.0, uniformity)), edge / (center + 1e-6)


def _extract_probes(video_path, scenes, probe_dir, duration):
    """Frame probe rapat (3fps, max 90) khusus deteksi gerak. Return [(ts, path)]."""
    import subprocess
    probe_dir.mkdir(parents=True, exist_ok=True)
    fps = min(3, max(1, int(90 / max(duration, 1))))
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(video_path), "-vf", f"fps={fps}",
                    str(probe_dir / "p_%04d.jpg")],
                   check=True, timeout=300)
    files = sorted(probe_dir.glob("p_*.jpg"))
    return [(i / fps, f) for i, f in enumerate(files)]


def motion_per_scene(video_path, scenes, frame_dir, duration) -> list:
    """Klasifikasi gerak per scene dari probe 3fps:
    static / subject-motion / camera-motion (pan/tilt/zoom)."""
    import shutil
    probe_dir = frame_dir / "probe"
    out = []
    try:
        probes = _extract_probes(video_path, scenes, probe_dir, duration)
    except Exception:
        return [{"scene_id": s["scene_id"], "class": "unknown", "confidence": "Low",
                 "method": "gagal ekstrak probe gerak", "detail": {}} for s in scenes]
    by_scene = {}
    for ts, p in probes:
        sid = next((s["scene_id"] for s in scenes if s["start"] <= ts <= s["end"]),
                   scenes[-1]["scene_id"] if scenes else 1)
        by_scene.setdefault(sid, []).append((ts, p))
    for sc in scenes:
        ps = by_scene.get(sc["scene_id"], [])
        if len(ps) < 2:
            out.append({"scene_id": sc["scene_id"], "class": "unknown",
                        "confidence": "Low", "method": "probe < 2 frame",
                        "detail": {}})
            continue
        votes = {"static": 0, "subject-motion": 0, "camera-motion": 0}
        dirs, energies = [], []
        thumbs = [Image.open(p).convert("L").resize((80, 45)) for _, p in ps]
        fulls = [Image.open(p).convert("L").resize((160, 90)) for _, p in ps]
        for a_t, b_t, a_f, b_f in zip(thumbs, thumbs[1:], fulls, fulls[1:]):
            energy, uniformity, edge_center = _block_energy(a_f, b_f)
            energies.append(round(energy, 3))
            if energy < 0.015:
                votes["static"] += 1
                continue
            (dx, dy), reduction = _pan_tilt(a_t, b_t)
            if edge_center > 2.0 and energy > 0.05:
                votes["camera-motion"] += 1
                dirs.append("zoom")
            elif reduction > 0.25 and (uniformity > 0.5 or abs(dx) + abs(dy) >= 6):
                votes["camera-motion"] += 1
                if abs(dx) >= abs(dy) and dx != 0:
                    dirs.append("pan-right" if dx < 0 else "pan-left")
                elif dy != 0:
                    dirs.append("tilt-up" if dy < 0 else "tilt-down")
            else:
                votes["subject-motion"] += 1
        total = sum(votes.values())
        # tie-break: bukti gerak lebih penting daripada "static"
        cls = max(("camera-motion", "subject-motion", "static"),
                  key=lambda k: votes[k])
        share = votes[cls] / total
        conf = ("High" if share >= 0.75 else "Medium" if share >= 0.5 else "Low")
        detail = {"pair_energies": energies,
                  "probe_fps": min(3, max(1, int(90 / max(duration, 1))))}
        if dirs:
            detail["likely_direction"] = max(set(dirs), key=dirs.count)
        out.append({"scene_id": sc["scene_id"], "class": cls,
                    "confidence": conf,
                    "method": "probe 3fps: frame differencing + shift-search pan/tilt "
                              "+ rasio tepi/tengah untuk zoom + uniformitas blok 3x3",
                    "detail": detail})
        for im in thumbs + fulls:
            im.close()
    shutil.rmtree(probe_dir, ignore_errors=True)
    return out


def color_per_scene(frames: list, scenes: list, frame_dir) -> list:
    """Warna dominan per scene (frame tengah scene)."""
    out = []
    for sc in scenes:
        fs = [f for f in frames if f.get("scene_id") == sc["scene_id"]]
        if not fs:
            continue
        mid = sorted(fs, key=lambda f: f["timestamp"])[len(fs) // 2]
        img = Image.open(frame_dir / mid["file"]).resize((48, 48)).convert("RGB")
        colors = sorted(img.getcolors(48 * 48), reverse=True)[:5]
        total = sum(c for c, _ in colors)
        dom = [{"hex": "#%02x%02x%02x" % col, "coverage": round(cnt / total, 2)}
               for cnt, col in colors]
        out.append({"scene_id": sc["scene_id"], "dominant": dom,
                    "confidence": "High",
                    "method": "kuantisasi 48x48 frame tengah scene"})
    return out


def lighting_per_scene(frames: list, scenes: list, frame_dir) -> list:
    """High-key/low-key + kontras dari statistik brightness."""
    out = []
    for sc in scenes:
        fs = [f for f in frames if f.get("scene_id") == sc["scene_id"]]
        if not fs:
            continue
        means, stds = [], []
        for f in fs:
            st = ImageStat.Stat(_gray_small(frame_dir, f["file"], 80, 45))
            means.append(st.mean[0])
            stds.append(st.stddev[0])
        m, s = sum(means) / len(means), sum(stds) / len(stds)
        key = "high-key" if m > 170 else "low-key" if m < 85 else "normal-key"
        contrast = "high" if s > 60 else "low" if s < 30 else "medium"
        out.append({"scene_id": sc["scene_id"], "key": key,
                    "brightness": round(m, 1), "contrast": contrast,
                    "contrast_value": round(s, 1), "confidence": "High",
                    "method": "mean/stddev brightness grayscale per frame"})
    return out


def editing_rhythm(scenes: list, duration: float) -> dict:
    """Pacing dari durasi shot."""
    durs = [s["duration"] for s in scenes if s["duration"] > 0]
    if not durs:
        return {"value": "unknown", "confidence": "Low", "method": "tidak ada scene"}
    avg = sum(durs) / len(durs)
    mean = avg
    std = (sum((d - mean) ** 2 for d in durs) / len(durs)) ** 0.5
    pacing = "fast" if avg < 2 else "medium" if avg < 5 else "slow"
    return {"avg_shot_length": round(avg, 2),
            "cuts_per_minute": round(len(durs) / max(duration, 0.1) * 60, 1),
            "pacing": pacing,
            "rhythm": "regular" if std / mean < 0.5 else "varied",
            "confidence": "High",
            "method": "statistik durasi scene dari deteksi FFmpeg"}


def camera_estimate(meta: dict, motion: list) -> dict:
    """Fakta teramati + batas jujur: focal length TIDAK bisa diukur dari video."""
    w, h = meta.get("width"), meta.get("height")
    orient = ("portrait" if h and w and h > w else
              "landscape" if h and w and w > h else "unknown")
    return {
        "orientation": {"value": orient, "confidence": "High",
                        "method": f"resolusi terukur {w}x{h}"},
        "aspect_ratio": {"value": meta.get("aspect_ratio"), "confidence": "High",
                         "method": "dari ffprobe"},
        "movement_per_scene": [
            {"scene_id": m["scene_id"], "class": m["class"],
             "confidence": m["confidence"],
             "direction": m["detail"].get("likely_direction")} for m in motion],
        "focal_length": {"value": None, "confidence": "N/A",
                         "method": "Cannot determine from available evidence — "
                                   "butuh estimasi visual AI (lihat laporan)"},
        "camera_model": {"value": None, "confidence": "N/A",
                         "method": "Cannot determine from available evidence"},
    }


def analyze(video_path, frames: list, scenes: list, meta: dict, frame_dir) -> dict:
    """Satu pintu engine -> dipakai pipeline, disimpan di DB, dikirim ke AI."""
    motion = motion_per_scene(video_path, scenes, frame_dir, meta.get("duration") or 0)
    return {
        "motion": motion,
        "color": color_per_scene(frames, scenes, frame_dir),
        "lighting": lighting_per_scene(frames, scenes, frame_dir),
        "editing": editing_rhythm(scenes, meta.get("duration") or 0),
        "camera": camera_estimate(meta, motion),
        "character_reference": {
            "method": "keyframe per scene untuk perbandingan visual oleh AI",
            "frames": [{"scene_id": f.get("scene_id"), "file": f["file"],
                        "timestamp": f["timestamp"]} for f in frames],
        },
    }


def selftest():
    e = editing_rhythm([{"duration": 1.5}, {"duration": 1.6}, {"duration": 1.4}], 4.5)
    assert e["pacing"] == "fast" and e["confidence"] == "High", e
    c = camera_estimate({"width": 640, "height": 960, "aspect_ratio": "2:3"}, [])
    assert c["orientation"]["value"] == "portrait"
    assert c["focal_length"]["value"] is None  # tidak mengarang
    print("analysis selftest ok")


if __name__ == "__main__":
    selftest()
