"""AI providers. Uniform interface; chat() gains base_url for OpenAI-compatible.

- OpenRouterProvider: default OpenRouter base URL.
- NineRouterProvider: 9Router gateway lokal (default http://127.0.0.1:20128/v1).
- CustomOpenAICompatibleProvider: base URL bebas dari user.
- MockProvider: laporan jujur dari data teramati, tanpa API key.
Cost: never invented — estimate() = counts + "unavailable", kecuali user
mengisi harga per 1M token di settings (pricing_*), lalu dihitung jujur.
"""
import base64
import json
import urllib.request
from pathlib import Path

from . import config

# §43 — aturan keras untuk AI analyzer
SYSTEM = """You are a video reverse-engineering analyst. Analyze ONLY what is visible/audible.

HARD RULES (§43):
- NEVER claim to know the video's original prompt. Every reconstruction is labeled:
  "Reconstructed prompt — estimated from observable characteristics."
- Always use the terms: estimated, inferred, reconstructed, likely.
- Separate every claim into: [Observed] = measurable fact from frames/metadata,
  [Inferred] = reasonable conclusion from evidence, [Speculation] = low-confidence guess.
- NEVER invent: focal length (mm), camera model, AI generator model, or the original prompt.
- If it cannot be known from the video, write exactly: "Unknown / Cannot determine from available evidence."
- Evidence from frames/video ALWAYS outranks assumption.
- Reconstruct prompts ONLY from characteristics actually visible.
- Temporal analysis is MANDATORY: motion continuity, camera movement, editing rhythm,
  pacing — never output image description alone.
- Output Markdown, in Indonesian unless the content is a prompt (prompts stay in English)."""

# §44 — struktur laporan akhir (23 seksi)
REPORT_STRUCTURE = """OUTPUT STRUCTURE — ikuti tepat 23 seksi ini sebagai heading:

VIDEO REVERSE ENGINEERING REPORT
1. Video Overview
2. Technical Metadata
3. Story Structure
4. Hook Analysis (3 detik pertama)
5. Scene Breakdown (per scene: subject, environment, camera, lighting, color, action, text)
6. Character Analysis
7. Environment Analysis
8. Camera Analysis (movement, angle, shot scale — semua berlabel confidence)
9. Lighting Analysis
10. Color Analysis
11. Motion Analysis (temporal: continuity, subject vs camera motion)
12. Editing Analysis (cuts, pacing, rhythm, transitions)
13. Audio Analysis (speech/music/effects/silence — dari transcript bila ada)
14. OCR / On-screen Text
15. Visual Style
16. MASTER RECONSTRUCTED PROMPT — diawali: "Reconstructed prompt — estimated from observable characteristics."
   Blok: [SUBJECT] [ENVIRONMENT] [ACTION] [CAMERA] [LENS / DEPTH OF FIELD] [LIGHTING]
   [COLOR GRADING] [ATMOSPHERE] [STYLE] [MOTION] [COMPOSITION] [TECHNICAL QUALITY],
   lalu gabung jadi satu prompt natural (English).
17. Scene-by-Scene Prompts (per scene: duration, prompt, camera, motion, transition)
18. Negative Prompt (relevan dengan video ini, bukan daftar generik)
19. Character Consistency Prompt (wardrobe, face/hair, anchor traits per scene)
20. Environment Consistency Prompt (location anchors yang harus dijaga)
21. Recommended Generation Settings (aspect ratio, duration, dan adaptasi per engine:
    Generic, Kling, Veo, Runway, Sora, Hailuo — natural language bila syntax khusus
    tidak diketahui; JANGAN mengarang parameter engine yang tidak pasti)
22. Confidence / Uncertainty (tiap klaim rekonstruksi: High/Medium/Low + alasan satu baris)
23. Recreation Workflow (langkah recreate: scene order, consistency checklist)"""

MODE_FOCUS = {
    "quick": "Berikan analisis ringkas: genre, visual style, mood, hook (3 detik pertama), pacing, dan 1 paragraf struktur cerita.",
    "standard": "Analisis lengkap: global analysis, scene-by-scene (subject, environment, camera, lighting, color, action, text), camera analysis, transcript & OCR yang tersedia, viral structure (hook, retention, CTA).",
    "deep": "Analisis mendalam seperti standard + temporal analysis (motion continuity, camera motion, editing rhythm), character consistency, reference sheet (character, environment, props, wardrobe, color palette, lighting, camera style).",
    "prompt": "FOKUS: prompt reverse engineering.\n" + REPORT_STRUCTURE,
}

IMG_TOKENS_EST = 1000  # labeled estimate, bukan harga


def estimate_cost(n_images: int, text_chars: int, pricing: dict | None = None) -> dict:
    est_tokens = n_images * IMG_TOKENS_EST + text_chars // 4
    out = {"images": n_images, "est_input_tokens": est_tokens}
    rate = (pricing or {}).get("per_1m_input")
    if rate:
        try:
            out["est_cost"] = f"~${est_tokens / 1_000_000 * float(rate):.4f} (dari harga yang Anda isi)"
        except (ValueError, TypeError):
            out["est_cost"] = "unavailable — harga di settings tidak valid"
    else:
        # ponytail: tidak ada provider yang expose pricing via chat API — jangan mengarang harga.
        out["est_cost"] = "unavailable — cek harga model di dashboard provider"
    return out


def build_messages(meta: dict, scenes: list, frames: list, frame_dir: Path,
                   mode: str, transcript: list | None = None,
                   ocr: list | None = None, analysis: dict | None = None):
    text = (
        f"Mode: {mode}\n\nVIDEO METADATA (observed, dari ffprobe):\n"
        f"{json.dumps(meta, indent=2)}\n\nSCENES (observed, dari deteksi scene):\n"
        f"{json.dumps(scenes, indent=2)}\n\nFRAME TIMESTAMPS (detik):\n"
        f"{json.dumps([f['timestamp'] for f in frames])}\n"
    )
    if analysis:
        text += ("\nMEASURED ANALYSIS (engine, observed — bukan tebakan):\n"
                 f"{json.dumps(analysis, indent=2)[:4000]}\n")
    if transcript:
        text += "\nTRANSCRIPT (dari Whisper lokal, sinkron per timestamp):\n"
        for s in transcript:
            text += f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}\n"
    if ocr:
        text += "\nOCR TEXT (dari Tesseract pada frame):\n"
        for o in ocr[:80]:
            text += f"[{o['timestamp']:.1f}s {o['position']}] {o['text']}\n"
    text += f"\n{MODE_FOCUS.get(mode, MODE_FOCUS['standard'])}\n"
    text += "Frame-frame berikut diurut berdasarkan timestamp."
    parts = [{"type": "text", "text": text}]
    for f in frames:
        raw = (frame_dir / f["file"]).read_bytes()
        b64 = base64.b64encode(raw).decode()
        parts.append({"type": "image_url",
                      "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    return text, parts


def _post(base_url: str, api_key: str, model: str, system: str,
          user_parts: list, temperature: float, max_tokens: int,
          extra_headers: dict | None = None) -> str:
    body = {"model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user_parts}],
            "temperature": temperature, "max_tokens": max_tokens}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    headers.update(extra_headers or {})
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            j = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:500]}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Tidak dapat mencapai {base_url}: {e.reason}")
    try:
        return j["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise RuntimeError(f"Respons tak terduga: {str(j)[:300]}")


class AIProvider:
    id = ""
    label = ""

    def chat(self, *, api_key, model, system, user_parts, temperature,
             max_tokens, base_url="") -> str:
        raise NotImplementedError

    def test(self, *, api_key, model, base_url="") -> dict:
        try:
            out = self.chat(api_key=api_key, model=model, system="Reply with exactly: OK",
                            user_parts=[{"type": "text", "text": "ping"}],
                            temperature=0, max_tokens=10, base_url=base_url)
            return {"ok": "OK" in out, "detail": out[:200]}
        except Exception as e:
            return {"ok": False, "detail": str(e)[:300]}


class OpenRouterProvider(AIProvider):
    id, label = "openrouter", "OpenRouter"

    def chat(self, **kw):
        if not kw.get("api_key"):
            raise ValueError("API key kosong — isi di Settings atau .env OPENROUTER_API_KEY")
        base = kw.pop("base_url", "") or config.OPENROUTER_BASE_URL
        return _post(base, extra_headers={"HTTP-Referer": "http://localhost:8000",
                                          "X-Title": "video-reverse-engineer"}, **kw)


class CustomOpenAICompatibleProvider(AIProvider):
    id, label = "custom", "Custom OpenAI-Compatible"

    def chat(self, **kw):
        base = kw.pop("base_url", "")
        if not base:
            raise ValueError("Base URL kosong — isi di Settings (mis. https://api.openai.com/v1)")
        if not kw.get("api_key"):
            raise ValueError("API key kosong — isi di Settings")
        return _post(base, **kw)


class NineRouterProvider(CustomOpenAICompatibleProvider):
    id, label = "ninerouter", "NineRouter"

    def chat(self, **kw):
        kw["base_url"] = kw.get("base_url") or config.NINEROUTER_BASE_URL
        return super().chat(**kw)


class MockProvider(AIProvider):
    """Laporan jujur tanpa AI: hanya data teramati."""
    id, label = "mock", "Mock (tanpa AI)"

    def chat(self, **kw):
        raise NotImplementedError("mock dipakai via mock_report(), bukan chat()")


PROVIDERS = {
    "openrouter": OpenRouterProvider(),
    "ninerouter": NineRouterProvider(),
    "custom": CustomOpenAICompatibleProvider(),
    "mock": MockProvider(),
}

# Model presets per provider untuk automatic model selection (CP5): murah/cepat
# untuk quick, capable untuk deep/prompt. User tetap bisa override di Settings.
MODEL_PRESETS = {
    "openrouter": {"quick": "google/gemini-2.5-flash-lite",
                   "standard": "google/gemini-2.5-flash",
                   "deep": "google/gemini-2.5-pro",
                   "prompt": "google/gemini-2.5-pro"},
    "ninerouter": {"quick": "auto-fast", "standard": "auto",
                   "deep": "auto-quality", "prompt": "auto-quality"},
    "custom": {"quick": "", "standard": "", "deep": "", "prompt": ""},
    "mock": {"quick": "mock", "standard": "mock", "deep": "mock", "prompt": "mock"},
}


def mock_report(meta: dict, scenes: list, frames: list,
                transcript: list | None = None, ocr: list | None = None,
                analysis: dict | None = None) -> str:
    lines = ["# Laporan Analisis (Mock — tanpa AI)", "",
             "> Bagian visual ditandai [Not AI-analyzed]. Hanya data terukur yang ditampilkan.", "",
             "## Metadata (Observed)", ""]
    for k, v in meta.items():
        lines.append(f"- **{k}**: {v}")
    lines += ["", "## Scenes (Observed)", ""]
    for s in scenes:
        lines.append(f"- Scene {s['scene_id']:02d}: {s['start']:.2f}s – {s['end']:.2f}s "
                     f"(durasi {s['duration']:.2f}s)")
    if analysis:
        lines += ["", "## Measured Analysis (Engine)", ""]
        for m in analysis.get("motion", []):
            d = m["detail"].get("likely_direction", "")
            lines.append(f"- Scene {m['scene_id']} motion: **{m['class']}** "
                         f"({m['confidence']}) {d} — {m['method'][:60]}")
        for c in analysis.get("color", []):
            cols = ", ".join(f"{d['hex']} {int(d['coverage']*100)}%" for d in c["dominant"][:3])
            lines.append(f"- Scene {c['scene_id']} warna: {cols}")
        for li in analysis.get("lighting", []):
            lines.append(f"- Scene {li['scene_id']} lighting: {li['key']}, "
                         f"kontras {li['contrast']} ({li['confidence']})")
        ed = analysis.get("editing", {})
        lines.append(f"- Editing: pacing {ed.get('pacing')}, rhythm {ed.get('rhythm')}, "
                     f"{ed.get('cuts_per_minute')} cuts/min")
        cam = analysis.get("camera", {})
        lines.append(f"- Kamera: orientasi {cam.get('orientation', {}).get('value')}, "
                     f"focal length: {cam.get('focal_length', {}).get('method')}")
    lines += ["", "## Transcript (Whisper)", ""]
    if transcript is None:
        lines.append("_STT dilewati (tidak tersedia)._")
    elif not transcript:
        lines.append("_Tidak ada speech terdeteksi / tidak ada audio._")
    else:
        for s in transcript:
            lines.append(f"- [{s['start']:.1f}-{s['end']:.1f}] {s['text']}")
    lines += ["", "## OCR (Tesseract)", ""]
    if ocr is None:
        lines.append("_OCR dilewati (tidak tersedia)._")
    elif not ocr:
        lines.append("_Tidak ada teks terdeteksi pada frame._")
    else:
        for o in ocr[:60]:
            lines.append(f"- [{o['timestamp']:.1f}s {o['position']}] {o['text']}")
    lines += ["", "## Frames", ""]
    for f in frames:
        lines.append(f"- `{f['file']}` @ {f['timestamp']:.2f}s (scene {f.get('scene_id')})")
    lines += ["", "> Pilih provider AI di Settings untuk analisis visual penuh."]
    return "\n".join(lines)


def selftest():
    assert estimate_cost(10, 4000)["images"] == 10
    assert "unavailable" in estimate_cost(1, 100)["est_cost"]
    assert estimate_cost(1000, 0, {"per_1m_input": "0.50"})["est_cost"].startswith("~$0.50")
    assert set(PROVIDERS) == {"openrouter", "ninerouter", "custom", "mock"}
    assert isinstance(PROVIDERS["ninerouter"], CustomOpenAICompatibleProvider)
    assert "Unknown / Cannot determine" in SYSTEM  # §43
    assert "23" in REPORT_STRUCTURE or "23." in REPORT_STRUCTURE  # §44
    try:
        PROVIDERS["custom"].chat(api_key="x", model="m", system="s",
                                 user_parts=[], temperature=0, max_tokens=1)
        raise AssertionError("custom tanpa base_url harus gagal")
    except ValueError:
        pass
    print("ai selftest ok")


if __name__ == "__main__":
    selftest()
