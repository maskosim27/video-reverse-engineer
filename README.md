# Video Reverse Engineering Analyzer

Aplikasi web lokal untuk menganalisis video pendek (upload atau URL) dan
me-reverse-engineer kemungkinan prompt AI video generator yang bisa
menghasilkan video serupa — dengan aturan anti-fabrikasi yang ketat:
tidak pernah mengklaim tahu prompt aslinya, semua dinyatakan sebagai
*estimated / inferred / reconstructed*, dan yang tidak bisa ditentukan
ditulis apa adanya sebagai `Unknown / Cannot determine from available evidence.`

Alur: **upload/URL → FFmpeg → deteksi scene → sampling frame → metadata →
analisis gerak/warna/lighting (terukur) → Whisper STT → Tesseract OCR →
AI vision → laporan 23-seksi**.

## Fitur

- **Input modular per platform** — Upload file lokal (fallback wajib, selalu
  tersedia), TikTok, YouTube/Shorts, Facebook, X/Twitter, Threads via yt-dlp.
  Setiap downloader adalah modul terpisah: kegagalan satu platform tidak
  merusak pipeline atau platform lain.
- **Engine analisis terukur** (`backend/app/analysis.py`) — klasifikasi gerak
  per scene (static / subject-motion / camera-motion + arah pan/tilt/zoom dari
  probe 3fps), warna dominan, lighting (high/low-key + kontras), editing rhythm
  (durasi shot rata-rata, cuts/menit), estimasi kamera yang jujur (focal length
  = "Cannot determine from available evidence", tidak pernah dikarang).
- **Provider AI** — OpenRouter, NineRouter (gateway lokal OpenAI-compatible,
  default `http://127.0.0.1:20128/v1`), Custom OpenAI-compatible (base URL
  bebas), dan Mock (laporan jujur dari data teramati, tanpa API key).
  Pemilihan model otomatis per provider+mode bila user mengosongkan model.
- **Multi-AI consensus** — jalankan beberapa provider sekaligus (background,
  kegagalan satu provider terisolasi), lalu satu provider juri menggabungkan
  hasil: Agreements / Disagreements / Merged Best-Estimate / Recommended Prompt.
- **Iterative recreation** — upload video hasil generate dari prompt, bandingkan
  dengan video asli (6 skor kemiripan: visual, motion, composition, color,
  camera, character), dapatkan improved prompt + daftar perbedaan konkret;
  riwayat iterasi tersimpan sebagai rantai parent.
- **Skills** — import `SKILL.md` dari URL GitHub (blob → raw otomatis),
  frontmatter `name/description/version`, enable/disable, versioning + rollback,
  custom system prompt. Skills adalah **teks konteks AI saja, tidak pernah
  dieksekusi** (tanpa eval/exec/subprocess).
- **STT lokal** (faster-whisper, model `WHISPER_MODEL`) dan **OCR Tesseract**
  (`OCR_LANG`, default `eng+ind`) pada frame — keduanya graceful skip bila
  model/binary tidak tersedia.
- **Export** — Markdown, TXT, JSON, dan ZIP (video + laporan + transcript +
  ocr.json + metadata + scenes + engine_analysis.json + frame sampel).
- **Cache** — video dengan hash sama tidak diproses ulang; frame disalin ke
  direktori job baru agar endpoint frames & multi-AI tetap berfungsi.

## Kebutuhan

- Python 3.11+
- **FFmpeg + ffprobe** (wajib — tanpa ini tidak ada pemrosesan video)
- Tesseract OCR binary (opsional; `apt install tesseract-ocr tesseract-ocr-ind`)
- API key provider (opsional — provider Mock bisa dipakai tanpa key)

## Instalasi

### Cara 1 — Docker (paling mudah)

```bash
cp .env.example .env
# isi SECRET_KEY di .env (wajib agar API key tersimpan awet setelah restart):
# python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
docker compose up -d --build
# buka http://localhost:3000
```

Docker image sudah mencakup FFmpeg, Tesseract (eng+ind), dan dependensi Python.

### Cara 2 — Lokal (venv)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# FFmpeg & tesseract harus terinstal di sistem:
#   Ubuntu/Debian: sudo apt install ffmpeg tesseract-ocr tesseract-ocr-ind
cp .env.example .env   # isi SECRET_KEY
cd backend && uvicorn main:app --port 8000
# buka http://localhost:8000
```

### Cara 3 — Development

Sama seperti cara 2, dengan reload otomatis:

```bash
cd backend && uvicorn main:app --port 8000 --reload
```

Frontend adalah 1 file `backend/static/index.html` (vanilla JS + CSS, tanpa
build, tanpa CDN — fully offline); edit langsung, refresh browser.

## Konfigurasi API key

**Jangan pernah taruh API key di chat / log / file yang di-commit.**
Dua cara:

1. **Via UI Settings** (disarankan) — key tersimpan di SQLite terenkripsi
   Fernet dengan `SECRET_KEY` dari `.env`. Frontend tidak pernah menerima key
   asli (hanya status terisi/tidak). ⚠️ Tanpa `SECRET_KEY` yang stabil, key
   tersimpan hilang setiap restart (kunci enkripsi ephemeral).
2. **Via `.env`** — `OPENROUTER_API_KEY`, `NINEROUTER_API_KEY`, `CUSTOM_API_KEY`,
   dsb. DB settings meng-override `.env`.

**OpenRouter** — daftar di openrouter.ai, buat key, isi di Settings →
provider OpenRouter (model default `google/gemini-2.5-flash`).

**NineRouter** — gateway lokal; jalankan NineRouter di mesin Anda
(default `http://127.0.0.1:20128/v1`), isi API key-nya di Settings →
provider NineRouter.

**Custom** — provider OpenAI-compatible apa pun: isi Base URL + API key + nama
model di Settings.

## Penggunaan

1. **Analisis video** — tab Upload (file lokal) atau URL (TikTok/YouTube/FB/X/
   Threads). Pilih provider + mode (quick/standard/deep/prompt-only), klik
   Process. Progress terpantau di daftar project; klik untuk melihat laporan,
   timeline scene, transcript, OCR, tab Analysis (data terukur), dan export.
2. **Multi-AI** — di panel "Advanced AI", centang 2+ provider, pilih juri
   consensus, klik Run. Hasil per provider + consensus muncul di tab Multi-AI.
3. **Iterative recreation** — salin "reconstructed prompt" dari laporan,
   generate video di tool AI favorit Anda, upload hasilnya via panel Compare,
   klik Compare. Anda dapat 6 skor kemiripan, perbedaan konkret, dan improved
   prompt untuk iterasi berikutnya. Riwayat iterasi tersimpan.
4. **Skills** — Settings → Skills → tempel URL GitHub ke file `SKILL.md`
   (mis. `https://github.com/user/repo/blob/main/SKILL.md`), Import.
   Aktifkan skill yang ingin dipakai sebagai konteks AI. Versi lama bisa
   di-rollback. Custom system prompt bisa ditambahkan manual.
5. **Cost** — isi `pricing_per_1m` (USD per 1 juta token) di Settings untuk
   estimasi biaya jujur. Tanpa harga dari Anda, biaya = "unavailable" —
   aplikasi tidak pernah mengarang harga model.

## Troubleshooting

| Gejala | Penyebab & solusi |
|---|---|
| `SECRET_KEY kosong` di log | Isi `SECRET_KEY` di `.env` (lihat Instalasi). Tanpa ini key tersimpan hilang saat restart. |
| STT "dilewati: gagal load model" | faster-whisper gagal unduh/load model (offline / proxy). Video tetap diproses tanpa transcript. Set `WHISPER_MODEL=tiny` untuk ukuran kecil. |
| OCR "binary tesseract tidak ditemukan" | Install `tesseract-ocr` (+ `tesseract-ocr-ind` untuk Indonesia). Di Docker sudah tersedia. |
| Download URL gagal (SSL / tidak bisa fetch) | Jaringan VM/server memblokir atau sertifikat bermasalah. Coba unduh manual lalu gunakan tab Upload. Kegagalan terisolasi per platform — platform lain tetap bisa dicoba. |
| "API key … kosong" | Isi key di Settings untuk provider tersebut, atau pakai Mock. |
| Provider timeout / rate limit | Job ditandai failed dengan pesan jelas; kurangi `max_tokens` atau coba lagi nanti. |
| Frame tidak muncul di job hasil cache | Sudah diperbaiki (cache-hit menyalin frame). Bila menemui job lama yang rusak, hapus project-nya dan proses ulang. |
| Laporan terasa "mengarang" | Periksa tab Analysis — semua klaim AI seharusnya berlabel [Observed]/[Inferred]/[Speculation] dan prompt berlabel "Reconstructed prompt — estimated…". Laporkan bila ada klaim pasti tentang prompt asli. |

## Arsitektur

```
backend/
  main.py            # FastAPI: routes, settings terenkripsi, job queue (2 thread)
  app/
    config.py        # env + paths (./data)
    db.py            # SQLite stdlib: jobs, settings, skills, recreations
    pipeline.py      # orkestrasi: download → probe → scenes → frames → STT →
                     #   OCR → analysis → AI → report (state jelas per tahap)
    sources.py       # VideoSource per platform (yt-dlp), UploadSource fallback
    video.py         # probe/scenes/frames murni FFmpeg (tanpa OpenCV)
    analysis.py      # engine terukur: motion/color/lighting/rhythm/camera
    stt.py           # faster-whisper (graceful skip)
    ocr.py           # tesseract (graceful skip)
    ai.py            # AIProvider interface, prompt §43, REPORT_STRUCTURE §44,
                     #   MODEL_PRESETS, estimasi biaya jujur
    multi.py         # multi-AI background + consensus juri
    compare.py       # compare original vs generated, 6 skor, improved prompt
    skills.py        # import SKILL.md GitHub, versioning, sandbox teks-only
    exporters.py     # MD/TXT/JSON/ZIP
  static/index.html  # frontend 1 file (vanilla JS)
data/                # videos/, frames/, app.db, skills/ (di-mount di Docker)
```

**Prinsip kegagalan terisolasi:** setiap downloader platform, setiap provider
AI, setiap tahap pipeline (STT/OCR/AI) dibungkus try/except sendiri — satu
kegagalan tidak merusak yang lain, dan status/error selalu jelas di UI.

## Keamanan & privasi

- API key terenkripsi (Fernet) di server; **tidak pernah dikirim ke frontend**,
  tidak pernah ditulis ke log (hanya status terisi/tidak).
- Tidak ada analytics, telemetri, atau upload otomatis ke mana pun.
- ⚠️ Bila memakai provider AI eksternal (OpenRouter/dsb), frame video, audio/
  transcript, dan teks analisis **dikirim ke provider tersebut** — jangan
  proses video sensitif dengan provider cloud; gunakan Mock atau NineRouter
  lokal bila perlu privasi penuh.
- Skills dari GitHub diperlakukan sebagai **teks saja** — tidak pernah
  dieksekusi sebagai kode.
- `SECRET_KEY` adalah satu-satunya rahasia deployment: jaga seperti password.

## Status verifikasi (2026-09-30)

Terverifikasi di VM: upload → analisis → laporan (Mock), export TXT/MD/JSON/
ZIP, multi-AI + consensus (fake provider), compare 6 skor + iterasi parent
chain, skill import/versioning/rollback, settings enkripsi, cache-hit, motion
engine pada video sintetis (static / subject-motion / pan-right).

Belum terverifikasi di lingkungan ini: download URL sungguhan per platform
(SSL VM bermasalah), transkripsi Whisper nyata, OCR nyata, Docker Compose run
(tidak ada docker di VM), dan pemanggilan provider AI sungguhan (tanpa key).
