"""OCR lokal via Tesseract pada frame yang sudah diekstrak.
Graceful: tanpa binary tesseract -> OCRError (pipeline skip dengan catatan)."""
import shutil
from pathlib import Path


class OCRError(Exception):
    pass


def _position(x, y, w, h, iw, ih) -> str:
    cx, cy = (x + w / 2) / iw, (y + h / 2) / ih
    v = "top" if cy < 0.33 else "bottom" if cy > 0.66 else "middle"
    hz = "left" if cx < 0.33 else "right" if cx > 0.66 else "center"
    return "center" if (v, hz) == ("middle", "center") else f"{v}-{hz}"


def ocr_frames(frames: list, frame_dir: Path, lang: str = "eng+ind",
               min_conf: int = 40, limit: int = 300) -> list:
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        raise OCRError("pytesseract/Pillow belum terinstall: pip install pytesseract")
    if shutil.which("tesseract") is None:
        raise OCRError("binary tesseract tidak ditemukan (apt install tesseract-ocr)")
    out = []
    for f in frames:
        img = Image.open(frame_dir / f["file"])
        iw, ih = img.size
        d = pytesseract.image_to_data(img, lang=lang,
                                      output_type=pytesseract.Output.DICT)
        n = len(d["text"])
        for i in range(n):
            try:
                conf = float(d["conf"][i])
            except (ValueError, TypeError):
                continue
            text = (d["text"][i] or "").strip()
            if conf < min_conf or not text:
                continue
            out.append({
                "text": text,
                "timestamp": f["timestamp"],
                "position": _position(d["left"][i], d["top"][i],
                                      d["width"][i], d["height"][i], iw, ih),
                "conf": round(conf, 1),
            })
            if len(out) >= limit:
                return out
    return out
