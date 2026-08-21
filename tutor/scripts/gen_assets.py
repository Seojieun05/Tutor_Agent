"""Render simulator worksheet JPEGs (PIL) + a placeholder hint WAV.

Images use ASCII math only, so no Korean font is required: the printed
problem on top, "handwritten" student work lines below.
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

# 시뮬레이터가 쓸 이미지·음성 자산 폴더.
ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "simulator" / "assets"

# 학생 풀이가 단계별로 달라지는 가짜 문제지 장면들.
SCENES = {
    # lin_001 progression: wrong sign (sign_flip_on_move) → corrected → solved
    "lin_001_problem": ("Solve: 3x + 5 = 20", []),
    "lin_001_wrong_sign": ("Solve: 3x + 5 = 20", ["3x = 20 + 5"]),
    "lin_001_step1_ok": ("Solve: 3x + 5 = 20", ["3x = 15"]),
    "lin_001_solved": ("Solve: 3x + 5 = 20", ["3x = 15", "x = 5"]),
    # novel equation, same template (TEMPLATE tier demo)
    "novel_lin_problem": ("Solve: 4x + 1 = 13", []),
    # quadratic (EXACT tier on quad_001)
    "quad_001_problem": ("Solve: x^2 - 5x + 6 = 0", []),
    "quad_001_wrong_factor": ("Solve: x^2 - 5x + 6 = 0", ["(x + 2)(x + 3) = 0"]),
    # derivative (EXACT tier on deriv_001)
    "deriv_001_problem": ("Differentiate: f(x) = x^3 + 2x", []),
    "deriv_001_wrong_exp": ("Differentiate: f(x) = x^3 + 2x", ["f'(x) = 3x^3 + 2"]),
}


# 장면들을 JPEG로 그려 낸다.
def render_images() -> list[Path]:
    from PIL import Image, ImageDraw

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for name, (problem, work) in SCENES.items():
        img = Image.new("RGB", (960, 640), "white")
        draw = ImageDraw.Draw(img)
        try:
            from PIL import ImageFont

            font_big = ImageFont.load_default(size=40)
            font_work = ImageFont.load_default(size=36)
        except Exception:
            font_big = font_work = None
        draw.text((60, 60), problem, fill="black", font=font_big)
        draw.line((60, 130, 900, 130), fill="grey", width=2)
        y = 200
        for line in work:
            # slight offset per line to look handwritten-ish
            draw.text((100, y), line, fill=(20, 20, 90), font=font_work)
            y += 90
        path = ASSETS_DIR / f"{name}.jpg"
        img.save(path, "JPEG", quality=92)
        out.append(path)
    return out


# "힌트 주세요" 음성 파일을 만든다(TTS가 있으면 합성, 없으면 무음).
def render_hint_wav() -> Path:
    """1s 440Hz placeholder tone, 16kHz/16-bit/mono. In echo mode any
    utterance counts as a hint request; for live STT record real speech."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    path = ASSETS_DIR / "hint.wav"
    rate = 16000
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        for i in range(rate):
            sample = int(12000 * math.sin(2 * math.pi * 440 * i / rate))
            w.writeframes(struct.pack("<h", sample))
    return path


# 커맨드라인 진입점.
def main() -> None:
    images = render_images()
    wav = render_hint_wav()
    for p in images + [wav]:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
