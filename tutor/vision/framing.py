"""Send the reading model the page, not the desk.

A camera mounted above a desk spends most of its pixels on the desk. In a real
capture from the board the A4 page filled about 540x640 of a 1600x1200 frame —
15% of the picture, roughly 65 DPI across the page, with handwriting only 25-30
pixels tall. Reliable handwriting OCR wants 150+ DPI.

Cropping cannot invent the missing pixels. What it can do is stop us throwing
away the ones we have: a vision model resizes whatever it is given into a fixed
tile budget, so a full frame spends ~85% of that budget on an empty white desk,
a keyboard and someone's hand, and the equation arrives smaller than it was
photographed. Hand it the page alone and the same budget lands entirely on the
page — worth roughly 2-3x in effective resolution, for free.

Two ways to say where the page is:

    WORKSHEET_ROI=0.18,0.36,0.36,0.55   fixed mount, fixed spot: exact and
                                        deterministic. Prefer this.
    automatic                           find the densest patch of ink that sits
                                        on something bright. Best-effort, and
                                        it refuses rather than guesses.

The automatic path is deliberately timid: if what it found is not page-shaped,
not page-sized, or not brighter than the rest of the frame, it returns the
original image untouched. A wrong crop can hide the student's work entirely,
which is worse than a wasted tile budget.
"""

from __future__ import annotations

import io
import logging

log = logging.getLogger(__name__)

Box = tuple[int, int, int, int]  # left, top, right, bottom, in pixels
Roi = tuple[float, float, float, float]  # x, y, w, h as fractions of the frame

# What a page may look like before we refuse to believe it is one.
MIN_AREA_FRACTION = 0.04
MAX_AREA_FRACTION = 0.70
MIN_ASPECT, MAX_ASPECT = 0.35, 2.8
# How much darker than its surroundings a pixel must be to count as ink (0-255).
INK_DELTA = 16
# Rows/columns holding less than this share of the busiest one are margin.
DENSITY_FLOOR = 0.08
# The box lands on the writing, not the paper's edge, so leave room for the
# ascenders and stray strokes the density trim shaved off.
PAD_FRACTION = 0.10

_warned_no_pillow = False


def prepare_for_reading(
    jpeg: bytes,
    *,
    roi: Roi | None = None,
    auto: bool = True,
    target_px: int = 1024,
    quality: int = 92,
) -> bytes:
    """Crop to the worksheet and size it for a reading model.

    Never raises and never returns nothing: on any doubt the original frame is
    passed through, because a lesson with a badly framed photo still works and a
    lesson with a blank one does not.
    """
    image = _open(jpeg)
    if image is None:
        return jpeg

    box = _roi_box(image.size, roi) if roi else (_find_page(image) if auto else None)
    if box is None:
        return jpeg

    cropped = image.crop(box)
    width, height = cropped.size
    if not width or not height:
        return jpeg

    # Enlarging adds no information. It does stop the model's own resizing from
    # landing below the stroke width, which is where handwriting turns to mush.
    scale = min(target_px / max(width, height), 3.0)
    if scale > 1.05:
        from PIL import Image

        cropped = cropped.resize(
            (round(width * scale), round(height * scale)), Image.LANCZOS
        )

    buffer = io.BytesIO()
    cropped.convert("RGB").save(buffer, format="JPEG", quality=quality)
    out = buffer.getvalue()
    log.info(
        "cropped %dx%d → %dx%d (%d → %d bytes)",
        *image.size, *cropped.size, len(jpeg), len(out),
    )
    return out


# --- locating the page --------------------------------------------------------


def _roi_box(size: tuple[int, int], roi: Roi) -> Box | None:
    width, height = size
    x, y, w, h = roi
    left, top = int(x * width), int(y * height)
    right, bottom = int((x + w) * width), int((y + h) * height)
    left, top = max(0, left), max(0, top)
    right, bottom = min(width, right), min(height, bottom)
    if right - left < 16 or bottom - top < 16:
        log.warning("WORKSHEET_ROI %s selects almost nothing; ignoring it", roi)
        return None
    return left, top, right, bottom


def _find_page(image) -> Box | None:
    """The densest run of ink that sits on something bright. None if unsure."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a core dependency
        return None

    from PIL import Image

    full_w, full_h = image.size
    small = image.convert("L")
    scale = 320 / max(full_w, full_h)
    if scale < 1:
        small = small.resize((max(1, round(full_w * scale)), max(1, round(full_h * scale))),
                             Image.BILINEAR)
    gray = np.asarray(small, dtype=np.float32)
    if gray.size < 64:
        return None

    ink = gray < (_box_blur(gray, radius=max(2, min(gray.shape) // 24)) - INK_DELTA)
    if not ink.any():
        return None

    rows = _dense_span(ink.sum(axis=1))
    cols = _dense_span(ink.sum(axis=0))
    if rows is None or cols is None:
        return None
    top, bottom = rows
    left, right = cols

    # The page is bright; a keyboard or a hand full of edges is not. Compare the
    # patch's PAPER, not its average — a page covered in writing averages darker
    # than the empty desk around it, which is the whole reason we want it.
    patch = gray[top : bottom + 1, left : right + 1]
    if patch.size == 0 or float(np.percentile(patch, 75)) < float(np.percentile(gray, 40)):
        log.info("auto-crop: the busiest region is darker than the frame; keeping the whole photo")
        return None

    height, width = gray.shape
    pad_y = int((bottom - top + 1) * PAD_FRACTION)
    pad_x = int((right - left + 1) * PAD_FRACTION)
    top, bottom = max(0, top - pad_y), min(height - 1, bottom + pad_y)
    left, right = max(0, left - pad_x), min(width - 1, right + pad_x)

    box_w, box_h = right - left + 1, bottom - top + 1
    area = (box_w * box_h) / float(width * height)
    aspect = box_w / box_h if box_h else 0
    if not (MIN_AREA_FRACTION <= area <= MAX_AREA_FRACTION):
        log.info("auto-crop: region is %.0f%% of the frame; keeping the whole photo", area * 100)
        return None
    if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
        log.info("auto-crop: region is %.2f:1; not page-shaped, keeping the whole photo", aspect)
        return None

    back = max(full_w, full_h) / max(width, height)
    return (
        int(left * back), int(top * back),
        min(full_w, int((right + 1) * back)), min(full_h, int((bottom + 1) * back)),
    )


def _dense_span(counts) -> tuple[int, int] | None:
    """Trim the margins: keep the contiguous run around the busiest line."""
    import numpy as np

    peak = float(counts.max())
    if peak <= 0:
        return None
    busy = np.flatnonzero(counts >= peak * DENSITY_FLOOR)
    if busy.size == 0:
        return None
    return int(busy[0]), int(busy[-1])


def _box_blur(a, radius: int):
    """Local mean via an integral image — the background to compare ink against."""
    import numpy as np

    r = max(1, int(radius))
    padded = np.pad(a, r, mode="edge")
    total = padded.cumsum(0).cumsum(1)
    total = np.pad(total, ((1, 0), (1, 0)))
    h, w = a.shape
    k = 2 * r + 1
    window = (
        total[k : k + h, k : k + w]
        - total[0:h, k : k + w]
        - total[k : k + h, 0:w]
        + total[0:h, 0:w]
    )
    return window / float(k * k)


def _open(jpeg: bytes):
    global _warned_no_pillow
    try:
        from PIL import Image
    except ImportError:
        if not _warned_no_pillow:
            _warned_no_pillow = True
            log.warning(
                "auto-crop needs Pillow (pip install -e \".[sim]\"); "
                "sending the full frame to the reading model instead"
            )
        return None
    try:
        image = Image.open(io.BytesIO(jpeg))
        image.load()
        return image
    except Exception as e:  # noqa: BLE001 — an undecodable frame is the VLM's problem
        log.warning("could not decode the frame for cropping: %s", e)
        return None


def parse_roi(text: str) -> Roi | None:
    """WORKSHEET_ROI="x,y,w,h" as fractions of the frame, e.g. 0.18,0.36,0.36,0.55."""
    parts = [p.strip() for p in (text or "").split(",") if p.strip()]
    if len(parts) != 4:
        return None
    try:
        x, y, w, h = (float(p) for p in parts)
    except ValueError:
        log.warning("WORKSHEET_ROI=%r is not four numbers; ignoring it", text)
        return None
    if not (0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1):
        log.warning("WORKSHEET_ROI=%r is out of the 0-1 range; ignoring it", text)
        return None
    return x, y, w, h
