"""Cropping the desk away before the reading model sees the frame.

Measured from a real capture: an A4 page filled 540x640 of a 1600x1200 frame,
so ~85% of the model's tile budget went on a white desk, a keyboard and a hand,
and the handwriting arrived smaller than it was photographed. Cropping recovers
those pixels. It cannot create any, which is why every test here is really
about the same thing: crop when we know where the page is, and keep the whole
photo whenever we do not. A wrong crop hides the student's work entirely.
"""

import io

import pytest

from tutor.vision.framing import parse_roi, prepare_for_reading

Image = pytest.importorskip("PIL.Image", reason="auto-crop needs Pillow")


def desk_with_page(
    size=(1600, 1200), page=(330, 180, 820, 790), desk=238, paper=250, ink=25,
    clutter=True,
):
    """A bright page on a bright desk — the case that makes this hard."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", size, (desk, desk, desk))
    draw = ImageDraw.Draw(img)
    draw.rectangle(page, fill=(paper, paper, paper))
    left, top, right, bottom = page
    for i in range(6):  # printed lines + a handwritten one
        y = top + 60 + i * 55
        draw.line([(left + 40, y), (right - 60, y)], fill=(ink, ink, ink), width=7)
    if clutter:  # a keyboard: darker, and busy with edges
        draw.rectangle((420, 850, 1150, 1150), fill=(170, 170, 170))
        for i in range(14):
            x = 430 + i * 50
            draw.rectangle((x, 870, x + 40, 1130), outline=(90, 90, 90), width=4)
    return to_jpeg(img)


def to_jpeg(img) -> bytes:
    buffer = io.BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def size_of(jpeg: bytes):
    from PIL import Image

    with Image.open(io.BytesIO(jpeg)) as im:
        return im.size


class TestExplicitRoi:
    def test_a_fixed_mount_gets_exactly_the_region_it_asked_for(self):
        jpeg = desk_with_page()
        out = prepare_for_reading(jpeg, roi=(0.25, 0.25, 0.5, 0.5), target_px=0)
        assert size_of(out) == (800, 600)

    def test_the_crop_is_enlarged_to_the_target(self):
        out = prepare_for_reading(desk_with_page(), roi=(0.25, 0.25, 0.5, 0.5),
                                  target_px=1024)
        assert max(size_of(out)) == 1024

    def test_enlargement_is_capped(self):
        """Upscaling adds no information; past ~3x it only adds bytes."""
        out = prepare_for_reading(desk_with_page(), roi=(0.4, 0.4, 0.05, 0.05),
                                  target_px=4000)
        assert max(size_of(out)) <= 80 * 3 + 1

    def test_a_degenerate_roi_is_ignored_rather_than_obeyed(self):
        jpeg = desk_with_page()
        assert prepare_for_reading(jpeg, roi=(0.5, 0.5, 0.001, 0.001)) == jpeg


class TestAutoCrop:
    def test_it_finds_the_writing_on_a_desk(self):
        """It lands on the ink, not the paper's edge — which is what we want:
        a student writing in one corner should not buy back the empty half."""
        jpeg = desk_with_page(clutter=False)
        out = prepare_for_reading(jpeg, auto=True, target_px=0)
        width, height = size_of(out)
        assert (width, height) != (1600, 1200), "the desk should have been cropped away"
        # the writing spans 390x283; the crop is that plus a margin
        assert 400 <= width <= 580, width
        assert 290 <= height <= 440, height
        assert width * height < 1600 * 1200 * 0.15

    def test_cropping_is_what_makes_the_writing_bigger(self):
        """The whole point, stated as a ratio: the page's share of the frame."""
        jpeg = desk_with_page(clutter=False)
        before = 490 * 610 / (1600 * 1200)
        out = prepare_for_reading(jpeg, auto=True, target_px=0)
        width, height = size_of(out)
        after = 490 * 610 / (width * height)
        assert after > before * 3

    def test_it_keeps_the_whole_photo_when_the_page_is_not_page_shaped(self):
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (1600, 1200), (238, 238, 238))
        draw = ImageDraw.Draw(img)
        draw.rectangle((40, 560, 1560, 640), fill=(250, 250, 250))  # 19:1
        draw.line([(80, 600), (1520, 600)], fill=(25, 25, 25), width=8)
        jpeg = to_jpeg(img)
        assert prepare_for_reading(jpeg, auto=True) == jpeg

    def test_it_keeps_the_whole_photo_when_the_ink_is_everywhere(self):
        from PIL import Image, ImageDraw

        img = Image.new("RGB", (1600, 1200), (240, 240, 240))
        draw = ImageDraw.Draw(img)
        for i in range(24):  # writing edge to edge: nothing to crop to
            draw.line([(20, 30 + i * 48), (1580, 30 + i * 48)], fill=(20, 20, 20), width=6)
        jpeg = to_jpeg(img)
        assert prepare_for_reading(jpeg, auto=True) == jpeg

    def test_a_blank_frame_is_passed_through(self):
        jpeg = to_jpeg(Image.new("RGB", (800, 600), (245, 245, 245)))
        assert prepare_for_reading(jpeg, auto=True) == jpeg

    def test_auto_off_means_untouched(self):
        jpeg = desk_with_page()
        assert prepare_for_reading(jpeg, auto=False) == jpeg

    def test_an_undecodable_frame_never_raises(self):
        junk = b"\xff\xd8not really a jpeg"
        assert prepare_for_reading(junk, auto=True) == junk


class TestParseRoi:
    @pytest.mark.parametrize("text", ["", "0.1,0.2", "a,b,c,d", "0.1,0.2,0.3",
                                      "0.1,0.2,0,0.5", "1.2,0.2,0.3,0.4"])
    def test_rubbish_is_refused(self, text):
        assert parse_roi(text) is None

    def test_a_good_one_is_parsed(self):
        assert parse_roi(" 0.18, 0.36 ,0.36,0.55 ") == (0.18, 0.36, 0.36, 0.55)


class TestWiring:
    def test_the_recognizer_crops_before_it_asks_the_model(self, monkeypatch):
        from tutor.config import Settings
        from tutor.vision.recognizer import Recognition, Recognizer

        seen = {}

        class SpyLLM:
            def complete_json(self, *, purpose, system, user, images=(), schema):
                seen["bytes"] = len(images[0])
                return Recognition(problem_text="x")

        jpeg = desk_with_page(clutter=False)
        settings = Settings(auto_crop=True, crop_target_px=1024)
        Recognizer(SpyLLM(), settings).recognize(jpeg)
        assert seen["bytes"] != len(jpeg), "the model was handed the uncropped frame"

    def test_without_settings_the_frame_is_sent_as_photographed(self):
        from tutor.vision.recognizer import Recognition, Recognizer

        seen = {}

        class SpyLLM:
            def complete_json(self, *, purpose, system, user, images=(), schema):
                seen["bytes"] = images[0]
                return Recognition(problem_text="x")

        jpeg = desk_with_page()
        Recognizer(SpyLLM()).recognize(jpeg)
        assert seen["bytes"] == jpeg
