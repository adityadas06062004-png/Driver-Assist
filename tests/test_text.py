import numpy as np
import pytest

from drowsiness import text as T


def canvas(value, w=260, h=70):
    return np.full((h, w, 3), value, np.uint8)


def test_light_text_on_dark_and_dark_text_on_light():
    dark, light = canvas(15), canvas(240)
    T.draw_text(dark, "Risk 42", (10, 45), size=24)
    T.draw_text(light, "Risk 42", (10, 45), size=24)
    assert dark.max() > 220            # white-ish ink on the dark picture
    assert light.min() < 40            # near-black ink on the light picture


def test_text_follows_a_background_that_changes_across_the_picture():
    img = np.zeros((70, 520, 3), np.uint8)
    img[:, 260:] = 240
    T.draw_text(img, "Left", (10, 45), size=24)
    T.draw_text(img, "Right", (270, 45), size=24)
    assert img[:, :260].max() > 220 and img[:, 260:].min() < 40


@pytest.mark.parametrize("bg", [10, 128, 245])
def test_coloured_text_keeps_enough_contrast(bg):
    img = canvas(bg)
    T.draw_text(img, "CRITICAL", (10, 45), size=24, color=(0, 0, 255))     # alert red
    ink_lum = T.luminance_bgr(_darkest_or_lightest_ink(img, bg))
    assert T.contrast_ratio(ink_lum, T.luminance_bgr((bg, bg, bg))) >= 3.0


def _darkest_or_lightest_ink(img, bg):
    flat = img.reshape(-1, 3).astype(int)
    far = np.abs(flat - bg).sum(axis=1)
    return tuple(flat[int(far.argmax())])


def test_readable_keeps_a_colour_that_already_has_contrast():
    assert T._readable((0, 255, 0), 0.0) == (0, 255, 0)


def test_centered_text_is_balanced_around_the_point():
    img = canvas(15, w=400)
    T.draw_text(img, "WAKE UP", (200, 45), size=26, anchor="m")
    cols = np.where((img > 150).any(axis=(0, 2)))[0]
    assert abs((cols.min() + cols.max()) / 2 - 200) <= 3


def test_text_partly_or_fully_off_screen_does_not_crash():
    img = canvas(15)
    T.draw_text(img, "edge", (-30, 40), size=24)
    T.draw_text(img, "edge", (250, 40), size=24)
    T.draw_text(img, "gone", (500, 40), size=24)
    T.draw_text(img, "", (10, 40))


def test_text_size_grows_with_font_size():
    assert T.text_size("Risk 100", 40)[0] > T.text_size("Risk 100", 20)[0]


def test_falls_back_to_opencv_when_no_font_is_available(monkeypatch):
    monkeypatch.setattr(T, "_font", lambda px, weight: None)
    img = canvas(15)
    T.draw_text(img, "Fallback", (10, 45), size=24)
    assert img.max() > 150
