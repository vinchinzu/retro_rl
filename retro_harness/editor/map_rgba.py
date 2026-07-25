"""RGBA compositing helpers for editor map overlays."""

from __future__ import annotations


def outline_rect(
    rgba: bytearray,
    width: int,
    height: int,
    x: int,
    y: int,
    rect_w: int,
    rect_h: int,
    color: tuple[int, int, int, int],
) -> None:
    for px in range(max(0, x), min(x + rect_w, width)):
        for py in (y, y + rect_h - 1):
            if 0 <= py < height:
                offset = (py * width + px) * 4
                rgba[offset : offset + 4] = bytes(color)
    for py in range(max(0, y), min(y + rect_h, height)):
        for px in (x, x + rect_w - 1):
            if 0 <= px < width:
                offset = (py * width + px) * 4
                rgba[offset : offset + 4] = bytes(color)


def blend_rect(
    rgba: bytearray,
    width: int,
    height: int,
    x: int,
    y: int,
    rect_w: int,
    rect_h: int,
    color: tuple[int, int, int, int],
) -> None:
    red, green, blue, alpha = color
    for py in range(max(0, y), min(y + rect_h, height)):
        for px in range(max(0, x), min(x + rect_w, width)):
            offset = (py * width + px) * 4
            inv = 255 - alpha
            rgba[offset] = (rgba[offset] * inv + red * alpha) // 255
            rgba[offset + 1] = (rgba[offset + 1] * inv + green * alpha) // 255
            rgba[offset + 2] = (rgba[offset + 2] * inv + blue * alpha) // 255


def fill_rect(
    rgba: bytearray,
    width: int,
    height: int,
    x: int,
    y: int,
    rect_w: int,
    rect_h: int,
    color: tuple[int, int, int, int],
) -> None:
    red, green, blue, alpha = color
    for py in range(max(0, y), min(y + rect_h, height)):
        row_offset = py * width * 4
        for px in range(max(0, x), min(x + rect_w, width)):
            offset = row_offset + px * 4
            rgba[offset] = red
            rgba[offset + 1] = green
            rgba[offset + 2] = blue
            rgba[offset + 3] = alpha


def draw_layer_marker(
    rgba: bytearray,
    width: int,
    height: int,
    center_x: int,
    center_y: int,
    marker_w: int,
    marker_h: int,
    fill: tuple[int, int, int, int],
    outline: tuple[int, int, int, int],
) -> None:
    x = int(round(center_x - marker_w / 2))
    y = int(round(center_y - marker_h / 2))
    if x >= width or y >= height or x + marker_w < 0 or y + marker_h < 0:
        return
    blend_rect(rgba, width, height, x, y, marker_w, marker_h, fill)
    outline_rect(rgba, width, height, x, y, marker_w, marker_h, outline)
