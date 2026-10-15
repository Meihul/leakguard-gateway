import unicodedata

_ZERO_WIDTH = ["​", "‌", "‍", "⁠", "﻿"]


def normalize(text):
    for ch in _ZERO_WIDTH:
        text = text.replace(ch, "")
    return unicodedata.normalize("NFKC", text)
