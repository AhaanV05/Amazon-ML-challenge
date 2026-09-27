"""Devanagari -> Latin transliteration tuned for matching English words written in Hindi script.

Long vowels collapse to short ones (aa -> a, ee -> i, oo -> u) because the targets are English
spellings such as "marketing" or "limited", not scholarly romanization.
"""
import re

_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "f", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "ळ": "l", "व": "v",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h", "ऴ": "zh", "ऱ": "r", "ऩ": "n",
    "क़": "q", "ख़": "kh", "ग़": "g", "ज़": "z", "ड़": "r", "ढ़": "rh", "फ़": "f", "य़": "y",
}
_NUKTA_BASE = {"क": "क़", "ख": "ख़", "ग": "ग़", "ज": "ज़", "ड": "ड़", "ढ": "ढ़", "फ": "फ़", "य": "य़"}
_VOWELS = {
    "अ": "a", "आ": "a", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u", "ऋ": "ri",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o", "ऍ": "e", "ॲ": "e", "ऎ": "e", "ऒ": "o",
}
_MATRAS = {
    "ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "ृ": "ri",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ॉ": "o", "ॅ": "e", "ॆ": "e", "ॊ": "o",
}
_VIRAMA, _NUKTA = "्", "़"
_MARKS = {"ं": "n", "ँ": "n", "ः": "h", "॰": "n"}  # ॰ is where Gurmukhi tippi lands
_SKIP = {"ॱ"}  # Gurmukhi addak (doubling mark)
_ANUSVARA_LABIAL = re.compile(r"n(?=[pbm])")
_DIGITS = {chr(0x0966 + i): str(i) for i in range(10)}
_INDIC = re.compile(r"[ऀ-ൿ]")
_INDIC_RUN = re.compile(r"[ऀ-ൿ]+")


def has_devanagari(s: str) -> bool:
    return bool(_INDIC.search(s))


def _to_devanagari(s: str) -> str:
    # Unicode lays out the nine Brahmic blocks in parallel (ISCII order), so a character's
    # offset inside its 128-codepoint block identifies the same letter in Devanagari.
    return "".join(chr(0x0900 + (ord(ch) - 0x0900) % 0x80) if "ঀ" <= ch <= "ൿ" else ch for ch in s)


def _word(w: str) -> str:
    chars = []
    for ch in w:
        if ch == _NUKTA and chars and chars[-1] in _NUKTA_BASE:
            chars[-1] = _NUKTA_BASE[chars[-1]]
        else:
            chars.append(ch)
    chars = [c for c in chars if c not in _SKIP]
    out = []
    pending = False            # last consonant still carries its inherent vowel
    pending_after_vowel = False  # ...and that consonant followed a vowel sound
    last_was_vowel = after_virama = in_cluster = False
    last = ""

    def flush_schwa() -> None:
        nonlocal pending, last_was_vowel
        if pending:
            out.append("a")
            last_was_vowel = True
        pending = False

    for i, ch in enumerate(chars):
        if ch in _CONSONANTS:
            if pending:
                # Schwa deletion (V C _ C V): बरसात -> barsat, एलएलपी -> elelpi.
                nxt = chars[i + 1] if i + 1 < len(chars) else ""
                if pending_after_vowel and (nxt in _MATRAS or nxt in _VOWELS):
                    pending, last_was_vowel = False, False
                else:
                    flush_schwa()
            out.append(_CONSONANTS[ch])
            pending, pending_after_vowel = True, last_was_vowel
            in_cluster, last, after_virama, last_was_vowel = after_virama, ch, False, False
        elif ch in _MATRAS:
            out.append(_MATRAS[ch])
            pending, after_virama, last_was_vowel = False, False, True
        elif ch == _VIRAMA:
            pending, after_virama = False, True
        elif ch in _VOWELS:
            # A consonant before an independent vowel closes its syllable.
            out.append(_VOWELS[ch])
            pending, after_virama, last_was_vowel = False, False, True
        else:
            flush_schwa()
            out.append(_MARKS.get(ch) or _DIGITS.get(ch) or ch)
            last_was_vowel = after_virama = False
    # Final inherent vowel is silent in Hindi (राम -> ram), except after a y/r/v cluster (आदित्य -> aditya).
    if pending and in_cluster and last in ("य", "र", "व"):
        out.append("a")
    return _ANUSVARA_LABIAL.sub("m", "".join(out))


def transliterate(s: str) -> str:
    """Transliterate any of the nine Indic scripts in s to Latin; Latin text passes through."""
    if not has_devanagari(s):
        return s
    return _INDIC_RUN.sub(lambda m: _word(_to_devanagari(m.group(0))), s)
