import re
import unicodedata

_LATIN_MARKS = re.compile(r"[̀-ͯ]")
_NON_ALNUM = re.compile(r"[^0-9a-zऀ-ॿ]+")
_POSTCODE = re.compile(r"(?<!\d)(\d{6}|\d{3} \d{3}|\d{5})(?!\d)")
_NUMBER = re.compile(r"\d+")


def basic_norm(s: str) -> str:
    """Lowercase, strip Latin accents (keeping Devanagari vowel signs), collapse non-alphanumerics."""
    s = _LATIN_MARKS.sub("", unicodedata.normalize("NFKD", s.lower()))
    return _NON_ALNUM.sub(" ", s).strip()


def postcode(address: str) -> str:
    found = _POSTCODE.findall(address)
    return found[-1].replace(" ", "") if found else ""


def numbers(address: str) -> set[str]:
    code = postcode(address)
    return {n for n in _NUMBER.findall(address) if n != code}
