"""Field normalization for names and addresses. Every rule here comes from noise seen in the training data."""
import re
import unicodedata

from .translit import transliterate

_LATIN_MARKS = re.compile(r"[̀-ͯ]")
_DOTTED_ACRONYM = re.compile(r"\b[a-z](?:\.[a-z])+\.?")
_DOMAIN = re.compile(r"\b(?:www\.)?([a-z0-9][a-z0-9-]*)\.(?:com|net|org|co|in|biz|info|fr|io|us)\b")
_ID_NOISE = re.compile(r"#\s*\d+|\b\d{4,}\b")
_NON_ALNUM = re.compile(r"[^0-9a-z]+")
_MIXED = re.compile(r"\b(?=[a-z0-9]*[a-z])(?=[a-z0-9]*[0-9])[a-z0-9]+\b")
_LEET = str.maketrans({"1": "l", "0": "o", "3": "e", "5": "s", "4": "a"})
_MESSRS = re.compile(r"\bm\s*/\s*s\b\.?")
_NUMBER_TOKEN = re.compile(r"^(\d+)[a-z]{0,3}$")
_LEADING_ZEROS = re.compile(r"^0+(?=\d)")  # S2/S3 pad house numbers: "0338" = "338"


def _fix_leet(tok: str) -> str:
    # "char1otte" -> "charlotte"; short tokens such as "a1" or "3rd" are left alone.
    digits = sum(c.isdigit() for c in tok)
    return tok.translate(_LEET) if digits == 1 and len(tok) - digits >= 3 else tok
_POSTCODE = re.compile(r"(?<!\d)(\d{6}|\d{3} \d{3}|\d{5})(?!\d)")

LEGAL = {
    "pvt": "private", "private": "private", "ltd": "limited", "limited": "limited", "llc": "llc", "llp": "llp",
    "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp", "co": "co", "company": "co",
    "plc": "plc", "lp": "lp", "pllc": "pllc", "opc": "opc", "sarl": "sarl", "sas": "sas", "sasu": "sasu",
    "sa": "sa", "eurl": "eurl", "sci": "sci", "snc": "snc", "gmbh": "gmbh", "pte": "pte", "pty": "pty",
}
# Hindi-script spellings of legal words after transliteration.
LEGAL.update({"praivet": "private", "limited": "limited", "elelpi": "llp", "kampani": "co", "inkarporeted": "inc"})

ADDRESS_ABBREV = {
    "rd": "road", "ct": "court", "ave": "avenue", "av": "avenue", "dr": "drive", "ln": "lane", "blvd": "boulevard",
    "bd": "boulevard", "bvd": "boulevard", "pl": "place", "hwy": "highway", "pkwy": "parkway", "cir": "circle",
    "ter": "terrace", "sq": "square", "apt": "apartment", "ste": "suite", "fl": "floor", "flr": "floor",
    "bldg": "building", "imp": "impasse", "chem": "chemin", "rte": "route", "fbg": "faubourg", "nr": "near",
    "opp": "opposite", "mkt": "market", "ngr": "nagar", "sec": "sector", "rly": "railway", "stn": "station",
    "dist": "district", "tq": "taluka", "tal": "taluka", "vill": "village", "po": "post",
    "crs": "cours", "all": "allee", "ch": "chemin", "che": "chemin",
}
ADDRESS_DROP = {"null", "cdp", "none", "na", "no", "n", "hno", "h", "house", "plot", "door", "c", "o", "india", "usa", "us",
                "france"}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky", "louisiana": "la",
    "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc",
    "north dakota": "nd", "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa",
    "rhode island": "ri", "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
}
INDIA_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br", "chhattisgarh": "cg",
    "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp", "jharkhand": "jh",
    "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp", "maharashtra": "mh", "manipur": "mn",
    "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl", "odisha": "od", "orissa": "od", "punjab": "pb",
    "rajasthan": "rj", "sikkim": "sk", "tamil nadu": "tn", "telangana": "tg", "tripura": "tr",
    "uttar pradesh": "up", "uttarakhand": "uk", "uttaranchal": "uk", "west bengal": "wb", "delhi": "dl",
    "jammu and kashmir": "jk", "ladakh": "la", "chandigarh": "ch", "puducherry": "py", "pondicherry": "py",
    "andaman and nicobar islands": "an", "dadra and nagar haveli": "dn", "daman and diu": "dd", "lakshadweep": "ld",
}
INDIA_CODE_ALIASES = {"ct": "cg", "or": "od", "ts": "tg", "ut": "uk", "tl": "tg"}
# French copies of one business name either the region or one of its departments ("Bordeaux, Nouvelle-Aquitaine" vs
# "Bordeaux, Gironde"), so both map to one region code, as US state names and abbreviations do.
FRANCE_REGIONS = {
    "hauts de france": "hdf", "nord": "hdf", "pas de calais": "hdf", "aisne": "hdf", "oise": "hdf", "somme": "hdf",
    "nouvelle aquitaine": "naq", "gironde": "naq", "pyrenees atlantiques": "naq", "landes": "naq",
    "charente maritime": "naq", "pays de la loire": "pdl", "loire atlantique": "pdl", "vendee": "pdl",
    "maine et loire": "pdl", "sarthe": "pdl", "mayenne": "pdl", "ile de france": "idf", "bretagne": "bre",
    "normandie": "nor", "grand est": "ges", "occitanie": "occ", "auvergne rhone alpes": "ara",
    "bourgogne franche comte": "bfc", "centre val de loire": "cvl", "provence alpes cote d azur": "pac",
    "corse": "cor",
}
STATES = {"US": US_STATES, "India": INDIA_STATES, "France": FRANCE_REGIONS}


_PHONETIC_RULES = [
    (re.compile(r"ph"), "f"), (re.compile(r"ck"), "k"), (re.compile(r"[ts]ion"), "shn"), (re.compile(r"ch"), "C"),
    (re.compile(r"c(?=[eiy])"), "s"), (re.compile(r"g(?=[ei])"), "j"), (re.compile(r"c"), "k"), (re.compile(r"q"), "k"),
    (re.compile(r"x"), "ks"), (re.compile(r"sh"), "s"), (re.compile(r"([tdbkg])h"), r"\1"), (re.compile(r"w"), "v"),
    (re.compile(r"z"), "j"), (re.compile(r"[aeiouy]"), ""), (re.compile(r"(.)\1+"), r"\1"), (re.compile(r"j$"), "s"),
]


def phonetic_key(token: str) -> str:
    """Consonant skeleton that equates English spellings with Hindi-script transliterations.

    sunrise/sanraij -> snrs, properties/propartij -> prprts, finance/fainens -> fnns.
    """
    if token.isdigit():
        return token
    for rx, rep in _PHONETIC_RULES:
        token = rx.sub(rep, token)
    return token.replace("C", "c")


def phonetic(text: str) -> str:
    return " ".join(k for k in (phonetic_key(t) for t in text.split()) if k)


def _ascii_lower(s: str) -> str:
    s = transliterate(s)
    return _LATIN_MARKS.sub("", unicodedata.normalize("NFKD", s.lower()))


def normalize_name(raw: str) -> tuple[str, str, bool]:
    """Return (core name, legal forms, looked like a domain)."""
    s = _ascii_lower(raw).replace("&", " and ")
    s = _MESSRS.sub(" ", s)
    s = _DOTTED_ACRONYM.sub(lambda m: m.group(0).replace(".", ""), s)
    is_domain = bool(_DOMAIN.search(s))
    s = _DOMAIN.sub(r"\1", s)
    s = _ID_NOISE.sub(" ", s)
    s = _NON_ALNUM.sub(" ", s)
    s = _MIXED.sub(lambda m: _fix_leet(m.group(0)), s)
    core, legal = [], []
    for tok in s.split():
        if tok in LEGAL:
            if LEGAL[tok]:
                legal.append(LEGAL[tok])
        else:
            core.append(tok)
    if not core and legal:
        core = legal[:]
    return " ".join(core), " ".join(sorted(set(legal))), is_domain


def _segment(seg: str) -> list[str]:
    toks = _NON_ALNUM.sub(" ", seg).split()
    out = []
    for i, tok in enumerate(toks):
        if tok == "st":
            # "Main St" -> street, "Rue St Denis" -> saint
            tok = "street" if i == len(toks) - 1 else "saint"
        elif tok == "r" and i + 1 < len(toks) and toks[i + 1].isalpha():
            tok = "rue"
        else:
            tok = ADDRESS_ABBREV.get(tok, tok)
        if tok not in ADDRESS_DROP:
            out.append(tok)
    return out


def normalize_address(raw: str, country: str) -> tuple[str, str, str]:
    """Return (normalized tokens, house/premise numbers, postcode)."""
    s = _ascii_lower(raw)
    found = _POSTCODE.findall(s)
    code = found[-1].replace(" ", "") if found else ""
    states = STATES.get(country, {})
    segments = [seg.strip() for seg in s.split(",") if seg.strip()]
    toks = []
    for seg in segments:
        key = " ".join(_NON_ALNUM.sub(" ", seg).split())
        if key in states:
            toks.append(states[key])
            continue
        if country == "India" and key in INDIA_CODE_ALIASES:
            toks.append(INDIA_CODE_ALIASES[key])
            continue
        toks.extend(_segment(seg))
    toks = [_LEADING_ZEROS.sub("", t) for t in toks]
    code_key = _LEADING_ZEROS.sub("", code)
    nums = set()
    for t in toks:
        m = _NUMBER_TOKEN.match(t)
        if m and m.group(1) != code_key:
            nums.add(m.group(1))
    return " ".join(toks), " ".join(sorted(nums)), code
