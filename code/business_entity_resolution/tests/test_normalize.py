import pytest

from src.normalize import normalize_address, normalize_name, phonetic


@pytest.mark.parametrize(
    "english,hindi_translit",
    [
        ("sunrise properties", "sanraij propartij"),
        ("east finance", "ist fainens"),
        ("modern impex", "modarn impeks"),
        ("baba industries", "baba indastrij"),
        ("lotus black", "lotas blaik"),
        ("jai developers", "jay devalapars"),
        ("red management", "red mainejment"),
    ],
)
def test_phonetic_bridges_transliteration(english, hindi_translit):
    assert phonetic(english) == phonetic(hindi_translit)


@pytest.mark.parametrize(
    "raw,core,legal",
    [
        ("Harbor All Independence, [L.L.C.]", "harbor all independence", "llc"),
        ("Harbor All Índependence, LLC", "harbor all independence", "llc"),
        ("Char1otte Intbrfaith Alliance", "charlotte intbrfaith alliance", ""),
        ("Travel Forex Private Ltd #67495", "travel forex", "limited private"),
        ("Veozeta - 2791794602", "veozeta", ""),
        ("M/s Drexzetalyra", "drexzetalyra", ""),
        ("Itdevelopers.Com", "itdevelopers", ""),
        ("COTTIER, CURRY  AND BROWN [FANNIE]", "cottier curry and brown fannie", ""),
        ("A1 Traders", "a1 traders", ""),
        ("राम मार्केटिंग प्राइवेट लिमिटेड", "ram marketing", "limited private"),
        ("Fractales Amis Groupe S.A.S", "fractales amis groupe", "sas"),
    ],
)
def test_names(raw, core, legal):
    got_core, got_legal, _ = normalize_name(raw)
    assert (got_core, got_legal) == (core, legal)


def test_domain_flag():
    assert normalize_name("wilfordhancock.com")[2] is True
    assert normalize_name("Wilford Hancock")[2] is False


@pytest.mark.parametrize(
    "raw,country,tokens,nums",
    [
        ("8907 MACKENZIE COURT, NC, CHRLOTTE", "US", "8907 mackenzie court nc chrlotte", "8907"),
        ("8907 Mackenzie Court, Charlotte, North Carolina", "US", "8907 mackenzie court charlotte nc", "8907"),
        ("115 HOMEWOOD CT, KEIZER, OR", "US", "115 homewood court keizer or", "115"),
        ("17 CHURCH ST, NULL, VILLAGE OF WARRENS, WI", "US", "17 church street village of warrens wi", "17"),
        ("Plot No 3-107, Hyderabad, TG", "India", "3 107 hyderabad tg", "107 3"),
        ("H.no 3-107, Hyderabad, Telangana", "India", "3 107 hyderabad tg", "107 3"),
        ("63 R. DE DIEPPE, LILLE, Hauts-de-France", "France", "63 rue de dieppe lille hdf", "63"),
        ("50 Rue Desse, Bordeaux, Gironde", "France", "50 rue desse bordeaux naq", "50"),
        ("50 Rue Desse, Bordeaux, Nouvelle-Aquitaine", "France", "50 rue desse bordeaux naq", "50"),
        ("N°24 R. SAINTE CATHERINE, LILLE, Nord", "France", "24 rue sainte catherine lille hdf", "24"),
        ("NO 42 CRS ANATOLE, Bordeaux", "France", "42 cours anatole bordeaux", "42"),
        ("19B Rue Du Lieutenant Mouillie, Saint-Herblain", "France", "19b rue du lieutenant mouillie saint herblain", "19"),
        ("12 Rue St Denis, Paris", "France", "12 rue saint denis paris", "12"),
        ("PLOT NO 0028N/P, K.V.RANGAREDDY", "India", "28n p k v rangareddy", "28"),
        ("0338, DL, DL", "India", "338 dl dl", "338"),
        ("12 Main St, Boston, MA 02134", "US", "12 main street boston ma 2134", "12"),
    ],
)
def test_addresses(raw, country, tokens, nums):
    got_tokens, got_nums, _ = normalize_address(raw, country)
    assert (got_tokens, got_nums) == (tokens, nums)


def test_postcode():
    assert normalize_address("Madhapur, Hyderabad 500 081", "India")[2] == "500081"
    assert normalize_address("1795 Westchester Drive, High Point, NC 27262", "US")[2] == "27262"
