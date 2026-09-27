import pytest

from src.translit import transliterate


@pytest.mark.parametrize(
    "hindi,latin",
    [
        ("राम", "ram"),
        ("मार्केटिंग", "marketing"),
        ("लिमिटेड", "limited"),
        ("आदित्य", "aditya"),
        ("मॉडर्न", "modarn"),
        ("फाइनेंस", "fainens"),
        ("१२३", "123"),
        ("एलएलपी", "elelpi"),
        ("कंपनी", "kampani"),
        ("प्राइवेट", "praivet"),
    ],
)
def test_words(hindi, latin):
    assert transliterate(hindi) == latin


def test_leaves_latin_untouched():
    assert transliterate("Ram Marketing Pvt Ltd") == "Ram Marketing Pvt Ltd"


def test_mixed_string():
    assert transliterate("राम Traders") == "ram Traders"


@pytest.mark.parametrize(
    "text,latin",
    [
        ("తెలంగాణ", "telangan"),   # Telugu
        ("রাম", "ram"),            # Bengali
        ("ગુજરાત", "gujrat"),      # Gujarati; Hindi-style schwa deletion
        ("ಕರ್ನಾಟಕ", "karnatak"),   # Kannada
    ],
)
def test_other_indic_scripts(text, latin):
    assert transliterate(text) == latin
