import pytest

from src.metric import entity_f05, macro_f05, oracle_ceiling


def test_pdf_example():
    pred = {"S2-00047", "S2-00193", "S3-00812"}
    true = {"S2-00047", "S3-00812"}
    assert entity_f05(pred, true) == pytest.approx(0.714, abs=1e-3)


@pytest.mark.parametrize(
    "pred,true,expected",
    [
        (set(), set(), 1.0),
        ({"S2-1"}, set(), 0.0),
        (set(), {"S2-1"}, 0.0),
        ({"S2-9"}, {"S2-1"}, 0.0),
        ({"S2-1"}, {"S2-1"}, 1.0),
        ({"S2-1"}, {"S2-1", "S3-2"}, 1.25 / 1.5),
        ({"S2-1", "S2-9"}, {"S2-1"}, 1.25 / 2.25),
    ],
)
def test_entity_cases(pred, true, expected):
    assert entity_f05(pred, true) == pytest.approx(expected)


def test_macro_counts_missing_predictions_as_empty():
    truth = {"a": set(), "b": {"S2-1"}}
    assert macro_f05({}, truth) == pytest.approx(0.5)


def test_oracle_ceiling():
    truth = {"a": set(), "b": {"S2-1", "S3-2"}, "c": {"S2-3"}}
    cands = {"b": {"S2-1", "S2-7"}, "c": set()}
    expected = (1.0 + 1.25 * 1 / (1 + 0.5) + 0.0) / 3
    assert oracle_ceiling(cands, truth) == pytest.approx(expected)
