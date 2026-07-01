import pytest
from src.fresh_rag.metrics import exact_match, f1_score


class TestExactMatch:
    def test_identical_strings(self):
        assert exact_match("Paris", "Paris") == 1.0

    def test_case_insensitive(self):
        assert exact_match("paris", "PARIS") == 1.0

    def test_whitespace_stripped(self):
        assert exact_match("  Paris  ", "Paris") == 1.0

    def test_different_strings(self):
        assert exact_match("Paris", "London") == 0.0

    def test_empty_strings(self):
        assert exact_match("", "") == 1.0

    def test_partial_match_is_zero(self):
        # "Paris" is not exact match for "Paris, France"
        assert exact_match("Paris", "Paris, France") == 0.0


class TestF1Score:
    def test_identical_strings(self):
        assert f1_score("Paris", "Paris") == 1.0

    def test_no_overlap(self):
        assert f1_score("Paris", "London") == 0.0

    def test_partial_overlap(self):
        # "Donald Trump" vs "Trump" -> precision 1.0, recall 0.5
        assert f1_score("Donald Trump", "Trump") == pytest.approx(2/3, rel=1e-3)

    def test_case_insensitive(self):
        assert f1_score("PARIS", "paris") == 1.0

    def test_empty_prediction(self):
        assert f1_score("", "Paris") == 0.0

    def test_empty_gold(self):
        assert f1_score("Paris", "") == 0.0

    def test_both_empty(self):
        assert f1_score("", "") == 1.0

    def test_subset_relationship(self):
        # "the president" predicted, "the president of the US" gold
        # tokens: pred={"the","president"} (2), gold={"the","president","of","us"} (4)
        # common=2, precision=2/2=1.0, recall=2/4=0.5
        # F1 = 2*1*0.5/(1+0.5) = 1/3 ≈ 0.6667
        pred = "the president"
        gold = "the president of the US"
        assert f1_score(pred, gold) == pytest.approx(2/3, rel=1e-3)
