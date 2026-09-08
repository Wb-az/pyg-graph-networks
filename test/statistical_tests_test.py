import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.common import statistical_tests as st


def test_holm_matches_known_values():
    # Classic example: p = [0.01, 0.04, 0.03] -> Holm = [0.03, 0.06, 0.06]
    out = st.holm_correction([0.01, 0.04, 0.03])
    assert np.allclose(out, [0.03, 0.06, 0.06])
    assert (out <= 1).all()


def test_mcnemar_exact_binomial():
    # 10 discordant one way, 2 the other: exact two-sided binomial p
    a = np.array([1] * 10 + [0] * 2 + [1] * 50 + [0] * 5, dtype=bool)
    b = np.array([0] * 10 + [1] * 2 + [1] * 50 + [0] * 5, dtype=bool)
    res = st.mcnemar_test(a, b, exact=True)
    assert (res["a_only"], res["b_only"]) == (10, 2)
    assert res["p_value"] == pytest.approx(stats.binomtest(2, 12, 0.5).pvalue)


def test_mcnemar_chi2_with_continuity():
    a = np.array([1] * 30 + [0] * 10 + [1] * 20, dtype=bool)
    b = np.array([0] * 30 + [1] * 10 + [1] * 20, dtype=bool)
    res = st.mcnemar_test(a, b, exact=False)
    expected = (abs(30 - 10) - 1) ** 2 / 40
    assert res["statistic"] == pytest.approx(expected)
    assert res["p_value"] == pytest.approx(stats.chi2.sf(expected, 1))


def test_mcnemar_identical_classifiers():
    a = np.array([1, 0, 1, 1, 0], dtype=bool)
    res = st.mcnemar_test(a, a)
    assert res["p_value"] == 1.0 and res["a_only"] == res["b_only"] == 0


def test_cochran_q_reduces_to_zero_for_identical_columns():
    x = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]], dtype=bool)
    assert st.cochran_q_test(x)["statistic"] == 0.0


def test_cochran_q_textbook_example():
    # Conover (1999) style: 3 treatments, 12 blocks; Q computed by the formula.
    x = np.array([[1, 1, 1], [1, 1, 1], [0, 1, 0], [1, 1, 0], [0, 0, 0], [1, 1, 1],
                  [1, 1, 1], [1, 1, 0], [0, 0, 1], [0, 1, 0], [1, 1, 1], [1, 1, 1]])
    k = 3
    col, row = x.sum(0), x.sum(1)
    q = (k - 1) * (k * (col ** 2).sum() - col.sum() ** 2) / (k * row.sum() - (row ** 2).sum())
    res = st.cochran_q_test(x.astype(bool))
    assert res["statistic"] == pytest.approx(q)
    assert res["p_value"] == pytest.approx(stats.chi2.sf(q, 2))


def test_pairwise_tables_have_holm_column():
    rng = np.random.default_rng(0)
    correct = pd.DataFrame(rng.random((100, 3)) > 0.3, columns=list("abc"))
    table = st.pairwise_mcnemar(correct)
    assert len(table) == 3 and "p_holm" in table and (table.p_holm >= table.p_value).all()
    scores = pd.DataFrame(rng.random((5, 3)), columns=list("abc"))
    w = st.pairwise_wilcoxon(scores)
    assert len(w) == 3 and (w.p_holm >= w.p_value).all()
    assert st.friedman_test(scores)["df"] == 2 and st.kruskal_test(scores)["df"] == 2
