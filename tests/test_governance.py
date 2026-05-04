"""Tests for the trust thermodynamics governor."""
import math
import pytest
from snc_core import Decision, behavioral_governance, trust_thermodynamic


class TestTrustFormula:
    def test_perfect_agreement_recovers_ppv(self):
        assert trust_thermodynamic(0.8, 0.0) == pytest.approx(0.8)

    def test_zero_t_comp_recovers_ppv(self):
        assert trust_thermodynamic(0.8, 0.5, t_comp=0.0) == pytest.approx(0.8)

    def test_max_disagreement_discounts(self):
        ppv = 0.9
        expected = ppv * math.exp(-1.0)
        assert trust_thermodynamic(ppv, 1.0, t_comp=1.0) == pytest.approx(expected)

    def test_adaptive_t_comp(self):
        ppv, sigma = 0.6, 0.5
        expected = ppv * math.exp(-sigma * (0.5 + (1 - 0.6)))
        assert trust_thermodynamic(ppv, sigma) == pytest.approx(expected)

    def test_invalid_ppv_raises(self):
        with pytest.raises(ValueError):
            trust_thermodynamic(1.5, 0.0)
        with pytest.raises(ValueError):
            trust_thermodynamic(-0.1, 0.0)


class TestBehavioralGovernance:
    def test_empty_population_abstains(self):
        d = behavioral_governance([])
        assert d.action == Decision.ABSTAIN

    def test_unanimous_high_conf_admits(self):
        cands = [{"answer": "42", "self_confidence": 0.9, "behavior_key": "42"} for _ in range(5)]
        d = behavioral_governance(cands, threshold=0.5)
        assert d.action == Decision.ADMIT
        assert d.modal_answer == "42"
        assert d.n_clusters == 1
        assert d.sigma_calib == 0.0
        assert d.trust == pytest.approx(0.9)

    def test_uniform_disagreement_abstains(self):
        cands = [{"answer": str(i), "self_confidence": 0.5, "behavior_key": str(i)} for i in range(5)]
        d = behavioral_governance(cands, threshold=0.5)
        assert d.action == Decision.ABSTAIN
        assert d.n_clusters == 5
        assert d.sigma_calib == pytest.approx(1.0)

    def test_threshold_admit_boundary(self):
        cands = [{"answer": "x", "self_confidence": 0.6, "behavior_key": "x"} for _ in range(3)]
        assert behavioral_governance(cands, threshold=0.5).action == Decision.ADMIT
        assert behavioral_governance(cands, threshold=0.7).action == Decision.ABSTAIN

    def test_modal_cluster_outvotes(self):
        # 4 of 5 cluster as "right"; with sigma > 0 trust may fall below the
        # default threshold, so we explicitly use a low threshold here to
        # verify modal selection logic independent of the admit gate.
        cands = ([{"answer": "right", "self_confidence": 0.8, "behavior_key": "R"}] * 4
                 + [{"answer": "wrong", "self_confidence": 0.6, "behavior_key": "W"}])
        d = behavioral_governance(cands, threshold=0.3)
        assert d.action == Decision.ADMIT
        assert d.modal_answer == "right"
        assert d.n_clusters == 2


def test_single_candidate_unanimous():
    cands = [{"answer": "x", "self_confidence": 0.7, "behavior_key": "x"}]
    d = behavioral_governance(cands, threshold=0.5)
    assert d.n_clusters == 1
    assert d.action == Decision.ADMIT


def test_zero_confidence_abstains():
    cands = [{"answer": "x", "self_confidence": 0.0, "behavior_key": "x"} for _ in range(5)]
    d = behavioral_governance(cands, threshold=0.1)
    assert d.action == Decision.ABSTAIN
    assert d.trust == 0.0
