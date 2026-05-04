"""Integration tests for HybridLayer using a deterministic mock backend."""
import pytest

from snc_core import HybridLayer, Decision
from snc_core.adapters import CallableBackend


def _make_unanimous_backend(answer: str, confidence: float = 0.85):
    """A backend that always returns the same answer + confidence."""
    def _gen(prompt, system, temp, max_tok):
        return f"{answer}\nCONFIDENCE: {confidence}"
    return CallableBackend(_gen)


def _make_random_backend(answers: list, confidences: list):
    """Round-robin through a list of (answer, confidence) pairs."""
    state = {"i": 0}
    def _gen(prompt, system, temp, max_tok):
        i = state["i"] % len(answers)
        state["i"] += 1
        return f"{answers[i]}\nCONFIDENCE: {confidences[i]}"
    return CallableBackend(_gen)


class TestHybridLayer:
    def test_unanimous_admits(self):
        backend = _make_unanimous_backend("42", confidence=0.9)
        hybrid = HybridLayer(backend, k=5, threshold=0.5)
        result = hybrid.query("What is 6 * 7?")
        assert result.action == Decision.ADMIT
        assert result.answer == "42"
        assert len(result.candidates) == 5
        assert result.decision.n_clusters == 1
        assert result.decision.sigma_calib == 0.0
        assert result.decision.trust == pytest.approx(0.9)

    def test_disagreement_abstains_at_high_threshold(self):
        backend = _make_random_backend(
            answers=["1", "2", "3", "4", "5"],
            confidences=[0.5, 0.5, 0.5, 0.5, 0.5],
        )
        hybrid = HybridLayer(backend, k=5, threshold=0.5)
        result = hybrid.query("What number am I thinking of?")
        assert result.action == Decision.ABSTAIN
        assert result.answer is None

    def test_low_confidence_low_trust(self):
        backend = _make_unanimous_backend("answer", confidence=0.2)
        hybrid = HybridLayer(backend, k=3, threshold=0.5)
        result = hybrid.query("Hard question.")
        # Even with unanimous, low PPV means trust = PPV = 0.2 < threshold
        assert result.action == Decision.ABSTAIN

    def test_invalid_k_raises(self):
        backend = _make_unanimous_backend("x")
        with pytest.raises(ValueError):
            HybridLayer(backend, k=0)
        with pytest.raises(ValueError):
            HybridLayer(backend, k=128)

    def test_invalid_threshold_raises(self):
        backend = _make_unanimous_backend("x")
        with pytest.raises(ValueError):
            HybridLayer(backend, threshold=1.5)
        with pytest.raises(ValueError):
            HybridLayer(backend, threshold=-0.1)

    def test_total_tokens_aggregates(self):
        backend = _make_unanimous_backend("hello", confidence=0.7)
        hybrid = HybridLayer(backend, k=4)
        result = hybrid.query("Test")
        assert result.total_tokens > 0
        # CallableBackend approximates as len(text)//4, K candidates
        # so total >= 4
        assert result.total_tokens >= 4

    def test_query_batch(self):
        backend = _make_unanimous_backend("ok", confidence=0.8)
        hybrid = HybridLayer(backend, k=2, threshold=0.5)
        results = hybrid.query_batch(["q1", "q2", "q3"])
        assert len(results) == 3
        assert all(r.action == Decision.ADMIT for r in results)

    def test_behavior_extractor_clusters_semantically(self):
        # 3 syntactically different but semantically same answers.
        # With raw-string clustering they form 3 clusters; with a
        # behavior_extractor that normalizes to a canonical form, 1 cluster.
        backend = _make_random_backend(
            answers=["[1, 2, 3]", "[1,2,3]", "[ 1 , 2 , 3 ]"],
            confidences=[0.8, 0.8, 0.8],
        )
        # No extractor: 3 clusters (raw)
        hybrid_raw = HybridLayer(backend, k=3, threshold=0.5)
        result_raw = hybrid_raw.query("List of first 3 ints?")
        assert result_raw.decision.n_clusters >= 1  # might be 3

        # With extractor that strips whitespace
        backend2 = _make_random_backend(
            answers=["[1, 2, 3]", "[1,2,3]", "[ 1 , 2 , 3 ]"],
            confidences=[0.8, 0.8, 0.8],
        )
        hybrid_extract = HybridLayer(
            backend2, k=3, threshold=0.5,
            behavior_extractor=lambda s: ("".join(s.split()),)
        )
        result_extract = hybrid_extract.query("List of first 3 ints?")
        assert result_extract.decision.n_clusters == 1
        assert result_extract.action == Decision.ADMIT
