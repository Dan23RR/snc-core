"""Tests for parsing helpers."""
import pytest

from snc_core import parse_confidence, parse_answer
from snc_core.parsing import extract_code


class TestParseConfidence:
    def test_separate_line(self):
        text = "The answer is 42.\nCONFIDENCE: 0.9"
        assert parse_confidence(text) == pytest.approx(0.9)

    def test_inline(self):
        text = "NON LO SO. CONFIDENCE: 0.0"
        assert parse_confidence(text) == pytest.approx(0.0)

    def test_case_insensitive(self):
        assert parse_confidence("answer\nconfidence: 0.7") == pytest.approx(0.7)
        assert parse_confidence("answer\nConfidence = 0.7") == pytest.approx(0.7)

    def test_clamps_above_one(self):
        assert parse_confidence("CONFIDENCE: 1.5") == pytest.approx(1.0)

    def test_clamps_below_zero(self):
        assert parse_confidence("CONFIDENCE: -0.3") == pytest.approx(0.0)

    def test_default_when_absent(self):
        assert parse_confidence("no marker here") == pytest.approx(0.5)

    def test_default_when_unparseable(self):
        assert parse_confidence("CONFIDENCE: high") == pytest.approx(0.5)


class TestParseAnswer:
    def test_strips_confidence_inline(self):
        text = "NON LO SO. CONFIDENCE: 0.0"
        assert "CONFIDENCE" not in parse_answer(text)

    def test_strips_confidence_newline(self):
        text = "42\nCONFIDENCE: 0.9"
        assert parse_answer(text).strip() == "42"

    def test_empty_input(self):
        assert parse_answer("") == ""
        assert parse_answer(None) == ""


class TestExtractCode:
    def test_python_fence(self):
        text = "Here is the code:\n```python\ndef f(): return 1\n```"
        assert "def f(): return 1" in extract_code(text)

    def test_unfenced_block(self):
        text = "```\ndef f(): return 1\n```"
        assert "def f(): return 1" in extract_code(text)

    def test_def_keyword_fallback(self):
        text = "Some preamble.\ndef foo():\n    return 'x'\n"
        result = extract_code(text)
        assert result.startswith("def foo():")

    def test_raw_text_fallback(self):
        text = "just some text"
        assert extract_code(text) == "just some text"
