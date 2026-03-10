"""
tests/test_chains.py
Unit tests for chains.py — no live Ollama or Qdrant required.

Coverage:
  • _clean_bullet          — markdown stripping, prefix, Qwen <think> removal
  • _strip_json_fences
  • _format_facts
  • _format_exemplars
  • _pick_supporting_facts
  • SemanticMatcher.infer_tone
  • Character tolerance boundary checks
  • Metric fidelity (numbers never mutated by helpers)
"""

import pytest
from app.models import (
    CoreFact,
    FormattingConstraints,
    JDTone,
    RoleType,
    ScoredFact,
    StyleExemplar,
    BulletDraft,
)
from app.chains import (
    _clean_bullet,
    _strip_json_fences,
    _format_facts,
    _format_exemplars,
    _pick_supporting_facts,
    SemanticMatcher,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_scored_fact(
    fact_id: str = "f-001",
    text: str = "Built SARIMA model reducing RMSE to 0.250",
    tools: list[str] | None = None,
    metrics: list[str] | None = None,
    project_id: str = "p-001",
    relevance: float = 0.9,
    keywords: list[str] | None = None,
) -> ScoredFact:
    return ScoredFact(
        fact=CoreFact(
            fact_id=fact_id,
            text=text,
            tools=tools or ["SARIMA"],
            metrics=metrics or ["RMSE 0.250"],
            outcome="Best forecast accuracy",
        ),
        project_id=project_id,
        relevance_score=relevance,
        matched_jd_keywords=keywords or ["SARIMA", "forecasting"],
    )


def make_exemplar(eid: str = "e-001", structure: str | None = None) -> StyleExemplar:
    return StyleExemplar(
        exemplar_id=eid,
        text="• Enhanced accuracy via ARIMAX | Reduced RMSE by 13.5%",
        role_type=RoleType.DATA_SCIENCE,
        similarity_score=0.85,
        uses_separator="|",
        sentence_structure=structure,
    )


# ── _clean_bullet ─────────────────────────────────────────────────────────────

class TestCleanBullet:
    def test_strips_markdown_bold(self):
        assert "**" not in _clean_bullet("**• Built a model**", "•")

    def test_strips_leading_dash(self):
        assert _clean_bullet("- Built a model", "•").startswith("•")

    def test_strips_leading_asterisk(self):
        assert _clean_bullet("* Built a model", "•").startswith("•")

    def test_adds_missing_prefix(self):
        assert _clean_bullet("Built a model", "•").startswith("•")

    def test_does_not_double_prefix(self):
        assert _clean_bullet("• Already prefixed", "•").count("•") == 1

    def test_strips_qwen_think_block(self):
        text = "<think>Reasoning here.</think>• Built a model"
        result = _clean_bullet(text, "•")
        assert "<think>" not in result
        assert "Reasoning" not in result
        assert result.startswith("•")

    def test_strips_multiline_think_block(self):
        text = "<think>\nLine 1\nLine 2\n</think>• Built SARIMA model"
        result = _clean_bullet(text, "•")
        assert "Line 1" not in result

    def test_strips_surrounding_whitespace(self):
        assert _clean_bullet("  • A bullet  ", "•") == _clean_bullet("  • A bullet  ", "•").strip()

    def test_custom_prefix(self):
        assert _clean_bullet("Some fact", "-").startswith("-")

    def test_empty_prefix_passes_through(self):
        result = _clean_bullet("Some fact text here", "")
        assert "Some fact" in result


# ── _strip_json_fences ────────────────────────────────────────────────────────

class TestStripJsonFences:
    def test_strips_json_fence(self):
        assert _strip_json_fences('```json\n{"k": "v"}\n```') == '{"k": "v"}'

    def test_strips_plain_fence(self):
        assert _strip_json_fences("```\n[1,2]\n```") == "[1,2]"

    def test_no_fence_unchanged(self):
        raw = '{"k": "v"}'
        assert _strip_json_fences(raw) == raw

    def test_strips_surrounding_whitespace(self):
        assert _strip_json_fences("  ```json\n{}\n```  ") == "{}"


# ── _format_facts ─────────────────────────────────────────────────────────────

class TestFormatFacts:
    def test_includes_fact_id_and_text(self):
        result = _format_facts([make_scored_fact("f-001", "Built SARIMA model")])
        assert "f-001" in result
        assert "Built SARIMA model" in result

    def test_includes_metrics(self):
        result = _format_facts([make_scored_fact(metrics=["RMSE 0.250", "87%"])])
        assert "RMSE 0.250" in result
        assert "87%" in result

    def test_includes_tools(self):
        result = _format_facts([make_scored_fact(tools=["LangChain", "GPT-4"])])
        assert "LangChain" in result and "GPT-4" in result

    def test_multiple_facts(self):
        facts = [
            make_scored_fact("f-001", "First fact"),
            make_scored_fact("f-002", "Second fact"),
        ]
        result = _format_facts(facts)
        assert "f-001" in result and "f-002" in result

    def test_empty_returns_fallback(self):
        assert "no facts" in _format_facts([]).lower()


# ── _format_exemplars ─────────────────────────────────────────────────────────

class TestFormatExemplars:
    def test_includes_exemplar_text(self):
        assert "Enhanced accuracy" in _format_exemplars([make_exemplar()])

    def test_includes_structure_when_present(self):
        result = _format_exemplars([make_exemplar(structure="verb → tool → metric")])
        assert "verb → tool → metric" in result

    def test_no_structure_field_omitted(self):
        assert "[structure:" not in _format_exemplars([make_exemplar(structure=None)])

    def test_empty_returns_fallback(self):
        assert "none" in _format_exemplars([]).lower()

    def test_numbered_list(self):
        result = _format_exemplars([make_exemplar("e-001"), make_exemplar("e-002")])
        assert "1." in result and "2." in result


# ── _pick_supporting_facts ────────────────────────────────────────────────────

class TestPickSupportingFacts:
    def test_picks_overlapping_keyword_facts(self):
        primary = make_scored_fact("f-001", keywords=["SARIMA", "forecasting"])
        facts = [
            primary,
            make_scored_fact("f-002", keywords=["forecasting", "RMSE"]),
            make_scored_fact("f-003", keywords=["React", "frontend"]),
        ]
        ids = [s.fact.fact_id for s in _pick_supporting_facts(primary, facts, exclude_idx=0)]
        assert "f-002" in ids
        assert "f-003" not in ids

    def test_excludes_primary(self):
        primary = make_scored_fact("f-001", keywords=["ML"])
        facts = [primary, make_scored_fact("f-002", keywords=["ML"])]
        ids = [s.fact.fact_id for s in _pick_supporting_facts(primary, facts, exclude_idx=0)]
        assert "f-001" not in ids

    def test_respects_max_supporting(self):
        primary = make_scored_fact("f-001", keywords=["ML"])
        others  = [make_scored_fact(f"f-{i:03d}", keywords=["ML"]) for i in range(2, 10)]
        result  = _pick_supporting_facts(primary, [primary] + others, exclude_idx=0, max_supporting=2)
        assert len(result) <= 2

    def test_no_overlap_returns_empty(self):
        primary = make_scored_fact("f-001", keywords=["SARIMA"])
        facts   = [primary, make_scored_fact("f-002", keywords=["Docker"])]
        assert _pick_supporting_facts(primary, facts, exclude_idx=0) == []


# ── SemanticMatcher.infer_tone ────────────────────────────────────────────────

class TestInferTone:
    def setup_method(self):
        self.matcher = SemanticMatcher()

    def test_known_tones(self):
        assert self.matcher.infer_tone({"tone": "highly_quantitative"}) == JDTone.HIGHLY_QUANTITATIVE
        assert self.matcher.infer_tone({"tone": "engineering_focused"}) == JDTone.ENGINEERING_FOCUSED

    def test_missing_defaults_to_balanced(self):
        assert self.matcher.infer_tone({}) == JDTone.BALANCED

    def test_unknown_defaults_to_balanced(self):
        assert self.matcher.infer_tone({"tone": "gibberish"}) == JDTone.BALANCED


# ── Char tolerance logic ──────────────────────────────────────────────────────

class TestCharTolerance:
    @pytest.mark.parametrize("target,tol,text,expected", [
        (130, 2, "x" * 128, True),
        (130, 2, "x" * 130, True),
        (130, 2, "x" * 132, True),
        (130, 2, "x" * 127, False),
        (130, 2, "x" * 133, False),
    ])
    def test_within_tolerance(self, target, tol, text, expected):
        c = FormattingConstraints(target_char_limit=target, tolerance=tol)
        assert (c.lower_bound <= len(text) <= c.upper_bound) == expected

    def test_closest_draft_selection(self):
        target = 130
        drafts = [
            BulletDraft(text="x"*115, char_count=115, iteration=1, within_tolerance=False, source_fact_ids=[]),
            BulletDraft(text="x"*125, char_count=125, iteration=2, within_tolerance=False, source_fact_ids=[]),
            BulletDraft(text="x"*135, char_count=135, iteration=3, within_tolerance=False, source_fact_ids=[]),
        ]
        best = min(drafts, key=lambda d: abs(d.char_count - target))
        assert best.char_count == 125


# ── Metric fidelity ───────────────────────────────────────────────────────────

class TestMetricFidelity:
    @pytest.mark.parametrize("metric", [
        "0.250", "87%", "₹20-27L", "8-12 weeks", "SARIMA(2,0,0)(1,0,0)[12]",
    ])
    def test_metric_survives_format_facts(self, metric):
        fact = ScoredFact(
            fact=CoreFact(fact_id="f-t", text=f"Result {metric}", metrics=[metric]),
            project_id="p-001",
            relevance_score=0.9,
        )
        assert metric in _format_facts([fact])
