"""
tests/test_chains.py
Unit tests for chains.py — no live Ollama or Qdrant required.

Coverage:
  • _clean_bullet          — markdown stripping, prefix, Qwen <think> removal
  • _strip_json_fences     — fence removal + <think> block stripping (M8)
  • _format_facts
  • _format_exemplars
  • _pick_supporting_facts
  • SemanticMatcher.infer_tone
  • SemanticMatcher.score_facts — json_mode retry on parse fail (H5)
  • Character tolerance boundary checks
  • Metric fidelity (numbers never mutated by helpers)
  • Groq 429 retry logic (H2)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
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
    _extract_first_json_value,
    _safe_parse_json,
    _format_facts,
    _format_exemplars,
    _pick_supporting_facts,
    _groq_retry_wait,
    _GROQ_MAX_RETRIES,
    _GROQ_RETRY_AFTER_DEFAULT,
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

    def test_preserves_internal_asterisks(self):
        """Internal *args / **kwargs must survive markdown stripping (M9)."""
        result = _clean_bullet("• df.apply(*args, **kwargs)", "•")
        assert "*args" in result
        assert "**kwargs" in result

    def test_strips_italic_wrapper(self):
        result = _clean_bullet("*• Built a model*", "•")
        assert result.count("*") == 0


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

    def test_strips_think_block_before_json(self):
        raw = "<think>I will output valid JSON now.</think>\n{\"k\": \"v\"}"
        assert _strip_json_fences(raw) == '{"k": "v"}'

    def test_strips_think_block_wrapped_in_fence(self):
        raw = "```json\n<think>reasoning</think>\n{\"a\": 1}\n```"
        result = _strip_json_fences(raw)
        assert "<think>" not in result
        assert '{"a": 1}' in result

    def test_no_think_block_unchanged(self):
        raw = '[{"fact_id": "f-1", "relevance_score": 0.9}]'
        assert _strip_json_fences(raw) == raw


# ── _extract_first_json_value + _safe_parse_json fallback ────────────────────

class TestExtractFirstJsonValue:
    """Robustness for Groq responses with trailing/leading prose around JSON."""

    def test_returns_none_when_no_json(self):
        assert _extract_first_json_value("just plain text, no brackets") is None

    def test_extracts_bare_array(self):
        assert _extract_first_json_value('[1, 2, 3]') == "[1, 2, 3]"

    def test_extracts_bare_object(self):
        assert _extract_first_json_value('{"k": "v"}') == '{"k": "v"}'

    def test_strips_trailing_prose(self):
        raw = '[{"id": "i0", "score": 0.9}]\n\nHope this helps!'
        assert _extract_first_json_value(raw) == '[{"id": "i0", "score": 0.9}]'

    def test_strips_leading_prose(self):
        raw = 'Here are the scores: [{"id": "i0"}]'
        assert _extract_first_json_value(raw) == '[{"id": "i0"}]'

    def test_respects_brackets_inside_strings(self):
        # The `]` inside the string value must not close the outer array.
        raw = '[{"text": "scored [pre-cleaning] data"}]'
        assert _extract_first_json_value(raw) == raw

    def test_respects_escaped_quotes_inside_strings(self):
        raw = '[{"text": "she said \\"hi\\""}]'
        assert _extract_first_json_value(raw) == raw

    def test_handles_nested_structures(self):
        raw = '[{"a": [1, [2, 3]], "b": {"c": [4]}}]'
        assert _extract_first_json_value(raw) == raw

    def test_unbalanced_returns_none(self):
        # Truncated mid-array — no matching close, so extraction fails.
        assert _extract_first_json_value('[{"id": "i0"') is None


class TestSafeParseJsonFallback:
    """_safe_parse_json delegates to the extractor on JSONDecodeError."""

    def test_parses_clean_array(self):
        assert _safe_parse_json('[1, 2]') == [1, 2]

    def test_parses_array_with_trailing_prose(self):
        # Bug-reproducer: this is the exact failure mode from the user's log.
        # Before the extractor fallback, json.loads raised "Extra data" and
        # the function returned None, triggering the json_mode envelope retry.
        result = _safe_parse_json(
            '[{"id": "i0", "relevance_score": 0.7}]\n\nLet me know if you need more!'
        )
        assert result == [{"id": "i0", "relevance_score": 0.7}]

    def test_returns_none_on_truly_malformed(self):
        assert _safe_parse_json("definitely not json at all") is None


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

    def test_fallback_when_primary_has_no_keywords(self):
        """Empty primary keywords → return next highest-scored facts instead of nothing."""
        # make_scored_fact uses `keywords or [...]` so construct directly to get truly empty list
        primary = ScoredFact(
            fact=CoreFact(fact_id="f-001", text="placeholder", tools=[], metrics=[], outcome=""),
            project_id="p-001", relevance_score=0.9, matched_jd_keywords=[],
        )
        f2 = make_scored_fact("f-002", keywords=["SQL"])
        f3 = make_scored_fact("f-003", keywords=["Python"])
        facts  = [primary, f2, f3]
        result = _pick_supporting_facts(primary, facts, exclude_idx=0)
        ids    = [s.fact.fact_id for s in result]
        assert "f-002" in ids
        assert "f-001" not in ids

    def test_fallback_respects_max_supporting(self):
        """Fallback still caps at max_supporting."""
        primary = ScoredFact(
            fact=CoreFact(fact_id="f-001", text="placeholder", tools=[], metrics=[], outcome=""),
            project_id="p-001", relevance_score=0.9, matched_jd_keywords=[],
        )
        others = [make_scored_fact(f"f-{i:03d}", keywords=["X"]) for i in range(2, 6)]
        result = _pick_supporting_facts(primary, [primary] + others, exclude_idx=0, max_supporting=2)
        assert len(result) == 2


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


# ── H5: score_facts json_mode retry + M8: _strip_json_fences <think> strip ───

class TestScoreFactsRetry:
    """SemanticMatcher.score_facts retries with json_mode=True on first parse failure."""

    def _make_project(self) -> "ProjectData":
        from app.models import ProjectData
        return ProjectData(
            project_id="p-001",
            title="Forecasting",
            core_facts=[
                CoreFact(fact_id="f-001", text="Built SARIMA model", tools=["SARIMA"], metrics=[]),
            ],
        )

    @pytest.mark.anyio
    async def test_succeeds_on_first_try_no_retry(self):
        """When the LLM returns valid JSON immediately, _ollama_chat is called once."""
        good_json = '[{"id": "i0", "relevance_score": 0.9, "matched_jd_keywords": ["SARIMA"]}]'
        with patch("app.chains._ollama_chat", new=AsyncMock(return_value=good_json)) as mock_chat:
            matcher = SemanticMatcher()
            result = await matcher.score_facts({}, [self._make_project()])

        assert mock_chat.call_count == 1
        assert result[0].relevance_score == 0.9
        assert result[0].matched_jd_keywords == ["SARIMA"]

    @pytest.mark.anyio
    async def test_retries_with_json_mode_on_first_parse_failure(self):
        """On first parse failure, score_facts retries once with json_mode=True."""
        bad_raw  = "Here are the scores: blah blah (malformed)"
        good_json = '[{"id": "i0", "relevance_score": 0.7, "matched_jd_keywords": []}]'
        with patch("app.chains._ollama_chat", new=AsyncMock(side_effect=[bad_raw, good_json])) as mock_chat:
            matcher = SemanticMatcher()
            result = await matcher.score_facts({}, [self._make_project()])

        assert mock_chat.call_count == 2
        # Second call must have json_mode=True
        _, second_kwargs = mock_chat.call_args_list[1]
        assert second_kwargs.get("json_mode") is True
        assert result[0].relevance_score == 0.7

    @pytest.mark.anyio
    async def test_falls_back_to_0_5_after_double_failure(self):
        """If both attempts return malformed JSON, every fact gets relevance_score=0.5."""
        bad = "not json at all"
        with patch("app.chains._ollama_chat", new=AsyncMock(return_value=bad)):
            matcher = SemanticMatcher()
            result = await matcher.score_facts({}, [self._make_project()])

        assert all(sf.relevance_score == 0.5 for sf in result)
        assert all(sf.matched_jd_keywords == [] for sf in result)

    @pytest.mark.anyio
    async def test_first_call_does_not_use_json_mode(self):
        """The initial call is made without json_mode so we don't waste quota on overhead."""
        good_json = '[{"id": "i0", "relevance_score": 0.8, "matched_jd_keywords": []}]'
        with patch("app.chains._ollama_chat", new=AsyncMock(return_value=good_json)) as mock_chat:
            matcher = SemanticMatcher()
            await matcher.score_facts({}, [self._make_project()])

        first_kwargs = mock_chat.call_args_list[0][1]
        assert not first_kwargs.get("json_mode", False)

    @pytest.mark.anyio
    async def test_no_collision_when_two_projects_share_fact_ids(self):
        """N1 regression: two projects with identical fact_ids must get distinct scores.

        Prior to the index-based score_map, score_map[fact_id] was the LAST entry
        in the LLM response sharing that fact_id, so the second project clobbered
        the first. Now scoring uses positional ids (i0, i1, ...).
        """
        from app.models import ProjectData
        proj_a = ProjectData(
            project_id="p-A",
            title="Forecasting",
            core_facts=[CoreFact(fact_id="f-1", text="Built SARIMA model")],
        )
        proj_b = ProjectData(
            project_id="p-B",
            title="Pricing",
            core_facts=[CoreFact(fact_id="f-1", text="Optimised prices via SLSQP")],
        )
        # LLM scores i0 high (the first fact), i1 low (the second fact).
        good_json = (
            '[{"id": "i0", "relevance_score": 0.95, "matched_jd_keywords": ["SARIMA"]},'
            ' {"id": "i1", "relevance_score": 0.10, "matched_jd_keywords": []}]'
        )
        with patch("app.chains._ollama_chat", new=AsyncMock(return_value=good_json)):
            result = await SemanticMatcher().score_facts({}, [proj_a, proj_b])

        by_pid = {sf.project_id: sf for sf in result}
        assert by_pid["p-A"].relevance_score == 0.95
        assert by_pid["p-B"].relevance_score == 0.10
        assert by_pid["p-A"].matched_jd_keywords == ["SARIMA"]
        assert by_pid["p-B"].matched_jd_keywords == []

    @pytest.mark.anyio
    async def test_scoring_max_tokens_scales_with_fact_count(self):
        """The scoring call must request enough output tokens to fit all facts.

        Settings default `llm_max_tokens=512` covers ~20 fact-entries; CVs with
        50+ facts (5+ projects × 4–6 bullets each) overflowed the cap, Groq
        truncated mid-array, and every fact past the cut-off defaulted to 0.5.
        That defaulted-to-0.5 set out-sorted real low LLM scores in the top-3
        mean rollup, producing the "every top project capped at 50%" symptom.
        """
        from app.models import ProjectData
        # 30 facts across 3 projects (10 each) — well past the 512-token cap.
        projects = [
            ProjectData(
                project_id=f"p-{i}",
                title=f"P{i}",
                core_facts=[
                    CoreFact(fact_id=f"f-{i}-{j}", text=f"fact {i}-{j} text")
                    for j in range(10)
                ],
            )
            for i in range(3)
        ]
        good_json = "[" + ",".join(
            f'{{"id":"i{i}","relevance_score":0.7,"matched_jd_keywords":[]}}'
            for i in range(30)
        ) + "]"
        with patch(
            "app.chains._ollama_chat", new=AsyncMock(return_value=good_json),
        ) as mock_chat:
            await SemanticMatcher().score_facts({}, projects)

        # Inspect the kwargs passed to _ollama_chat — must include a max_tokens
        # that covers all 30 facts (30 * 40 = 1200 minimum).
        _, kwargs = mock_chat.call_args_list[0]
        max_tokens = kwargs.get("max_tokens")
        assert max_tokens is not None, "score_facts must pass max_tokens explicitly"
        assert max_tokens >= 30 * 40, (
            f"max_tokens={max_tokens} too low for 30 facts — Groq would truncate "
            "the response and partial facts would default to 0.5."
        )

    @pytest.mark.anyio
    async def test_parses_array_with_trailing_prose(self):
        """Groq's Llama 3.3 often appends explanatory text after a closed JSON
        array (e.g. "...]\\n\\nHope this helps!"). json.loads raises 'Extra data'
        on that, so _safe_parse_json fell through to the json_mode retry — which
        on Groq forces a JSON object response, breaking the array prompt and
        sending every score to the 0.5 fallback. After the fix _safe_parse_json
        extracts the first balanced JSON value, so this parses on the first try
        and no retry happens.
        """
        from app.models import ProjectData
        proj = ProjectData(
            project_id="p-A",
            title="Forecasting",
            core_facts=[CoreFact(fact_id="f-1", text="Built SARIMA model")],
        )
        raw_with_prose = (
            '[{"id": "i0", "relevance_score": 0.82, "matched_jd_keywords": ["SARIMA"]}]'
            "\n\nHope this helps! Let me know if you need anything else."
        )
        with patch(
            "app.chains._ollama_chat", new=AsyncMock(return_value=raw_with_prose),
        ) as mock_chat:
            result = await SemanticMatcher().score_facts({}, [proj])

        assert mock_chat.call_count == 1, "Trailing prose should not trigger a retry"
        assert result[0].relevance_score == 0.82
        assert result[0].matched_jd_keywords == ["SARIMA"]

    @pytest.mark.anyio
    async def test_retry_unwraps_scores_envelope(self):
        """Q-followup: Groq's json_mode forces a JSON object, so the array-shaped
        primary prompt can't succeed under retry. The envelope retry asks for
        {"scores": [...]} and unwraps it. Before the fix, the wrapped response
        failed the isinstance(list) check and every fact got 0.5.
        """
        from app.models import ProjectData
        proj = ProjectData(
            project_id="p-A",
            title="Forecasting",
            core_facts=[CoreFact(fact_id="f-1", text="Built SARIMA model")],
        )
        # First call: garbage that won't parse, forcing the envelope retry.
        bad_first = "totally not json"
        # Retry: Groq's json_mode response wrapping the scores array.
        wrapped = (
            '{"scores": [{"id": "i0", "relevance_score": 0.74, '
            '"matched_jd_keywords": ["forecasting"]}]}'
        )
        with patch(
            "app.chains._ollama_chat",
            new=AsyncMock(side_effect=[bad_first, wrapped]),
        ) as mock_chat:
            result = await SemanticMatcher().score_facts({}, [proj])

        assert mock_chat.call_count == 2
        # The retry must run with json_mode=True (Groq enforces object response).
        _, second_kwargs = mock_chat.call_args_list[1]
        assert second_kwargs.get("json_mode") is True
        assert result[0].relevance_score == 0.74
        assert result[0].matched_jd_keywords == ["forecasting"]

    @pytest.mark.anyio
    async def test_retry_unwraps_scored_facts_alias(self):
        """Defence-in-depth: if Llama ignores the "scores" envelope key and
        uses "scored_facts" (the exact key Groq emitted in the bug report log),
        we still unwrap it instead of falling through to the 0.5 default.
        """
        from app.models import ProjectData
        proj = ProjectData(
            project_id="p-A",
            title="Forecasting",
            core_facts=[CoreFact(fact_id="f-1", text="Built SARIMA model")],
        )
        bad_first = "not json"
        aliased = (
            '{"scored_facts": [{"id": "i0", "relevance_score": 0.61, '
            '"matched_jd_keywords": []}]}'
        )
        with patch(
            "app.chains._ollama_chat",
            new=AsyncMock(side_effect=[bad_first, aliased]),
        ):
            result = await SemanticMatcher().score_facts({}, [proj])

        assert result[0].relevance_score == 0.61

    @pytest.mark.asyncio
    async def test_accepts_fact_id_alias_in_response(self):
        """Q4: Llama 3.3 routinely emits 'fact_id' instead of the prompted 'id',
        because CoreFact.fact_id exists elsewhere in the schema and the model
        anchors on it. Before this fix every lookup fell through to the 0.5
        default and every project rendered as a uniform 50% match.
        """
        from app.models import ProjectData
        proj = ProjectData(
            project_id="p-A",
            title="Forecasting",
            core_facts=[
                CoreFact(fact_id="f-1", text="Built SARIMA model"),
                CoreFact(fact_id="f-2", text="Optimised prices via SLSQP"),
            ],
        )
        # LLM uses "fact_id" instead of "id" — valid JSON, wrong key name.
        wrong_key_json = (
            '[{"fact_id": "i0", "relevance_score": 0.95, "matched_jd_keywords": ["SARIMA"]},'
            ' {"fact_id": "i1", "relevance_score": 0.30, "matched_jd_keywords": []}]'
        )
        with patch("app.chains._ollama_chat", new=AsyncMock(return_value=wrong_key_json)):
            result = await SemanticMatcher().score_facts({}, [proj])

        scores = sorted([sf.relevance_score for sf in result], reverse=True)
        assert scores == [0.95, 0.30], (
            f"Expected scores extracted via 'fact_id' alias, got {scores} "
            f"(0.5/0.5 means the alias fix regressed)"
        )


# ── H2: Groq 429 retry logic ─────────────────────────────────────────────────

def _make_http_status_error(status_code: int, retry_after: str | None = None) -> tuple:
    """Return (mock_response, HTTPStatusError) for a given status code."""
    import httpx
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.headers = {"retry-after": retry_after} if retry_after else {}
    exc = httpx.HTTPStatusError("mocked", request=MagicMock(), response=resp)
    resp.raise_for_status = MagicMock(side_effect=exc)
    return resp, exc


class TestHostedLLMConfig:
    """_hosted_llm_config resolves the active hosted LLM provider.

    Default (LLM_PROVIDER=groq + groq_api_key set) must return the Groq tuple
    — guarantees zero behaviour change for the existing deployment.
    Switching LLM_PROVIDER=openrouter with a key set must return OpenRouter's
    tuple even if a Groq key is also present.
    """

    def test_returns_groq_tuple_by_default(self):
        from app.chains import _hosted_llm_config, _using_groq, settings
        with patch.object(settings, "llm_provider", "groq"), \
             patch.object(settings, "groq_api_key", "gsk-test"), \
             patch.object(settings, "groq_base_url", "https://api.groq.com/openai/v1"), \
             patch.object(settings, "groq_model", "llama-3.3-70b-versatile"):
            cfg = _hosted_llm_config()
            assert cfg == ("gsk-test", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile")
            assert _using_groq() is True

    def test_picks_openrouter_when_selected(self):
        from app.chains import _hosted_llm_config, settings
        # Both keys present — selector decides. OpenRouter must win.
        with patch.object(settings, "llm_provider", "openrouter"), \
             patch.object(settings, "groq_api_key", "gsk-should-be-ignored"), \
             patch.object(settings, "openrouter_api_key", "sk-or-v1-test"), \
             patch.object(settings, "openrouter_base_url", "https://openrouter.ai/api/v1"), \
             patch.object(settings, "openrouter_model", "meta-llama/llama-3.3-70b-instruct:free"):
            cfg = _hosted_llm_config()
            assert cfg == (
                "sk-or-v1-test",
                "https://openrouter.ai/api/v1",
                "meta-llama/llama-3.3-70b-instruct:free",
            )

    def test_returns_none_when_selected_provider_has_no_key(self):
        """LLM_PROVIDER=openrouter but empty OPENROUTER_API_KEY → fall back to
        Ollama (cfg is None). Must NOT silently use Groq even if Groq has a key.
        """
        from app.chains import _hosted_llm_config, _using_groq, settings
        with patch.object(settings, "llm_provider", "openrouter"), \
             patch.object(settings, "groq_api_key", "gsk-test-but-not-selected"), \
             patch.object(settings, "openrouter_api_key", ""):
            assert _hosted_llm_config() is None
            assert _using_groq() is False

    @pytest.mark.anyio
    async def test_groq_chat_uses_openrouter_base_url_when_selected(self):
        """_groq_chat must POST to OpenRouter's URL with OpenRouter's bearer
        token when LLM_PROVIDER=openrouter. The function name is historical;
        the routing target follows _hosted_llm_config.
        """
        from app.chains import _groq_chat, settings
        import httpx
        good_resp = MagicMock(spec=httpx.Response)
        good_resp.status_code = 200
        good_resp.raise_for_status = MagicMock()
        good_resp.json = MagicMock(return_value={
            "choices": [{"message": {"content": "ok"}}]
        })
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=good_resp)

        with patch("app.chains.get_http", return_value=mock_client), \
             patch.object(settings, "llm_provider", "openrouter"), \
             patch.object(settings, "openrouter_api_key", "sk-or-v1-abc"), \
             patch.object(settings, "openrouter_base_url", "https://openrouter.ai/api/v1"), \
             patch.object(settings, "openrouter_model", "or-model"):
            await _groq_chat([{"role": "user", "content": "test"}])

        # Verify the URL and bearer token came from OpenRouter, and the model
        # in the payload is the OpenRouter model — not the Groq one.
        assert mock_client.post.call_count == 1
        call_args = mock_client.post.call_args
        assert call_args.args[0] == "https://openrouter.ai/api/v1/chat/completions"
        assert call_args.kwargs["headers"]["Authorization"] == "Bearer sk-or-v1-abc"
        assert call_args.kwargs["json"]["model"] == "or-model"


class TestGroqRetryWait:
    def test_reads_retry_after_header(self):
        import httpx
        resp = MagicMock(spec=httpx.Response)
        resp.headers = {"retry-after": "30"}
        assert _groq_retry_wait(resp) == 30.0

    def test_uses_default_when_header_absent(self):
        import httpx
        resp = MagicMock(spec=httpx.Response)
        resp.headers = {}
        assert _groq_retry_wait(resp) == _GROQ_RETRY_AFTER_DEFAULT

    def test_uses_default_on_non_numeric_header(self):
        import httpx
        resp = MagicMock(spec=httpx.Response)
        resp.headers = {"retry-after": "soon"}
        assert _groq_retry_wait(resp) == _GROQ_RETRY_AFTER_DEFAULT


class TestGroqChatRetry:
    """_groq_chat retries on 429 and raises after exhausting retries.

    These tests exercise retry/circuit-breaker logic, not provider selection,
    so we pin LLM_PROVIDER=groq + groq_api_key="test-key" for the whole class.
    Without this, `_hosted_llm_config()` returns None (the autouse settings
    isolator clears env-derived values) and _groq_chat fails before reaching
    the retry path under test.
    """

    @pytest.fixture(autouse=True)
    def _pin_groq_provider(self):
        from app.chains import settings
        with patch.object(settings, "llm_provider", "groq"), \
             patch.object(settings, "groq_api_key", "test-key"), \
             patch.object(settings, "groq_base_url", "https://api.groq.com/openai/v1"), \
             patch.object(settings, "groq_model", "test-model"):
            yield

    def _good_response(self, content: str = "• bullet"):
        import httpx
        resp = MagicMock(spec=httpx.Response)
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value={
            "choices": [{"message": {"content": content}}]
        })
        return resp

    @pytest.mark.anyio
    async def test_retries_once_on_429_then_succeeds(self):
        from app.chains import _groq_chat
        bad_resp, _ = _make_http_status_error(429, retry_after="0")
        good_resp = self._good_response()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=[bad_resp, good_resp])

        with patch("app.chains.get_http", return_value=mock_client), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            result = await _groq_chat([{"role": "user", "content": "test"}])

        assert mock_client.post.call_count == 2
        assert result == "• bullet"

    @pytest.mark.anyio
    async def test_raises_after_max_retries(self):
        import httpx
        from app.chains import _groq_chat
        bad_resp, _ = _make_http_status_error(429, retry_after="0")

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=bad_resp)

        with patch("app.chains.get_http", return_value=mock_client), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(httpx.HTTPStatusError):
                await _groq_chat([{"role": "user", "content": "test"}])

        assert mock_client.post.call_count == _GROQ_MAX_RETRIES

    @pytest.mark.anyio
    async def test_non_429_error_raises_immediately(self):
        import httpx
        from app.chains import _groq_chat
        bad_resp, _ = _make_http_status_error(500)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=bad_resp)

        with patch("app.chains.get_http", return_value=mock_client), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(httpx.HTTPStatusError):
                await _groq_chat([{"role": "user", "content": "test"}])

        assert mock_client.post.call_count == 1  # no retry for non-429

    @pytest.mark.anyio
    async def test_raises_quota_exhausted_when_retry_after_exceeds_cap(self):
        """The hosted provider returns Retry-After in the thousands of seconds
        when the daily token quota is empty. Blocking the request for an hour
        just hangs the client and ties up the slot — fail fast with
        HostedLLMQuotaExhausted so the endpoint can return 503 with a clear
        message instead.
        """
        from app.chains import _groq_chat, HostedLLMQuotaExhausted, settings
        # Retry-After = 4402s (the exact value from the user's bug log).
        bad_resp, _ = _make_http_status_error(429, retry_after="4402")
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=bad_resp)

        with patch("app.chains.get_http", return_value=mock_client), \
             patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep, \
             patch.object(settings, "groq_max_retry_wait_seconds", 30):
            with pytest.raises(HostedLLMQuotaExhausted) as exc_info:
                await _groq_chat([{"role": "user", "content": "test"}])

        assert exc_info.value.retry_after_seconds == 4402.0
        assert "quota exhausted" in str(exc_info.value).lower()
        # We must NOT have slept for the 4402-second value — that's the whole
        # point of the cap. Single failed attempt, then immediate raise.
        assert mock_client.post.call_count == 1
        mock_sleep.assert_not_called()

    @pytest.mark.anyio
    async def test_circuit_breaker_fails_subsequent_calls_without_network(self):
        """Once one call learns the quota is exhausted, subsequent calls in the
        same process must fail fast WITHOUT making another doomed Groq
        request. Without this, /parse fires N back-to-back 429s (one per
        project) and burns Groq's request-quota even though the LLM never
        runs. The exact symptom in the user's bug report: 7 doomed 429s in
        ~300ms after the first quota-exhausted response.
        """
        from app.chains import _groq_chat, HostedLLMQuotaExhausted, settings
        bad_resp, _ = _make_http_status_error(429, retry_after="1074")
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=bad_resp)

        with patch("app.chains.get_http", return_value=mock_client), \
             patch("asyncio.sleep", new_callable=AsyncMock), \
             patch.object(settings, "groq_max_retry_wait_seconds", 30):
            # First call: trips the breaker.
            with pytest.raises(HostedLLMQuotaExhausted):
                await _groq_chat([{"role": "user", "content": "test"}])
            calls_after_first = mock_client.post.call_count
            # Second call: must NOT hit the network — breaker fails it instantly.
            with pytest.raises(HostedLLMQuotaExhausted):
                await _groq_chat([{"role": "user", "content": "test"}])

        # Total post count must equal what it was after the first call.
        # If it increased, the breaker isn't preventing the doomed roundtrip.
        assert mock_client.post.call_count == calls_after_first, (
            f"Circuit breaker leaked: post called {mock_client.post.call_count} times "
            f"(expected {calls_after_first}). Subsequent calls must fail without network IO."
        )

    @pytest.mark.anyio
    async def test_circuit_breaker_clears_after_cooldown(self):
        """After Retry-After elapses, the breaker auto-closes and calls retry
        Groq normally. Implemented via a monotonic deadline, not a counter.
        """
        from app.chains import _groq_chat, settings, _trip_quota_circuit
        good_resp = self._good_response()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=good_resp)

        # Manually trip the breaker with a wait already in the past so the
        # check on entry treats it as closed.
        _trip_quota_circuit(-1.0)

        with patch("app.chains.get_http", return_value=mock_client), \
             patch.object(settings, "groq_max_retry_wait_seconds", 30):
            result = await _groq_chat([{"role": "user", "content": "test"}])

        assert result == "• bullet"
        assert mock_client.post.call_count == 1

    @pytest.mark.anyio
    async def test_still_retries_when_retry_after_within_cap(self):
        """Per-minute rate limits cite ≤60s. Those should still retry normally
        without raising — the cap is only for clearly-quota-exhausted values.
        """
        from app.chains import _groq_chat, settings
        bad_resp, _ = _make_http_status_error(429, retry_after="5")
        good_resp = self._good_response()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=[bad_resp, good_resp])

        with patch("app.chains.get_http", return_value=mock_client), \
             patch("asyncio.sleep", new_callable=AsyncMock), \
             patch.object(settings, "groq_max_retry_wait_seconds", 30):
            result = await _groq_chat([{"role": "user", "content": "test"}])

        assert result == "• bullet"
        assert mock_client.post.call_count == 2
