"""
tests/test_parser.py
Unit tests for app/parser.py — no live Ollama or file system needed.
All LLM calls are mocked; document parsing is tested with synthetic bytes.
"""

import io
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.parser import (
    RawProject,
    _BULLET_MARKER_CHARS,
    _clean_bullet_text,
    _is_label,
    _is_pgdba_template,
    _is_project_heading,
    _make_slug,
    _strip_fences,
    extract_facts,
    parse_and_stream,
    parse_document_bytes,
    parse_pdf_bytes,
)
from app.models import CoreFact


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

class TestMakeSlug:
    def test_simple_title(self):
        slug = _make_slug("Cuckoo.ai")
        parts = slug.rsplit("-", 1)
        assert parts[0] == "cuckoo-ai"
        assert len(parts[1]) == 6                    # 6-char hex hash suffix

    def test_long_title_truncated(self):
        slug = _make_slug("Time Series Analysis – Hourly Wages and More Stuff")
        assert len(slug) <= 27                        # 20 prefix + dash + 6 hash

    def test_special_characters_removed(self):
        slug = _make_slug("SARIMA(2,0,0)(1,0,0)[12] Analysis")
        assert "(" not in slug
        assert "[" not in slug

    def test_returns_lowercase(self):
        assert _make_slug("MyProject") == _make_slug("myproject")

    def test_similar_titles_produce_unique_slugs(self):
        s1 = _make_slug("Time Series Analysis – Hourly Wages")
        s2 = _make_slug("Time Series Analysis of Stock Returns")
        assert s1 != s2


class TestStripFences:
    def test_plain_json(self):
        assert _strip_fences('[{"a": 1}]') == '[{"a": 1}]'

    def test_json_fence(self):
        assert _strip_fences('```json\n[{"a": 1}]\n```') == '[{"a": 1}]'

    def test_plain_fence(self):
        assert _strip_fences('```\n[{"a": 1}]\n```') == '[{"a": 1}]'

    def test_whitespace_stripped(self):
        result = _strip_fences("  \n[1,2,3]\n  ")
        assert result == "[1,2,3]"


class TestIsLabel:
    def test_label_with_colon(self):
        assert _is_label("Data & Preprocessing:") is True

    def test_label_model_development(self):
        assert _is_label("SARIMA Model Development:") is True

    def test_not_label_no_colon(self):
        assert _is_label("Built a forecasting model") is False

    def test_not_label_long_line(self):
        # Long lines with colons are probably sentences, not labels
        long = "Built SARIMA(2,0,0)(1,0,0)[12] model using ACF/PACF: reduced RMSE to 0.250"
        assert _is_label(long) is False


# ─────────────────────────────────────────────────────────────────────────────
# PDF parsing (mocked pdfplumber)
# ─────────────────────────────────────────────────────────────────────────────

class TestPdfParsing:
    def test_extracts_bullet_lines(self):
        """Mock pdfplumber to return text with ▪-prefixed bullets."""
        mock_page = MagicMock()
        mock_page.extract_text.return_value = (
            "ACADEMIC PROJECTS\n"
            "  ▪ Built SARIMA(2,0,0)(1,0,0)[12] reducing RMSE to 0.250\n"
            "  ▪ Conducted ADF test and validated residuals with Ljung-Box\n"
            "  Short line\n"                   # < 30 chars — should be excluded
        )
        mock_pdf = MagicMock()
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__  = MagicMock(return_value=False)
        mock_pdf.pages     = [mock_page]

        with patch("pdfplumber.open", return_value=mock_pdf):
            projects = parse_pdf_bytes(b"fake-pdf-bytes")

        assert len(projects) == 1
        assert projects[0].title == "CV Bullets"
        assert len(projects[0].bullets) == 2
        assert "RMSE" in projects[0].bullets[0]

    def test_empty_pdf_returns_empty(self):
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "No bullet content here."
        mock_pdf = MagicMock()
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__  = MagicMock(return_value=False)
        mock_pdf.pages     = [mock_page]

        with patch("pdfplumber.open", return_value=mock_pdf):
            projects = parse_pdf_bytes(b"fake-pdf-bytes")

        assert projects == []


# ─────────────────────────────────────────────────────────────────────────────
# PGDBA-template PDF parsing (private-use bullet glyphs, two-column layout)
# ─────────────────────────────────────────────────────────────────────────────

def _word(text: str, x0: float, top: float) -> dict:
    """Build a pdfplumber-shaped word box with sane bottom/x1 estimates."""
    return {"text": text, "x0": x0, "top": top, "x1": x0 + len(text) * 5, "bottom": top + 10}


def _char(text: str, x0: float, top: float) -> dict:
    return {"text": text, "x0": x0, "top": top, "x1": x0 + 4, "bottom": top + 10}


def _make_pgdba_page(words: list[dict], chars: list[dict], width: float = 600.0):
    page = MagicMock()
    page.extract_words.return_value = words
    page.chars = chars
    page.width = width
    # Generic-fallback path uses extract_text — leave it harmless in case PGDBA path bails.
    page.extract_text.return_value = ""
    return page


def _open_pdf_with_pages(pages: list) -> MagicMock:
    pdf = MagicMock()
    pdf.__enter__ = MagicMock(return_value=pdf)
    pdf.__exit__  = MagicMock(return_value=False)
    pdf.pages     = pages
    return pdf


class TestPgdbaPath:
    """Coverage for _parse_pdf_pgdba — the coordinate-based extractor used when
    the PDF's char layer contains the PGDBA private-use bullet glyphs."""

    def test_is_pgdba_template_detects_wingdings_marker(self):
        # The Wingdings glyph U+F0A7 is what pdfplumber returns in page.chars
        # for the bullet markers used across most PGDBA CVs.
        chars = [_char("", 110, 90), _char("B", 130, 90)]
        assert _is_pgdba_template(chars) is True

    def test_is_pgdba_template_false_without_markers(self):
        chars = [_char("B", 130, 90), _char("u", 138, 90)]
        assert _is_pgdba_template(chars) is False
        # And the marker set itself contains exactly what we expect.
        assert _BULLET_MARKER_CHARS == {"", "§"}

    def test_clean_bullet_text_strips_mid_line_marker_split(self):
        # Real-world artifact: "LMs] ▪ Enhanced..." — keep the longest segment.
        out = _clean_bullet_text("LMs] ▪ Enhanced retrieval pipeline reducing latency by 40ms")
        assert out.startswith("Enhanced")
        assert "▪" not in out

    def test_pgdba_path_segments_into_multiple_projects(self):
        # Page width 600 → marker at x0=110 → threshold = max(30, 107) = 107 pt.
        # Words with x0 ≤ 107 land in the left column (project titles); words
        # with x0 > 107 are right-column bullet content.
        words = [
            # Section header — toggles in_target = True
            _word("ACADEMIC",  20, 50),  _word("PROJECTS",  90, 50),
            # Project title #1 — left column only
            _word("Time",      20, 70),  _word("Series",    50, 70),
            # Bullet for project #1 — right column only
            _word("Built",    130, 90),  _word("SARIMA",   180, 90),
            _word("reducing", 240, 90),  _word("RMSE",     320, 90),
            _word("to",       370, 90),  _word("0.250",    400, 90),
            _word("with",     440, 90),  _word("careful",  480, 90),
            _word("tuning",   530, 90),
            # Project title #2 — left column only
            _word("ML",        20, 110), _word("Project",   50, 110),
            # Bullet for project #2 — right column only
            _word("Trained",  130, 130), _word("XGBoost",  180, 130),
            _word("achieving",240, 130), _word("87%",      320, 130),
            _word("accuracy", 360, 130), _word("on",       420, 130),
            _word("validation",450, 130),
        ]
        chars = [_char("", 110, 90), _char("", 110, 130)]
        pdf   = _open_pdf_with_pages([_make_pgdba_page(words, chars)])

        with patch("pdfplumber.open", return_value=pdf):
            projects = parse_pdf_bytes(b"fake")

        assert [p.title for p in projects] == ["Time Series", "ML Project"]
        assert len(projects[0].bullets) == 1
        assert "SARIMA" in projects[0].bullets[0]
        assert "0.250"  in projects[0].bullets[0]   # number preserved
        assert len(projects[1].bullets) == 1
        assert "XGBoost" in projects[1].bullets[0]
        assert "87%"     in projects[1].bullets[0]

    def test_pgdba_path_skips_bullets_after_stop_header(self):
        # WORK EXPERIENCE bullets must NOT be ingested as project content —
        # the section gate flips off when a stop header is hit.
        words = [
            _word("ACADEMIC",  20, 50),  _word("PROJECTS",  90, 50),
            _word("Real",      20, 70),  _word("Project",   50, 70),
            _word("Built",    130, 90),  _word("SARIMA",   180, 90),
            _word("reducing", 240, 90),  _word("RMSE",     320, 90),
            _word("to",       370, 90),  _word("0.250",    400, 90),
            _word("over",     440, 90),  _word("baseline", 480, 90),
            # Stop header — gate closes here
            _word("WORK",      20, 110), _word("EXPERIENCE",70, 110),
            # Bullet under WORK EXPERIENCE — must be ignored
            _word("Worked",   130, 130), _word("at",       190, 130),
            _word("CompanyX", 220, 130), _word("on",       290, 130),
            _word("internal", 320, 130), _word("tools",    390, 130),
            _word("for",      440, 130), _word("logging",  470, 130),
        ]
        chars = [_char("", 110, 90), _char("", 110, 130)]
        pdf   = _open_pdf_with_pages([_make_pgdba_page(words, chars)])

        with patch("pdfplumber.open", return_value=pdf):
            projects = parse_pdf_bytes(b"fake")

        assert len(projects) == 1
        assert projects[0].title == "Real Project"
        assert len(projects[0].bullets) == 1
        assert "SARIMA" in projects[0].bullets[0]
        # The post-stop-header bullet must NOT have leaked in.
        assert all("Worked" not in b for b in projects[0].bullets)


# ─────────────────────────────────────────────────────────────────────────────
# parse_document_bytes dispatch
# ─────────────────────────────────────────────────────────────────────────────

class TestParseDocumentBytes:
    def test_unsupported_extension_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            parse_document_bytes(b"data", "resume.txt")

    def test_unsupported_csv_raises(self):
        with pytest.raises(ValueError):
            parse_document_bytes(b"data", "data.csv")

    def test_docx_routes_to_docx_parser(self):
        with patch("app.parser.parse_docx_bytes", return_value=[]) as mock:
            parse_document_bytes(b"x", "file.docx")
        mock.assert_called_once()

    def test_pdf_routes_to_pdf_parser(self):
        with patch("app.parser.parse_pdf_bytes", return_value=[]) as mock:
            parse_document_bytes(b"x", "file.pdf")
        mock.assert_called_once()

    def test_case_insensitive_extension(self):
        with patch("app.parser.parse_docx_bytes", return_value=[]) as mock:
            parse_document_bytes(b"x", "File.DOCX")
        mock.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# LLM fact extraction
# ─────────────────────────────────────────────────────────────────────────────

def _mock_llm(response_content: str):
    """Helper: return an AsyncMock of _ollama_chat that returns the given string."""
    return AsyncMock(return_value=response_content)


class TestExtractFacts:
    @pytest.mark.asyncio
    async def test_success_returns_core_facts(self):
        raw = json.dumps([{
            "fact_id":  "ts-1",
            "text":     "Built SARIMA(2,0,0)(1,0,0)[12] model reducing RMSE to 0.250",
            "tools":    ["SARIMA", "SLSQP"],
            "metrics":  ["RMSE=0.250"],
            "outcome":  "reduced forecasting error",
        }])
        project = RawProject("Time Series", ["Built SARIMA model reducing RMSE to 0.250"])
        with patch("app.chains._ollama_chat", new=_mock_llm(raw)):
            facts = await extract_facts(project)

        assert len(facts) == 1
        assert facts[0].fact_id   in ("ts-1", "time-series-1")
        assert facts[0].metrics   == ["RMSE=0.250"]
        assert "SARIMA" in facts[0].tools

    @pytest.mark.asyncio
    async def test_metric_preserved_exactly(self):
        """Numbers must never be altered during extraction."""
        raw = json.dumps([{
            "fact_id":  "p-1",
            "text":     "Achieved RMSE=0.250 and AUC=0.944",
            "tools":    [],
            "metrics":  ["RMSE=0.250", "AUC=0.944"],
            "outcome":  "",
        }])
        project = RawProject("My Project", ["RMSE 0.250, AUC 0.944"])
        with patch("app.chains._ollama_chat", new=_mock_llm(raw)):
            facts = await extract_facts(project)

        assert "RMSE=0.250" in facts[0].metrics
        assert "AUC=0.944"  in facts[0].metrics

    @pytest.mark.asyncio
    async def test_bad_json_falls_back_to_bullets(self):
        project = RawProject("Project X", [
            "Built a model with 87% accuracy using XGBoost",
            "Reduced latency by 40ms at p99",
        ])
        with patch("app.chains._ollama_chat", new=_mock_llm("not valid json {{")):
            facts = await extract_facts(project)

        assert len(facts) >= 1
        assert all(isinstance(f, CoreFact) for f in facts)

    @pytest.mark.asyncio
    async def test_network_error_falls_back(self):
        project = RawProject("Network Project", ["Built something with RMSE 0.5"])
        with patch("app.chains._ollama_chat", new=AsyncMock(side_effect=Exception("connection refused"))):
            facts = await extract_facts(project)

        assert len(facts) >= 1

    @pytest.mark.asyncio
    async def test_non_list_response_falls_back(self):
        raw     = json.dumps({"error": "unexpected object"})
        project = RawProject("My Project", ["A bullet with 42% improvement"])
        with patch("app.chains._ollama_chat", new=_mock_llm(raw)):
            facts = await extract_facts(project)

        assert len(facts) >= 1

    @pytest.mark.asyncio
    async def test_multiple_facts_returned(self):
        raw = json.dumps([
            {"fact_id": "p-1", "text": "Fact one", "tools": ["Python"], "metrics": ["87%"], "outcome": ""},
            {"fact_id": "p-2", "text": "Fact two", "tools": ["SQL"],    "metrics": ["42ms"], "outcome": ""},
            {"fact_id": "p-3", "text": "Fact three","tools": [],        "metrics": [],       "outcome": ""},
        ])
        project = RawProject("Multi", ["b1", "b2", "b3"])
        with patch("app.chains._ollama_chat", new=_mock_llm(raw)):
            facts = await extract_facts(project)

        assert len(facts) == 3

    @pytest.mark.asyncio
    async def test_handles_many_bullets_within_cap(self):
        """Projects at or below _EXTRACT_MAX_BULLETS bullets should not trigger a warning."""
        from app.parser import _EXTRACT_MAX_BULLETS
        project = RawProject("Big Project", [f"Bullet {i} with metric {i}%" for i in range(_EXTRACT_MAX_BULLETS)])
        raw = json.dumps([
            {"fact_id": "bp-1", "text": "Summary fact", "tools": [], "metrics": ["5%"], "outcome": ""}
        ])
        with patch("app.chains._ollama_chat", new=_mock_llm(raw)), \
             patch("app.parser.logger") as mock_log:
            facts = await extract_facts(project)
        mock_log.warning.assert_not_called()
        assert len(facts) >= 1

    @pytest.mark.asyncio
    async def test_warns_and_truncates_beyond_cap(self):
        """Warns when a project exceeds _EXTRACT_MAX_BULLETS bullets (M7)."""
        from app.parser import _EXTRACT_MAX_BULLETS
        oversized = RawProject("Big Project", [f"Bullet {i} with metric {i}%" for i in range(_EXTRACT_MAX_BULLETS + 5)])
        raw = json.dumps([
            {"fact_id": "bp-1", "text": "Summary fact", "tools": [], "metrics": ["5%"], "outcome": ""}
        ])
        with patch("app.chains._ollama_chat", new=_mock_llm(raw)), \
             patch("app.parser.logger") as mock_log:
            facts = await extract_facts(oversized)
        mock_log.warning.assert_called_once()
        assert len(facts) >= 1

    @pytest.mark.asyncio
    async def test_fence_stripped_before_json_parse(self):
        raw = '```json\n[{"fact_id":"f-1","text":"Test fact","tools":[],"metrics":[],"outcome":""}]\n```'
        project = RawProject("Fenced", ["Test fact"])
        with patch("app.chains._ollama_chat", new=_mock_llm(raw)):
            facts = await extract_facts(project)
        assert len(facts) == 1
        assert facts[0].fact_id == "f-1"


# ─────────────────────────────────────────────────────────────────────────────
# Full streaming pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _mock_llm_for_extract(num_facts: int = 1):
    """Return a mock _ollama_chat that returns `num_facts` valid facts as JSON."""
    facts = [
        {"fact_id": f"p-{i}", "text": f"Fact {i} with 87% improvement", "tools": [], "metrics": [f"{i}%"], "outcome": ""}
        for i in range(num_facts)
    ]
    return AsyncMock(return_value=json.dumps(facts))


class TestParseAndStream:
    @pytest.mark.asyncio
    async def test_unsupported_type_yields_error(self):
        events = [(et, d) async for et, d in
                  parse_and_stream(b"x", "resume.txt")]
        assert events[0][0] == "error"

    @pytest.mark.asyncio
    async def test_empty_doc_yields_error(self):
        with patch("app.parser.parse_docx_bytes", return_value=[]):
            events = [(et, d) async for et, d in
                      parse_and_stream(b"x", "empty.docx")]
        assert events[0][0] == "error"
        assert "No projects" in events[0][1]["error_message"]

    @pytest.mark.asyncio
    async def test_parse_error_yields_error(self):
        with patch("app.parser.parse_docx_bytes", side_effect=Exception("corrupt")):
            events = [(et, d) async for et, d in
                      parse_and_stream(b"x", "bad.docx")]
        assert events[0][0] == "error"

    @pytest.mark.asyncio
    async def test_single_project_yields_progress_project_done(self):
        with patch("app.parser.parse_docx_bytes",
                   return_value=[RawProject("Test Project", ["Built a model with RMSE=0.5"])]), \
             patch("app.chains._ollama_chat", new=_mock_llm_for_extract(1)):
            events = [(et, d) async for et, d in
                      parse_and_stream(b"x", "test.docx")]

        types = [e[0] for e in events]
        assert "progress" in types
        assert "project"  in types
        assert "done"     in types
        assert types[-1]  == "done"

    @pytest.mark.asyncio
    async def test_two_projects_yield_two_project_events(self):
        with patch("app.parser.parse_docx_bytes", return_value=[
            RawProject("Project A", ["Bullet with 87% metric"]),
            RawProject("Project B", ["Bullet with RMSE 0.5"]),
        ]), patch("app.chains._ollama_chat", new=_mock_llm_for_extract(1)):
            events = [(et, d) async for et, d in
                      parse_and_stream(b"x", "test.docx")]

        project_events = [d for et, d in events if et == "project"]
        assert len(project_events) == 2
        assert project_events[0]["project"]["title"] == "Project A"
        assert project_events[1]["project"]["title"] == "Project B"

    @pytest.mark.asyncio
    async def test_done_event_contains_correct_counts(self):
        with patch("app.parser.parse_docx_bytes", return_value=[
            RawProject("P1", ["b1 with 42%"]),
            RawProject("P2", ["b2 with 87%"]),
        ]), patch("app.chains._ollama_chat", new=_mock_llm_for_extract(2)):
            events = [(et, d) async for et, d in
                      parse_and_stream(b"x", "test.docx")]

        done = next(d for et, d in events if et == "done")
        assert done["total_projects"] == 2
        assert done["total_facts"]    == 4   # 2 projects × 2 facts each

    @pytest.mark.asyncio
    async def test_project_event_contains_valid_project_data(self):
        with patch("app.parser.parse_docx_bytes",
                   return_value=[RawProject("Cuckoo.ai", ["Bullet with 87% improvement"])]), \
             patch("app.chains._ollama_chat", new=_mock_llm_for_extract(1)):
            events = [(et, d) async for et, d in
                      parse_and_stream(b"x", "test.docx")]

        proj_event = next(d for et, d in events if et == "project")
        project = proj_event["project"]
        assert project["title"]       == "Cuckoo.ai"
        assert "project_id"           in project
        assert len(project["core_facts"]) >= 1

    @pytest.mark.asyncio
    async def test_fact_extraction_failure_still_yields_project(self):
        """Even if LLM fails, project is yielded with fallback facts."""
        with patch("app.parser.parse_docx_bytes",
                   return_value=[RawProject("Resilient Project", ["A bullet with 50% improvement"])]), \
             patch("app.chains._ollama_chat", new=AsyncMock(side_effect=Exception("LLM down"))):
            events = [(et, d) async for et, d in
                      parse_and_stream(b"x", "test.docx")]

        types = [e[0] for e in events]
        assert "project" in types
        assert "done"    in types

    @pytest.mark.asyncio
    async def test_progress_events_precede_project_events(self):
        with patch("app.parser.parse_docx_bytes",
                   return_value=[RawProject("P1", ["b1 with metric 40%"])]), \
             patch("app.chains._ollama_chat", new=_mock_llm_for_extract(1)):
            events = [(et, d) async for et, d in
                      parse_and_stream(b"x", "test.docx")]

        types = [e[0] for e in events]
        first_project = next(i for i, t in enumerate(types) if t == "project")
        # There must be at least one 'progress' before the first 'project'
        assert any(types[j] == "progress" for j in range(first_project))
