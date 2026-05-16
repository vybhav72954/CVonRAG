"""
tests/test_parser.py
Unit tests for app/parser.py — no live Ollama or file system needed.
All LLM calls are mocked; document parsing is tested with synthetic bytes.
"""

import io
import json
import re
from pathlib import Path

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


def _make_pgdba_page(words: list[dict], chars: list[dict],
                     width: float = 600.0, height: float = 800.0):
    page = MagicMock()
    page.extract_words.return_value = words
    page.chars = chars
    page.width = width
    page.height = height
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
        # And the marker set itself contains exactly what we expect:
        #   (Wingdings, private-use), § (section sign),
        # ▪ U+25AA (added in issue #24 to recognise Word-exported
        # PGDBA templates that use the standard Unicode bullet rather
        # than the private-use Wingdings glyph).
        assert _BULLET_MARKER_CHARS == {"", "§", "▪"}

    def test_is_pgdba_template_detects_unicode_black_square(self):
        # U+25AA is the standard Unicode "Black Small Square" used by
        # some Word-exported PGDBA templates (e.g. 24BM6JP47). Prior to
        # issue #24 the template sniff missed these CVs and they fell
        # through to the regex fallback which collapsed every project
        # into "CV Bullets".
        chars = [_char("▪", 110, 90), _char("B", 130, 90)]
        assert _is_pgdba_template(chars) is True

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

    def test_pgdba_parallel_title_layout(self):
        """Issue #24 — titles wrap parallel to bullets, not above them.

        Real PGDBA CVs use a layout where the project title spans 2–3 rows in
        the left column *at the same Y-coordinates as the bullets* in the
        right column. The old `left_words and not right_words` rule (which
        only fired on stray title-only rows) collapsed every project into the
        "CV Bullets" fallback. The Y-gap clustering replacement must detect
        the within-vs-between project boundary from line spacing alone.
        """
        # Two projects, each with 3 parallel rows (title fragment + bullet).
        # Within-project gap = 12pt, between-project gap = 16pt. Marker at
        # x0=110, threshold ≈ 107pt.
        words = [
            _word("ACADEMIC",  20,  50), _word("PROJECTS", 90, 50),
            # Project 1, row 1
            _word("Airbnb",    20,  70), _word("Price",     50, 70),
            _word("Predicted", 130, 70), _word("listing",  200, 70),
            _word("prices",    260, 70), _word("for",      310, 70),
            _word("74k+",      340, 70), _word("Airbnb",   380, 70),
            _word("properties",430, 70),
            # Project 1, row 2 (gap 12pt)
            _word("Prediction", 20, 82), _word("Tackled",  130, 82),
            _word("outliers",  180, 82), _word("via",      230, 82),
            _word("VIF",       260, 82), _word("backward", 290, 82),
            _word("selection", 350, 82),
            # Project 1, row 3 (gap 12pt)
            _word("(Regression)",20,94),_word("Built",     130, 94),
            _word("XGBoost",   170, 94),_word("regressor", 230, 94),
            _word("achieving", 300, 94),_word("R2=0.76",   370, 94),
            # Project 2, row 1 (gap 16pt → boundary)
            _word("Natural",   20, 110), _word("Gas",       60, 110),
            _word("Forecasted",130,110), _word("monthly",  200, 110),
            _word("gas",       260,110), _word("consumption",290,110),
            _word("over",      380,110), _word("50",       420, 110),
            _word("years",     440,110),
            # Project 2, row 2 (gap 12pt)
            _word("Consumption",20,122), _word("ADF",      130, 122),
            _word("test",      170,122), _word("for",      210, 122),
            _word("stationarity",240,122),_word("then",    320, 122),
            _word("differenced",360,122),
            # Project 2, row 3 (gap 12pt)
            _word("(Time",     20,134), _word("Series)",   60, 134),
            _word("SARIMA",   130,134), _word("MAPE",     180, 134),
            _word("3.5%",     220,134), _word("Ljung-Box",260, 134),
            _word("residual", 330,134),
        ]
        # Markers at every bullet row to anchor the column threshold.
        chars = [_char("", 110, y) for y in (70, 82, 94, 110, 122, 134)]
        pdf   = _open_pdf_with_pages([_make_pgdba_page(words, chars)])

        with patch("pdfplumber.open", return_value=pdf):
            projects = parse_pdf_bytes(b"fake")

        assert len(projects) == 2
        # Title is the concatenation of all left-column words in the cluster.
        assert projects[0].title == "Airbnb Price Prediction (Regression)"
        assert projects[1].title == "Natural Gas Consumption (Time Series)"
        # Bullets stay attached to their correct project.
        assert any("XGBoost" in b for b in projects[0].bullets)
        assert any("SARIMA" in b for b in projects[1].bullets)
        # No cross-contamination: SARIMA isn't in the Airbnb project, etc.
        assert all("SARIMA" not in b for b in projects[0].bullets)
        assert all("XGBoost" not in b for b in projects[1].bullets)

    def test_pgdba_per_section_clustering(self):
        """Issue #24 — each project section clusters against its own gap
        distribution. A single global threshold can miss within-vs-between
        boundaries in one section because of noise in another. Each section
        gets 2 projects × 3 rows so per-section Y-gap detection has enough
        signal to recover the within-vs-between split (12pt vs 16pt).
        """
        # Within-project rows step by 12pt; project boundary jumps by 16pt;
        # section transition jumps by 30pt.
        # Bullet text must exceed 30 chars to pass the length filter, so each
        # bullet row carries enough words to comfortably clear it.
        words = [
            _word("ACADEMIC", 20, 50), _word("PROJECTS", 90, 50),
            # ── ACADEMIC: Project A (3 parallel rows)
            _word("Project",   20,  70), _word("A1",        50,  70),
            _word("Built",    130,  70), _word("baseline", 180,  70),
            _word("OLS",      260,  70), _word("regression",300, 70),
            _word("model",    380,  70), _word("Adj-R2=0.78",430, 70),
            _word("Title",     20,  82), _word("two",       50,  82),
            _word("Handled",  130,  82), _word("VIF",      200,  82),
            _word("outliers", 240,  82), _word("Jackknife",310, 82),
            _word("influential",380, 82),_word("rows",     450,  82),
            _word("(A)",       20,  94),
            _word("Achieved", 130,  94), _word("test",     200,  94),
            _word("R2=0.85",  250,  94),_word("with",     320,  94),
            _word("ElasticNet",360, 94),_word("model",    430,  94),
            # ── Project B (gap 16pt to row at y=110)
            _word("Project",   20, 110), _word("B1",        50, 110),
            _word("Forecasted",130, 110),_word("hourly",   210, 110),
            _word("energy",   260, 110), _word("demand",   310, 110),
            _word("over",     360, 110), _word("18",       400, 110),
            _word("months",   420, 110),
            _word("Title",     20, 122), _word("two",       50, 122),
            _word("Stationarity",130,122),_word("via",     220, 122),
            _word("ADF",      260, 122), _word("then",     300, 122),
            _word("differenced",340,122),_word("series",   420, 122),
            _word("(B)",       20, 134),
            _word("Final",    130, 134),_word("SARIMA",   170, 134),
            _word("MAPE",     230, 134),_word("3.5%",     280, 134),
            _word("Ljung-Box",320, 134),_word("residual", 390, 134),
            _word("clean",    450, 134),
            # ── ADDITIONAL header (section transition gap ~26pt)
            _word("ADDITIONAL",20, 160),_word("PROJECTS",  90, 160),
            # ── Project C (3 rows in ADDITIONAL)
            _word("Project",   20, 180), _word("C1",        50, 180),
            _word("Built",    130, 180), _word("RAG",      180, 180),
            _word("retrieval",220, 180), _word("with",     290, 180),
            _word("FAISS",    330, 180), _word("vector",   380, 180),
            _word("index",    430, 180),
            _word("Title",     20, 192), _word("two",       50, 192),
            _word("Chunked",  130, 192), _word("docs",     200, 192),
            _word("via",      250, 192), _word("LangChain",290, 192),
            _word("recursive",360, 192),_word("splitter",  430, 192),
            _word("(C)",       20, 204),
            _word("Retrieved",130, 204), _word("top-k",    210, 204),
            _word("results",  260, 204), _word("with",     310, 204),
            _word("cosine",   350, 204), _word("scores",   400, 204),
            # ── Project D (gap 16pt to y=220)
            _word("Project",   20, 220), _word("D1",        50, 220),
            _word("Clustered",130, 220), _word("users",    210, 220),
            _word("with",     270, 220), _word("DBSCAN",   320, 220),
            _word("on",       380, 220), _word("RFM",      410, 220),
            _word("features", 440, 220),
            _word("Title",     20, 232), _word("two",       50, 232),
            _word("Customer", 130, 232), _word("segments", 200, 232),
            _word("derived",  280, 232), _word("via",      340, 232),
            _word("PCA",      380, 232), _word("reduction",410, 232),
            _word("(D)",       20, 244),
            _word("Silhouette",130, 244),_word("score",    220, 244),
            _word("0.62",     270, 244), _word("on",       310, 244),
            _word("held-out", 340, 244),_word("data",     410, 244),
        ]
        chars = [_char("", 110, y) for y in
                 (70, 82, 94, 110, 122, 134, 180, 192, 204, 220, 232, 244)]
        pdf   = _open_pdf_with_pages([_make_pgdba_page(words, chars)])

        with patch("pdfplumber.open", return_value=pdf):
            projects = parse_pdf_bytes(b"fake")

        # Two projects per section, in document order.
        assert len(projects) == 4
        # Titles are the concatenated left-column words of each cluster.
        assert projects[0].title == "Project A1 Title two (A)"
        assert projects[1].title == "Project B1 Title two (B)"
        assert projects[2].title == "Project C1 Title two (C)"
        assert projects[3].title == "Project D1 Title two (D)"
        # Bullets stay attached to their correct project; no cross-section leak.
        assert any("OLS"    in b for b in projects[0].bullets)
        assert any("SARIMA" in b for b in projects[1].bullets)
        assert any("FAISS"  in b for b in projects[2].bullets)
        assert any("DBSCAN" in b for b in projects[3].bullets)

    def test_pgdba_cross_page_section_keeps_project_boundary(self):
        """A project section spanning two pages must not fuse the last
        project on page N with the first project on page N+1.

        pdfplumber's ``word["top"]`` is page-local, so without a running
        page-Y offset the cross-page gap is a large negative number
        (e.g. ``100 - 700 = -600``). The negative value fails the
        ``gaps[i-1] > threshold`` check, silently merging the two adjacent
        projects across the page boundary. The fix maintains a cumulative
        offset that adds each prior page's height to ``y_top``.
        """
        page_height = 800.0

        # Page 1: ACADEMIC PROJECTS header + Project A (3 rows ending near
        # the bottom of the page).
        page1_words = [
            _word("ACADEMIC", 20, 50), _word("PROJECTS", 90, 50),
            _word("Project",   20, 700), _word("A1",        50, 700),
            _word("Built",    130, 700), _word("OLS",      180, 700),
            _word("regression",230, 700),_word("baseline", 300, 700),
            _word("model",    370, 700),
            _word("Title",     20, 712), _word("two",       50, 712),
            _word("Handled",  130, 712), _word("VIF",      200, 712),
            _word("outliers", 240, 712), _word("Jackknife",310, 712),
            _word("(A)",       20, 724),
            _word("Achieved", 130, 724), _word("test",     200, 724),
            _word("R2=0.85",  250, 724),_word("with",     320, 724),
            _word("ElasticNet",360, 724),
        ]
        page1_chars = [_char("", 110, y) for y in (700, 712, 724)]

        # Page 2: continuation of ACADEMIC — Project B at the top of the
        # page (no fresh section header). y_top on page 2 starts back at
        # ~100, which is < the y_top of any page-1 row. Naive gap arithmetic
        # would produce ~-600 and merge A & B.
        page2_words = [
            _word("Project",   20, 100), _word("B1",        50, 100),
            _word("Forecasted",130, 100),_word("hourly",   210, 100),
            _word("energy",   260, 100), _word("demand",   310, 100),
            _word("Title",     20, 112), _word("two",       50, 112),
            _word("Stationarity",130,112),_word("via",     220, 112),
            _word("ADF",      260, 112), _word("differenced",310, 112),
            _word("(B)",       20, 124),
            _word("Final",    130, 124),_word("SARIMA",   170, 124),
            _word("MAPE",     230, 124),_word("3.5%",     280, 124),
            _word("Ljung-Box",320, 124),_word("residual", 390, 124),
        ]
        page2_chars = [_char("", 110, y) for y in (100, 112, 124)]

        pdf = _open_pdf_with_pages([
            _make_pgdba_page(page1_words, page1_chars, height=page_height),
            _make_pgdba_page(page2_words, page2_chars, height=page_height),
        ])

        with patch("pdfplumber.open", return_value=pdf):
            projects = parse_pdf_bytes(b"fake")

        # Two distinct projects, not one fused mega-project.
        assert len(projects) == 2
        assert projects[0].title.startswith("Project A1")
        assert projects[1].title.startswith("Project B1")
        # OLS belongs to A; SARIMA belongs to B — no cross-page leak.
        assert any("OLS"    in b for b in projects[0].bullets)
        assert all("SARIMA" not in b for b in projects[0].bullets)
        assert any("SARIMA" in b for b in projects[1].bullets)
        assert all("OLS"    not in b for b in projects[1].bullets)


# ─────────────────────────────────────────────────────────────────────────────
# parse_document_bytes dispatch
# ─────────────────────────────────────────────────────────────────────────────

class TestParseDocumentBytes:
    def test_unsupported_extension_user_path_raises_docx_only(self):
        """User path: any non-.docx extension gets the docx-only error
        (issue #28). No format-name leakage to the caller."""
        with pytest.raises(ValueError, match=r"Only \.docx biodata files are supported"):
            parse_document_bytes(b"data", "resume.txt")

    def test_unsupported_extension_admin_path_raises_generic(self):
        """Admin path keeps the original 'Use .docx or .pdf' message so the
        ingest / eval scripts surface the more permissive contract."""
        with pytest.raises(ValueError, match="Unsupported"):
            parse_document_bytes(b"data", "resume.txt", caller="admin")

    def test_unsupported_csv_raises(self):
        with pytest.raises(ValueError):
            parse_document_bytes(b"data", "data.csv")

    def test_docx_routes_to_docx_parser(self):
        with patch("app.parser.parse_docx_bytes", return_value=[]) as mock:
            parse_document_bytes(b"x", "file.docx")
        mock.assert_called_once()

    def test_pdf_rejected_on_user_path(self):
        """Issue #28: PDFs are not supported on the user upload path. Verify
        the ValueError message carries the docx-only conversion guidance so
        the API layer can surface it back to the user verbatim."""
        with patch("app.parser.parse_pdf_bytes", return_value=[]) as pdf_mock:
            with pytest.raises(ValueError, match=r"Only \.docx biodata files are supported"):
                parse_document_bytes(b"x", "file.pdf")
        pdf_mock.assert_not_called()

    def test_pdf_routes_to_pdf_parser_on_admin_path(self):
        """Admin / eval callers can still parse PDFs via caller='admin' — the
        scripts in scripts/ingest_pdfs.py, scripts/build_eval_set.py, and
        scripts/evaluate.py rely on this path."""
        with patch("app.parser.parse_pdf_bytes", return_value=[]) as mock:
            parse_document_bytes(b"x", "file.pdf", caller="admin")
        mock.assert_called_once()

    def test_case_insensitive_extension(self):
        with patch("app.parser.parse_docx_bytes", return_value=[]) as mock:
            parse_document_bytes(b"x", "File.DOCX")
        mock.assert_called_once()


# ─────────────────────────────────────────────────────────────────────────────
# Shipped sample biodata regression guard (issue #29)
# ─────────────────────────────────────────────────────────────────────────────

_SAMPLE_BIODATA_PATH = (
    Path(__file__).resolve().parent.parent / "frontend" / "static" / "sample-biodata.docx"
)


@pytest.fixture(scope="class")
def parsed_sample_biodata():
    """Parse the shipped sample once per test class.

    Skips (rather than erroring with FileNotFoundError) when the artifact is
    missing — keeps the rest of the suite triagable when CI fails on a fresh
    checkout that hasn't pulled LFS / static assets yet.
    """
    if not _SAMPLE_BIODATA_PATH.is_file():
        pytest.skip(f"Sample biodata missing at {_SAMPLE_BIODATA_PATH}")
    return parse_document_bytes(
        _SAMPLE_BIODATA_PATH.read_bytes(), "sample-biodata.docx"
    )


class TestSampleBiodata:
    """Guards the static sample shipped at /sample-biodata.docx.

    Any future parser change (heading detection, label filtering, bullet length
    threshold) that would cause first-time users to upload the sample and see a
    broken parse must trip this test in CI before it ships.
    """

    def test_sample_file_exists_and_under_size_cap(self):
        """Sample ships at the expected path and stays small enough to keep
        the static-asset bundle (and a user's first download) snappy."""
        assert _SAMPLE_BIODATA_PATH.is_file(), (
            f"Missing shipped sample at {_SAMPLE_BIODATA_PATH}"
        )
        size = _SAMPLE_BIODATA_PATH.stat().st_size
        assert size < 50_000, f"Sample biodata exceeds 50 KB cap: {size} bytes"

    def test_sample_biodata_parses_cleanly(self, parsed_sample_biodata):
        """Sample must parse into ≥4 distinct projects with ≥2 bullets each via
        the production dispatch (parse_document_bytes). Guards the structural
        contract a first-time user will see when they upload this file."""
        projects = parsed_sample_biodata
        assert len(projects) >= 4, (
            f"Sample must parse into ≥4 distinct projects, got {len(projects)}"
        )
        for p in projects:
            assert len(p.bullets) >= 2, (
                f"Project {p.title!r} has only {len(p.bullets)} bullets — needs ≥2"
            )
            assert p.title and len(p.title) >= 5, (
                f"Project title looks malformed: {p.title!r}"
            )

    def test_sample_exercises_number_preservation_surface(self, parsed_sample_biodata):
        """The sample should hit every numeric format the alchemist must preserve
        verbatim: percentages, decimals, integers (with thousands separators),
        and scientific notation. Guards against accidentally stripping examples
        when editing the sample copy."""
        all_text = " ".join(b for p in parsed_sample_biodata for b in p.bullets)

        assert re.search(r"\d+(\.\d+)?%", all_text), "no percentage in sample"
        assert re.search(r"\b\d+\.\d+\b", all_text), "no decimal in sample"
        assert re.search(r"\d{1,3},\d{3}", all_text), (
            "no thousands-separated integer in sample"
        )
        assert re.search(r"\d(\.\d+)?[eE]-?\d+", all_text), (
            "no scientific notation in sample"
        )

    def test_sample_carries_at_least_one_known_tool(self, parsed_sample_biodata):
        """The sample mentions concrete tools so downstream LLM fact extraction
        has something to populate `CoreFact.tools` with. Pure text-only bullets
        would silently degrade the demo."""
        all_text = " ".join(b for p in parsed_sample_biodata for b in p.bullets).lower()
        known_tools = ("python", "scikit-learn", "pytorch", "pandas", "numpy", "statsmodels")
        assert any(t in all_text for t in known_tools), (
            f"sample bullets mention none of {known_tools}"
        )


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
    async def test_quota_exhausted_aborts_stream_cleanly(self):
        """HostedLLMQuotaExhausted on extract_facts must abort the whole parse
        with a single error event — NOT silently fall back to bullet-verbatim
        facts for every project. The latter produces a parsed-but-junk CV (no
        tools, no metrics, no outcomes) that floors downstream scoring.
        """
        from app.chains import HostedLLMQuotaExhausted
        with patch("app.parser.parse_docx_bytes", return_value=[
            RawProject("P1", ["b1"]),
            RawProject("P2", ["b2"]),
            RawProject("P3", ["b3"]),
        ]), patch(
            "app.chains._ollama_chat",
            new=AsyncMock(side_effect=HostedLLMQuotaExhausted(1074.0)),
        ):
            events = [(et, d) async for et, d in
                      parse_and_stream(b"x", "test.docx")]

        types = [e[0] for e in events]
        # Exactly one error event, no project events, no done event.
        assert types.count("error") == 1, f"Expected 1 error event, got types={types}"
        assert "project" not in types, "Must not yield partial/degraded projects on quota exhaustion"
        assert "done" not in types, "Must not signal completion on quota exhaustion"
        error_event = next(d for et, d in events if et == "error")
        assert "quota exhausted" in error_event["error_message"].lower()

    @pytest.mark.asyncio
    async def test_caps_facts_at_project_data_max(self):
        """N5: LLM returning >12 facts must not blow up ProjectData validation.

        Cap fires inside extract_facts; downstream construction sees ≤12 facts.
        """
        from app.parser import _MAX_FACTS_PER_PROJECT
        # LLM returns 15 valid facts — more than ProjectData accepts (max_length=12).
        with patch("app.parser.parse_docx_bytes",
                   return_value=[RawProject("Big P", [f"bullet {i} with {i}%" for i in range(15)])]), \
             patch("app.chains._ollama_chat", new=_mock_llm_for_extract(15)):
            events = [(et, d) async for et, d in
                      parse_and_stream(b"x", "test.docx")]

        # Stream completed normally — no error event from validation.
        types = [e[0] for e in events]
        assert "done" in types
        assert types[-1] == "done"
        proj_event = next(d for et, d in events if et == "project")
        assert len(proj_event["project"]["core_facts"]) == _MAX_FACTS_PER_PROJECT

    @pytest.mark.asyncio
    async def test_validation_error_yields_per_project_error_and_continues(self):
        """N9: a Pydantic validation failure on one project must not kill the SSE.

        Yields an error event for the bad project and continues with the next.
        """
        with patch("app.parser.parse_docx_bytes", return_value=[
            RawProject("Bad Project", ["Bullet with 87% metric"]),
            RawProject("Good Project", ["Bullet with 42% metric"]),
        ]), patch("app.chains._ollama_chat", new=_mock_llm_for_extract(1)), \
             patch("app.parser.ProjectData", side_effect=[ValueError("bad"), MagicMock(model_dump=lambda: {"title": "Good Project", "core_facts": []})]):
            events = [(et, d) async for et, d in
                      parse_and_stream(b"x", "test.docx")]

        types = [e[0] for e in events]
        assert "error" in types               # the bad project surfaced an error event
        assert "project" in types             # the good project still yielded
        assert types[-1] == "done"            # stream completed normally
        # done count reflects only successfully-yielded projects
        done = next(d for et, d in events if et == "done")
        assert done["total_projects"] == 1

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
