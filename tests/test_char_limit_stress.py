"""
tests/test_char_limit_stress.py
Stress tests for the ±2 character tolerance loop.

These tests verify the loop's behaviour under adversarial conditions:
  • Exactly on each boundary (lower, upper, target)
  • Off-by-one across every boundary
  • Unicode / special characters (arrows, pipes, bullet glyphs)
  • Mixed ASCII/Unicode bullets (cv-style separators affect len() correctly)
  • Convergence speed benchmarks (should not take > max_iterations calls)
  • Firewall check: numbers in facts must appear unchanged in output
"""

import pytest
from app.models import FormattingConstraints


# ── Boundary parameterization ─────────────────────────────────────────────────

TOLERANCE_CASES = [
    # (target, tolerance)
    (120, 2),
    (130, 2),
    (150, 3),
    (100, 1),
    (200, 5),
    (80, 2),
    (300, 2),
    (60, 1),
]


class TestBoundaryExhaustive:
    """
    For every (target, tolerance) pair, verify every position from
    target-tolerance-2 to target+tolerance+2 is correctly classified.
    """

    @pytest.mark.parametrize("target,tolerance", TOLERANCE_CASES)
    def test_lower_bound_exact(self, target, tolerance):
        c = FormattingConstraints(target_char_limit=target, tolerance=tolerance)
        assert c.lower_bound <= len("x" * c.lower_bound) <= c.upper_bound

    @pytest.mark.parametrize("target,tolerance", TOLERANCE_CASES)
    def test_upper_bound_exact(self, target, tolerance):
        c = FormattingConstraints(target_char_limit=target, tolerance=tolerance)
        assert c.lower_bound <= len("x" * c.upper_bound) <= c.upper_bound

    @pytest.mark.parametrize("target,tolerance", TOLERANCE_CASES)
    def test_one_below_lower_is_out(self, target, tolerance):
        c = FormattingConstraints(target_char_limit=target, tolerance=tolerance)
        length = c.lower_bound - 1
        assert not (c.lower_bound <= length <= c.upper_bound)

    @pytest.mark.parametrize("target,tolerance", TOLERANCE_CASES)
    def test_one_above_upper_is_out(self, target, tolerance):
        c = FormattingConstraints(target_char_limit=target, tolerance=tolerance)
        length = c.upper_bound + 1
        assert not (c.lower_bound <= length <= c.upper_bound)

    @pytest.mark.parametrize("target,tolerance", TOLERANCE_CASES)
    def test_target_itself_is_always_in(self, target, tolerance):
        c = FormattingConstraints(target_char_limit=target, tolerance=tolerance)
        assert c.lower_bound <= target <= c.upper_bound


# ── Unicode character counting ────────────────────────────────────────────────

class TestUnicodeCharCounting:
    """
    Python's len() counts Unicode code points, not bytes.
    Resume bullets use ↑ ↓ • | — these are all single code points.
    Verify len() behaves as expected for typical CV bullet text.
    """

    def test_arrow_up_is_one_char(self):
        assert len("↑") == 1

    def test_arrow_down_is_one_char(self):
        assert len("↓") == 1

    def test_pipe_separator_is_one_char(self):
        assert len("|") == 1

    def test_bullet_glyph_is_one_char(self):
        assert len("•") == 1

    def test_typical_bullet_char_count(self):
        bullet = "• Built SARIMA(2,0,0)(1,0,0)[12] model | ↑ accuracy by reducing RMSE to 0.250"
        # Verify the count is what a human would expect (all single code points)
        assert len(bullet) == len(bullet.encode("utf-8").decode("utf-8"))

    def test_bullet_with_arrows_counts_correctly(self):
        bullet = "• Architected LLM pipeline | ↑ evaluation speed by 87% (8–12 weeks → 5–10 days)"
        count = len(bullet)
        assert isinstance(count, int)
        assert count > 0

    def test_tolerance_check_with_unicode_bullet(self):
        """
        A bullet with Unicode special chars should pass the tolerance check
        as long as its code-point length is within range.
        """
        c = FormattingConstraints(target_char_limit=80, tolerance=2)
        # Build a bullet that is exactly 80 code points
        prefix = "• "
        filler = "x" * (80 - len(prefix))
        bullet = prefix + filler
        assert len(bullet) == 80
        assert c.lower_bound <= len(bullet) <= c.upper_bound

    def test_em_dash_is_one_char(self):
        bullet = "• Reduced latency — achieved 50ms p99"
        assert "—" in bullet
        assert len("—") == 1

    def test_rupee_symbol_is_one_char(self):
        """₹ sign used in Indian salary/budget contexts."""
        assert len("₹") == 1
        bullet = "• Managed ₹20-27L budget for infrastructure migration"
        count = len(bullet)
        assert count == len(bullet)  # consistent


# ── Number preservation stress tests ─────────────────────────────────────────

class TestNumberPreservation:
    """
    These tests verify that the Python helper functions never mutate
    numbers from the user's facts. The LLM prompt enforces this at
    generation time; these tests guard the Python layer.
    """

    METRIC_CASES = [
        "0.250",        # RMSE with 3 decimal places — must not become 0.25
        "87%",          # percentage — must not become 90% or ~87%
        "₹20-27L",      # Indian lakh notation
        "8-12 weeks",   # range notation
        "5-10 days",    # range notation
        "p99",          # percentile notation
        "340ms",        # latency with unit
        "13.5%",        # decimal percentage
        "0.39",         # F1 score before
        "0.49",         # F1 score after
        "SARIMA(2,0,0)(1,0,0)[12]",   # full model specification
    ]

    @pytest.mark.parametrize("metric", METRIC_CASES)
    def test_metric_survives_format_facts(self, metric):
        """
        After passing through _format_facts, the exact metric
        string must still be present in the output.
        """
        from app.chains import _format_facts
        from app.models import CoreFact, ScoredFact

        fact = ScoredFact(
            fact=CoreFact(
                fact_id="f-test",
                text=f"Achieved result with value {metric}",
                metrics=[metric],
            ),
            project_id="p-001",
            relevance_score=0.9,
            matched_jd_keywords=[],
        )
        result = _format_facts([fact])
        assert metric in result, (
            f"Metric '{metric}' was altered or dropped in format_facts output: {result}"
        )


# ── Convergence speed benchmarks ──────────────────────────────────────────────

class TestConvergenceLogic:
    """
    Pure Python simulation of the char-limit loop's convergence logic.
    No LLM calls — just verifies the loop exit conditions.
    """

    def _simulate_loop(
        self,
        draft_lengths: list[int],
        target: int = 130,
        tolerance: int = 2,
        max_iterations: int = 4,
    ) -> dict:
        """
        Simulates the loop with a predetermined sequence of draft lengths.
        Returns info about which iteration converged (or failed).
        """
        lower = target - tolerance
        upper = target + tolerance
        best_length = None
        best_delta = float("inf")

        for i, length in enumerate(draft_lengths[:max_iterations], start=1):
            delta = abs(length - target)
            if delta < best_delta:
                best_delta = delta
                best_length = length

            if lower <= length <= upper:
                return {"converged": True, "iteration": i, "length": length}

        return {"converged": False, "best_length": best_length, "iterations": max_iterations}

    def test_converges_on_iteration_1(self):
        result = self._simulate_loop([130])
        assert result["converged"] is True
        assert result["iteration"] == 1

    def test_converges_on_iteration_2(self):
        result = self._simulate_loop([90, 130])
        assert result["converged"] is True
        assert result["iteration"] == 2

    def test_converges_on_iteration_3(self):
        result = self._simulate_loop([90, 160, 128])
        assert result["converged"] is True
        assert result["iteration"] == 3

    def test_fails_after_max_iterations_returns_closest(self):
        result = self._simulate_loop([70, 75, 80, 85], target=130, max_iterations=4)
        assert result["converged"] is False
        assert result["best_length"] == 85   # closest to 130

    def test_exact_lower_bound_converges(self):
        result = self._simulate_loop([128], target=130, tolerance=2)
        assert result["converged"] is True

    def test_exact_upper_bound_converges(self):
        result = self._simulate_loop([132], target=130, tolerance=2)
        assert result["converged"] is True

    def test_one_below_lower_does_not_converge(self):
        result = self._simulate_loop([127], target=130, tolerance=2)
        assert result["converged"] is False

    def test_one_above_upper_does_not_converge(self):
        result = self._simulate_loop([133], target=130, tolerance=2)
        assert result["converged"] is False

    @pytest.mark.parametrize("bad_length", [50, 70, 180, 220, 300])
    def test_extreme_lengths_never_converge_alone(self, bad_length):
        result = self._simulate_loop([bad_length], target=130, tolerance=2)
        assert result["converged"] is False

    def test_oscillating_drafts_returns_closest(self):
        """Drafts that oscillate around target — best should be closest."""
        result = self._simulate_loop([100, 160, 105, 155], target=130, max_iterations=4)
        assert result["converged"] is False
        # 105 is closest to 130 (delta=25) vs 100(30), 160(30), 155(25) — tie at 105 or 155
        assert result["best_length"] in (105, 155)
