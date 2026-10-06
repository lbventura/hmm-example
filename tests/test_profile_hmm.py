"""Tests for ProfileHMM."""

from io import StringIO

import numpy as np

from amino_acids import AA_INDEX, BACKGROUND_FREQ
from msa import match_column_mask, parse_stockholm
from protein_profile_hmm import (
    NEG_INF,
    T_DD,
    T_DM,
    T_II,
    T_IM,
    T_MD,
    T_MI,
    T_MM,
    ProfileHMM,
    log2sumexp2,
)


def _build(rows: list[str], **kwargs: float) -> ProfileHMM:
    """Build a ``ProfileHMM`` from an inline Stockholm MSA over ``rows``."""
    body = "\n".join(f"s{i + 1} {r}" for i, r in enumerate(rows))
    data = f"# STOCKHOLM 1.0\n{body}\n//\n"
    msa = parse_stockholm(StringIO(data))
    return ProfileHMM.build_from_msa(msa, **kwargs)


def test_log2sumexp2_matches_direct_for_finite_values() -> None:
    assert abs(log2sumexp2([1.0, 2.0, 3.0]) - np.log2(14.0)) < 1e-9


def test_log2sumexp2_handles_neg_inf_mixed() -> None:
    assert abs(log2sumexp2([NEG_INF, 5.0, NEG_INF]) - 5.0) < 1e-9


def test_log2sumexp2_all_neg_inf_returns_neg_inf() -> None:
    assert log2sumexp2([NEG_INF, NEG_INF]) == NEG_INF


def test_log2sumexp2_empty_returns_neg_inf() -> None:
    assert log2sumexp2([]) == NEG_INF


def test_build_from_msa_picks_correct_number_of_match_columns() -> None:
    hmm = _build(["ACDE", "ACDE"], symfrac=0.5, pseudocount=1.0)
    assert hmm.K == 4


def test_build_from_msa_treats_low_residue_columns_as_insert() -> None:
    hmm = _build(["AC.DE", "AC.DE", "AC.DE"], symfrac=0.5, pseudocount=1.0)
    assert hmm.K == 4


def test_build_from_msa_uses_occupancy_not_identity_for_match_columns() -> None:
    """Different residues still share match positions when no rows have gaps."""
    rows = [
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        "DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD",
        "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
    ]

    assert match_column_mask(rows).tolist() == [True] * 44
    hmm = _build(rows, symfrac=0.5, pseudocount=1.0)
    assert hmm.K == 44

    # All paths are B -> M1 -> ... -> M44, emitting A, D, or C throughout.
    # Each match state observes A=1, D=1, C=1. Adding one pseudocount
    # per amino acid gives 23 total counts: 2 for A/C/D, 1 for each other.
    expected_match = np.full(20, 1 / 23)
    expected_match[[AA_INDEX[aa] for aa in "ACD"]] = 2 / 23
    np.testing.assert_allclose(2**hmm.log_match_emit, np.tile(expected_match, (44, 1)))

    # No insert residues were observed; all 45 slots retain the background.
    np.testing.assert_allclose(2**hmm.log_insert_emit, np.tile(BACKGROUND_FREQ, (45, 1)))
    # Each internal match state observes 3 M->M and no M->I or M->D.
    # With pseudocounts, outgoing counts are [4, 1, 1], summing to 6.
    np.testing.assert_allclose(
        2 ** hmm.log_trans[1:44, [T_MM, T_MI, T_MD]],
        np.tile([4 / 6, 1 / 6, 1 / 6], (43, 1)),
    )


def test_build_from_msa_distinguishes_insert_columns_from_deleted_positions() -> None:
    """A gap skips a match position, but does nothing in an insert column."""
    rows = [
        "AAAAAAAAAAAAAAAAAA.AAAAAAAAA.AAAAAAAAAA.AAAA",
        "DDD.DDDDDDDDDDDDDD.DDDDDDDDDDDDDDDDDDDD.DDDD",
        "CCC.CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC.CCCC",
    ]
    # Alignment columns (1-based):
    #   column       4       19       29       40
    #   residues    A..     ..C      .DC      ...
    #   occupancy   1/3     1/3      2/3      0/3
    #   assignment  I3      I17      M27      I37 (empty)
    mask = match_column_mask(rows, symfrac=0.5)
    assert (np.flatnonzero(~mask) + 1).tolist() == [4, 19, 40]
    assert mask[28]  # Column 29 remains a match column despite A's gap.
    hmm = _build(rows, symfrac=0.5, pseudocount=1.0)
    assert hmm.K == 41

    # A: B -> ... -> M3(A) -> I3(A) -> ... -> M26(A) -> D27 -> M28(A) -> ...
    # D: B -> M1(D) -> ... -> M41(D), ignoring gaps in insert columns.
    # C: B -> ... -> M17(C) -> I17(C) -> M18(C) -> ... -> M41(C).
    # M27 observes only C and D: 2 observations + 20 pseudocounts = 22.
    # Match emission arrays use index k-1, so M27 is at index 26.
    expected_match = np.full(20, 1 / 22)
    expected_match[[AA_INDEX[aa] for aa in "CD"]] = 2 / 22
    np.testing.assert_allclose(2 ** hmm.log_match_emit[26], expected_match)

    # Insert slots start with 20 background-weighted counts. I3 observes
    # one A and I17 one C, so each now has 21 total counts.
    for slot, residue in [(3, "A"), (17, "C")]:
        expected_insert = 20 * BACKGROUND_FREQ.copy()
        expected_insert[AA_INDEX[residue]] += 1
        np.testing.assert_allclose(2 ** hmm.log_insert_emit[slot], expected_insert / 21)
        # Two rows go straight to the next match; one takes the insert.
        np.testing.assert_allclose(
            2 ** hmm.log_trans[slot, [T_MM, T_MI, T_MD]], [3 / 6, 2 / 6, 1 / 6]
        )
        np.testing.assert_allclose(2 ** hmm.log_trans[slot, [T_IM, T_II]], [2 / 3, 1 / 3])

    # At M26, two rows continue to M27; A takes D27, then returns to M28.
    # Transition arrays use source position k directly, without subtracting 1.
    np.testing.assert_allclose(2 ** hmm.log_trans[26, [T_MM, T_MI, T_MD]], [3 / 6, 1 / 6, 2 / 6])
    np.testing.assert_allclose(2 ** hmm.log_trans[27, [T_DM, T_DD]], [2 / 3, 1 / 3])

    # Column 40 contains only gaps: no emissions or transitions are counted.
    # Removing that column entirely leaves the learned profile unchanged.
    without_empty_column = _build([row[:39] + row[40:] for row in rows])
    assert without_empty_column.K == hmm.K
    np.testing.assert_allclose(hmm.log_match_emit, without_empty_column.log_match_emit)
    np.testing.assert_allclose(hmm.log_insert_emit, without_empty_column.log_insert_emit)
    np.testing.assert_allclose(hmm.log_trans, without_empty_column.log_trans)


def test_emission_rows_are_normalised_probabilities_in_log_space() -> None:
    hmm = _build(["ACDE", "ACDE"], symfrac=0.5, pseudocount=1.0)
    for k in range(hmm.K):
        probs = 2.0 ** hmm.log_match_emit[k]
        assert abs(probs.sum() - 1.0) < 1e-9
    for k in range(hmm.K + 1):
        probs = 2.0 ** hmm.log_insert_emit[k]
        assert abs(probs.sum() - 1.0) < 1e-6


def test_transition_distributions_normalised_per_source_state() -> None:
    hmm = _build(["ACDE", "ACDE"], symfrac=0.5, pseudocount=1.0)
    for k in range(1, hmm.K):
        m_sum = sum(2.0 ** hmm.log_trans[k, t] for t in (T_MM, T_MI, T_MD))
        i_sum = sum(2.0 ** hmm.log_trans[k, t] for t in (T_IM, T_II))
        d_sum = sum(2.0 ** hmm.log_trans[k, t] for t in (T_DM, T_DD))
        assert abs(m_sum - 1.0) < 1e-6
        assert abs(i_sum - 1.0) < 1e-6
        assert abs(d_sum - 1.0) < 1e-6


def test_viterbi_on_training_sequence_is_positive() -> None:
    hmm = _build(["ACDEFGHIK"] * 3)
    score = hmm.viterbi("ACDEFGHIK")
    assert score > 5.0


def test_viterbi_on_random_sequence_is_lower() -> None:
    hmm = _build(["ACDEFGHIK"] * 3)
    good = hmm.viterbi("ACDEFGHIK")
    bad = hmm.viterbi("WWWWWWWWW")
    assert good > bad


def test_viterbi_score_is_finite() -> None:
    hmm = _build(["ACDEFGHIK"] * 2)
    score = hmm.viterbi("ACDE")
    assert np.isfinite(score)


def test_forward_geq_viterbi_on_training_sequence() -> None:
    hmm = _build(["ACDEFGHIK"] * 3)
    v = hmm.viterbi("ACDEFGHIK")
    f = hmm.forward("ACDEFGHIK")
    assert f >= v - 1e-6


def test_forward_geq_viterbi_on_unrelated_sequence() -> None:
    hmm = _build(["ACDEFGHIK"] * 2)
    v = hmm.viterbi("WWWWWWWWW")
    f = hmm.forward("WWWWWWWWW")
    assert f >= v - 1e-6
