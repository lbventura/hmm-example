"""Tests for the amino-acid alphabet and background frequencies."""

from amino_acids import (
    AA_ALPHABET,
    AA_INDEX,
    BACKGROUND_FREQ,
    encode,
    is_residue,
)


def test_alphabet_has_20_canonical_residues() -> None:
    assert len(AA_ALPHABET) == 20
    assert set(AA_ALPHABET) == set("ACDEFGHIKLMNPQRSTVWY")


def test_index_is_inverse_of_alphabet() -> None:
    for i, aa in enumerate(AA_ALPHABET):
        assert AA_INDEX[aa] == i


def test_background_sums_to_one() -> None:
    assert abs(BACKGROUND_FREQ.sum() - 1.0) < 1e-9


def test_background_has_20_entries() -> None:
    assert BACKGROUND_FREQ.shape == (20,)


def test_background_all_positive() -> None:
    assert (BACKGROUND_FREQ > 0).all()


def test_encode_handles_canonical_residues() -> None:
    assert encode("A") == AA_INDEX["A"]
    assert encode("y") == AA_INDEX["Y"]


def test_encode_returns_minus_one_for_unknown_residue() -> None:
    assert encode("X") == -1
    assert encode("-") == -1
    assert encode(".") == -1


def test_is_residue_filters_alignment_chars() -> None:
    assert is_residue("A") is True
    assert is_residue("y") is True
    assert is_residue("-") is False
    assert is_residue(".") is False
    assert is_residue("X") is False
