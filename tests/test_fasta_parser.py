"""Tests for the minimal FASTA parser."""

from io import StringIO

from protein_profile_hmm.fasta_parser import parse_fasta


def test_parse_single_sequence() -> None:
    data = ">seq1\nMVLSPADKT\nNVKAAWGKV\n"
    records = list(parse_fasta(StringIO(data)))
    assert records == [("seq1", "MVLSPADKTNVKAAWGKV")]


def test_parse_multiple_sequences() -> None:
    data = ">a\nACDE\n>b desc\nFGHI\nKLMN\n"
    records = list(parse_fasta(StringIO(data)))
    assert records == [("a", "ACDE"), ("b", "FGHIKLMN")]


def test_parse_strips_whitespace_and_uppercases() -> None:
    data = ">x\n  acde  \n  fghi\n"
    records = list(parse_fasta(StringIO(data)))
    assert records == [("x", "ACDEFGHI")]


def test_parse_skips_blank_lines() -> None:
    data = "\n>seq1\nACDE\n\nFGHI\n\n>seq2\n\nKLMN\n"
    records = list(parse_fasta(StringIO(data)))
    assert records == [("seq1", "ACDEFGHI"), ("seq2", "KLMN")]


def test_parse_uses_first_token_as_name() -> None:
    data = ">P68871 HBB_HUMAN hemoglobin subunit beta\nMVHLT\n"
    records = list(parse_fasta(StringIO(data)))
    assert records == [("P68871", "MVHLT")]


def test_parse_empty_input_yields_no_records() -> None:
    records = list(parse_fasta(StringIO("")))
    assert records == []
