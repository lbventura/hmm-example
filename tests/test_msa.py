"""Tests for the Stockholm MSA parser and match-column determination."""

from io import StringIO

from msa import match_column_mask, parse_stockholm


def test_parse_stockholm_returns_msa_with_names_and_rows() -> None:
    data = "# STOCKHOLM 1.0\nseq1   AC.DE\nseq2   A-.DE\nseq3   AC.DE\n//\n"
    msa = parse_stockholm(StringIO(data))
    assert msa.names == ["seq1", "seq2", "seq3"]
    assert msa.rows == ["AC.DE", "A-.DE", "AC.DE"]


def test_parse_stockholm_concatenates_blocked_alignments() -> None:
    data = "# STOCKHOLM 1.0\nseq1   AC\nseq2   A-\n\nseq1   DE\nseq2   DE\n//\n"
    msa = parse_stockholm(StringIO(data))
    assert msa.names == ["seq1", "seq2"]
    assert msa.rows == ["ACDE", "A-DE"]


def test_parse_stockholm_skips_comment_lines() -> None:
    data = "# STOCKHOLM 1.0\n#=GF ID Test\n#=GS seq1 DE description\nseq1   ACDE\n//\n"
    msa = parse_stockholm(StringIO(data))
    assert msa.names == ["seq1"]
    assert msa.rows == ["ACDE"]


def test_parses_real_globins4() -> None:
    path = "data/globins4.sto"
    with open(path) as f:
        msa = parse_stockholm(f)
    assert len(msa.names) == 4
    assert {len(r) for r in msa.rows} == {len(msa.rows[0])}
    assert "HBB_HUMAN" in msa.names


def test_match_column_mask_default_symfrac_half() -> None:
    rows = ["AC.DE", "A-.DE", "AC.DE"]
    mask = match_column_mask(rows, symfrac=0.5)
    assert mask.tolist() == [True, True, False, True, True]


def test_match_column_mask_with_high_symfrac() -> None:
    rows = ["AC.DE", "A-.DE", "AC.DE"]
    mask = match_column_mask(rows, symfrac=0.8)
    assert mask.tolist() == [True, False, False, True, True]
