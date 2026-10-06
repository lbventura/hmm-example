"""End-to-end test: profile HMM built from globins4.sto should rank globin
sequences higher than the non-globin 7LESS_DROME tyrosine kinase."""

from pathlib import Path

import pytest

from fasta_parser import parse_fasta, read_first
from msa import parse_stockholm
from protein_profile_hmm import ProfileHMM

DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def globin_profile() -> ProfileHMM:
    with open(DATA / "globins4.sto") as f:
        msa = parse_stockholm(f)
    return ProfileHMM.build_from_msa(msa)


def test_HBB_HUMAN_scores_positively(globin_profile: ProfileHMM) -> None:
    _, seq = read_first(DATA / "HBB_HUMAN")
    score = globin_profile.viterbi(seq)
    assert score > 0, f"HBB_HUMAN Viterbi score should be positive, got {score:.2f}"


def test_7LESS_DROME_scores_lower_than_HBB_HUMAN(globin_profile: ProfileHMM) -> None:
    _, hbb = read_first(DATA / "HBB_HUMAN")
    _, seven = read_first(DATA / "7LESS_DROME.fa")
    pos = globin_profile.viterbi(hbb)
    neg = globin_profile.viterbi(seven)
    assert pos > neg, f"globin ({pos:.2f}) should outscore non-globin ({neg:.2f})"


def test_globin_family_median_outscores_negative_control(globin_profile: ProfileHMM) -> None:
    with open(DATA / "globins45.fa") as f:
        globins = list(parse_fasta(f))
    _, neg_seq = read_first(DATA / "7LESS_DROME.fa")
    neg_score = globin_profile.viterbi(neg_seq)
    pos_scores = [globin_profile.viterbi(seq) for _, seq in globins]
    median = sorted(pos_scores)[len(pos_scores) // 2]
    assert median > neg_score, (
        f"median globin score {median:.2f} should exceed 7LESS_DROME {neg_score:.2f}"
    )
