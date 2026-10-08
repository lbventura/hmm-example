"""Protein profile-HMM demo.

Builds a profile from the 4-sequence globin MSA, then scores:
  - HBB_HUMAN (positive control, one of the training sequences)
  - the 45 globins in globins45.fa (positive eval set)
  - 7LESS_DROME (negative control: a tyrosine kinase, not a globin)

Prints a ranked table of Viterbi and Forward bit scores.
"""

import logging
from pathlib import Path

from .fasta_parser import parse_fasta, read_first
from .msa import parse_stockholm
from .protein_profile_hmm import ProfileHMM

DATA = Path(__file__).resolve().parent.parent / "data"
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("  Protein Profile HMM Demo  (HMMER tutorial datasets)")

    logger.info(f"\nBuilding profile from {DATA / 'globins4.sto'} ...")
    with open(DATA / "globins4.sto") as f:
        msa = parse_stockholm(f)
    hmm = ProfileHMM.build_from_msa(msa)
    logger.info(f"  K = {hmm.K} match columns")
    logger.info(f"  trained on {msa.n_seqs} sequences")

    queries: list[tuple[str, str, str]] = []  # (sign, label, seq)

    name, seq = read_first(DATA / "HBB_HUMAN")
    queries.append(("+", f"{name} (HBB_HUMAN file)", seq))

    with open(DATA / "globins45.fa") as f:
        for n, s in parse_fasta(f):
            queries.append(("+", n, s))

    name, seq = read_first(DATA / "7LESS_DROME.fa")
    queries.append(("-", f"{name} (non-globin, tyrosine kinase)", seq))

    logger.info(f"\nScoring {len(queries)} sequences ...\n")
    rows = []
    for sign, label, seq in queries:
        v = hmm.viterbi(seq)
        f_ = hmm.forward(seq)
        rows.append((sign, label, len(seq), v, f_))

    rows.sort(key=lambda r: r[3], reverse=True)
    logger.info(f"  {'':1s} {'Sequence':<48s} {'len':>5s}  {'Viterbi':>10s}  {'Forward':>10s}")
    for sign, label, L, v, f_ in rows:
        logger.info(f"  {sign} {label[:48]:<48s} {L:>5d}  {v:>10.2f}  {f_:>10.2f}")

    pos_v = [r[3] for r in rows if r[0] == "+"]
    neg_v = [r[3] for r in rows if r[0] == "-"]
    if pos_v and neg_v:
        pos_v.sort()
        median = pos_v[len(pos_v) // 2]
        logger.info(f"  Positive Viterbi median: {median:8.2f} bits")
        logger.info(f"  Negative control      : {neg_v[0]:8.2f} bits")
        logger.info(f"  Margin                : {median - neg_v[0]:8.2f} bits")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
