"""Download HMMER tutorial datasets used for protein HMM testing.

The HMMER project (https://github.com/EddyRivasLab/hmmer) ships a `tutorial/`
directory designed for exactly this use case: small, well-known protein
sequence files that work as a positive set (globins) plus a clear negative
control (7LESS_DROME, a tyrosine kinase).

Note: HMMER's `7LESS_DROME` ships as a Swiss-Prot flat-file. We convert it to
plain FASTA on download so downstream tools see a single uniform format.
"""

import logging
import re
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BASE_URL = "https://raw.githubusercontent.com/EddyRivasLab/hmmer/master/tutorial"
logger = logging.getLogger(__name__)

FILES = [
    ("globins4.sto", "MSA used to BUILD the profile (4 globins, Stockholm)"),
    ("globins45.fa", "Positive test set: 45 globin sequences (FASTA)"),
    ("HBB_HUMAN", "Positive control: human beta-hemoglobin (FASTA)"),
    ("7LESS_DROME", "Negative control: D. melanogaster sevenless (Swiss-Prot)"),
]


def download_one(filename: str) -> Path:
    url = f"{BASE_URL}/{filename}"
    dest = DATA_DIR / filename
    logger.info(f"  fetching {url}")
    with urllib.request.urlopen(url, timeout=30) as resp:
        dest.write_bytes(resp.read())
    logger.info(f"    saved -> {dest}  ({dest.stat().st_size} bytes)")
    return dest


def _swissprot_to_fasta(sp_path: Path, fa_path: Path) -> None:
    """Extract the sequence from a Swiss-Prot flat-file and write FASTA.

    Swiss-Prot files have an `ID` line at the top (with the entry name) and a
    `SQ` block of indented residue lines at the end terminated by `//`.
    """
    text = sp_path.read_text()
    # Entry name (e.g., "7LESS_DROME") from the ID line
    m = re.search(r"^ID\s+(\S+)", text, flags=re.MULTILINE)
    name = m.group(1) if m else sp_path.name
    # Description from the first DE line, best effort
    m_de = re.search(r"^DE\s+(.*)$", text, flags=re.MULTILINE)
    desc = m_de.group(1).strip() if m_de else ""
    # Sequence: everything between "SQ ..." line and "//" terminator,
    # stripped of whitespace and digits.
    m_sq = re.search(r"^SQ\s.*?\n(.*?)^//", text, flags=re.MULTILINE | re.DOTALL)
    if not m_sq:
        raise ValueError(f"No SQ block found in {sp_path}")
    raw_seq = m_sq.group(1)
    seq = re.sub(r"[^A-Za-z]", "", raw_seq).upper()
    fa_path.write_text(f">{name} {desc}\n{seq}\n")
    logger.info(f"    converted -> {fa_path} ({len(seq)} residues)")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading HMMER tutorial datasets into {DATA_DIR} ...")
    for name, role in FILES:
        download_one(name)
        logger.info(f"    role: {role}")

    # Convert the Swiss-Prot file to FASTA so the rest of the pipeline is uniform.
    sp = DATA_DIR / "7LESS_DROME"
    fa = DATA_DIR / "7LESS_DROME.fa"
    _swissprot_to_fasta(sp, fa)

    logger.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
