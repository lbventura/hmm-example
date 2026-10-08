# Protein profile HMM

## Modules and execution

- `protein_profile_hmm.py`: `ProfileHMM.build_from_msa` constructs a profile using counts and pseudocount smoothing; `viterbi` scores the best alignment path, while `forward` sums over allowed paths.
- `amino_acids.py`: canonical amino-acid alphabet, residue encoding, and background frequencies used in scoring.
- `msa.py`: Stockholm alignment parsing and selection of consensus columns using `symfrac`, the minimum fraction of canonical residues in a column. This measures residue occupancy, not agreement on a single amino acid.
- `fasta_parser.py`: reads FASTA records as names and protein sequences.
- `protein_demo.py`: builds a globin profile and compares scores for globins and an unrelated tyrosine kinase.
- `download_data.py`: fetches HMMER tutorial datasets and converts the Swiss-Prot negative control to FASTA.

After following the [development setup](../README.md#set-up-a-development-environment), run from the repository root:

```bash
python -m protein_profile_hmm.protein_demo
python -m pytest tests/test_profile_hmm.py tests/test_globins_integration.py
```

To download fresh copies of the [datasets](../data/DATASETS.md), overwriting the existing files:

```bash
python -m protein_profile_hmm.download_data
```

## Profile architecture

Each of the `K` consensus positions has a Match state (`M_k`) and a silent Delete state (`D_k`). Insert states (`I_k`) represent extra residues between consensus positions, including boundary slots. The Begin state (`B`) is silent.

- **Match:** consumes a residue aligned to consensus position `k`; the residue need not equal the most common amino acid there.
- **Insert:** consumes an extra residue without advancing the consensus position.
- **Delete:** advances past a consensus position without consuming a residue.

The transition array stores nine types per slot:

| Index | Transition | Meaning |
| --- | --- | --- |
| 0 | M_k → M_{k+1} | Advance to the next match |
| 1 | M_k → I_k | Start an insertion |
| 2 | M_k → D_{k+1} | Start a deletion |
| 3 | I_k → M_{k+1} | End an insertion |
| 4 | I_k → I_k | Extend an insertion |
| 5 | D_k → M_{k+1} | End a deletion |
| 6 | D_k → D_{k+1} | Extend a deletion |
| 7 | B → M_1 | Begin at the first match (slot 0 only) |
| 8 | B → D_1 | Begin by deleting the first position (slot 0 only) |

Scoring spans the full profile, allowing deletions, and consumes the query residues; boundary Insert states accommodate flanking residues. Emission and transition probabilities are stored in log2 space. Scores are log-odds in bits relative to a background model: positive scores favor the profile over that background, but are not calibrated significance estimates or HMMER E-values.
