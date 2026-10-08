# Datasets

All datasets in this directory are downloaded from the **HMMER** project's
`tutorial/` directory. HMMER ships these files specifically as small,
well-known examples for learning profile-HMM concepts, which makes them an
ideal fit for testing this educational re-implementation.

- Project: <https://github.com/EddyRivasLab/hmmer>
- Tutorial source path: `tutorial/` on the `master` branch
- Raw URL pattern: `https://raw.githubusercontent.com/EddyRivasLab/hmmer/master/tutorial/<FILE>`
- License: HMMER is distributed under the BSD-3-Clause license (see the
  HMMER repository's `LICENSE` file). The small example sequences come from
  public sources (Swiss-Prot / Pfam seed) and are widely redistributed.
- Fetcher: run `python -m protein_profile_hmm.download_data` from the repository root to (re)download.

## Files

### `globins4.sto`  —  build set
- **Format:** Stockholm 1.0 multiple sequence alignment
- **Contents:** 4 globin proteins — HBB_HUMAN, HBA_HUMAN, MYG_PHYCA, GLB5_PETMA
- **Size:** 863 bytes, ~165 aligned columns
- **Role:** input to `ProfileHMM.build_from_msa(...)`. Defines the consensus
  columns (match states) and the family's position-specific amino-acid
  preferences.
- **Why this file:** it's the smallest, simplest MSA in the HMMER tutorial and
  models a single, well-conserved domain (the globin fold) that is easy to
  visualise and verify.

### `globins45.fa`  —  positive evaluation set
- **Format:** plain (unaligned) FASTA
- **Contents:** 45 globin sequences from across species
- **Size:** 7210 bytes
- **Role:** every sequence here should score **positively** (higher than the
  negative control) against a profile built from `globins4.sto`. Demonstrates
  that the profile generalises beyond the 4 sequences it was trained on.

### `HBB_HUMAN`  —  positive control
- **Format:** plain FASTA, one sequence
- **Contents:** human β-hemoglobin (one of the 4 training sequences)
- **Size:** 183 bytes
- **Role:** the smallest possible positive smoke-test — the profile must
  score it strongly because it was part of the training MSA.

### `7LESS_DROME`  —  negative control (Swiss-Prot)
- **Format:** Swiss-Prot / UniProtKB flat-file (NOT FASTA)
- **Contents:** *Drosophila melanogaster* "sevenless" — a 2554-residue
  receptor tyrosine kinase, totally unrelated to globins.
- **Size:** 19087 bytes
- **Role:** must score noticeably **lower** than any globin. Serves as the
  family-discrimination test in the test suite.

### `7LESS_DROME.fa`  —  negative control (converted)
- **Format:** plain FASTA
- **Contents:** the 2554-residue sequence extracted from `7LESS_DROME` by
  `protein_profile_hmm/download_data.py`.
- **Role:** lets the rest of the pipeline (parsers, scorers, demo) work
  exclusively with FASTA so we don't carry a Swiss-Prot parser through the
  whole codebase.

## Why these files vs Pfam directly?

The InterPro/Pfam ecosystem has a more authoritative globin alignment
(PF00042) but URL stability has changed across the EBI → InterPro migration.
The HMMER tutorial files are guaranteed-stable, version-controlled, BSD
licensed, format-compatible, and are the canonical examples in the HMMER
documentation. For an educational re-implementation that targets HMMER as the
reference, they are the natural choice.
