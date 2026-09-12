# Private hospital and laboratory cohorts

## Access

The private data are **not publicly downloadable**. Access to either cohort
requires prior written approval from the corresponding author and the responsible
data custodians. Approval may require a scientific proposal, institutional
affiliation, ethics approval, and a data-use agreement. A request does not imply
that access will be granted.

- Hospital cohort: collected at the First Affiliated Hospital of the University
  of Science and Technology of China, Anhui, China.
- Laboratory cohort: collected from students at Hefei University of Technology.

Contact the corresponding author through the published article or the contact
address shown on the final repository landing page. Do not open a public GitHub
issue containing participant information.

## Local-only directory layout

After authorization, arrange the governed copy as follows. These files are
ignored by Git.

```text
data/private/
|-- hospital/
|   |-- metadata.xlsx
|   `-- <face and fingertip video subdirectories>/
`-- laboratory/
    |-- labels.csv
    |-- face/
    `-- finger/
```

If the authorized data use a different layout, edit only `config/local.json`
(which is ignored) or the `private_layout` block in a private configuration.

## Input schema expected by the released parser

The hospital workbook uses its first worksheet. The released parser expects the
following zero-based columns: age 0, sex 1, raw subject identifier 3, BP string
(`SBP/DBP`) 4, HR 5, acquisition date 6, height 7, and weight 8. Hospital video
stems must contain a numeric subject identifier and an ISO-like date; `face` and
`finger` identify modality.

The laboratory CSV stores one acquisition per column. Column names follow
`<subject>_<YYYY-M-D>_<session>`, and rows 0--2 contain SBP, DBP, and HR. Face
videos use the column name plus `.mp4`; fingertip waveforms use `.txt`.

Raw identifiers are converted to secret-salted SHA-256 pseudonyms. Set
`BP_ID_SALT` in the shell before running `build_manifest.py`. Never publish the
salt, raw identifiers, pseudonym lookup table, local paths, or generated manifest.

## Reference protocol represented by the labels

Participants were seated with the left arm supported at heart level. Hospital
participants rested for at least 30 min before measurement. BP was measured with
an Omron A862 electronic upper-arm monitor immediately before and after the
synchronous video--PPG acquisition. Acquisition-level SBP and DBP are the
respective two-reading arithmetic means. When either paired SBP or DBP differed
by more than 10 mmHg, the complete acquisition was discarded and repeated.

This documentation describes the study snapshot used by the paper; it does not
waive ethics, consent, or institutional restrictions.
