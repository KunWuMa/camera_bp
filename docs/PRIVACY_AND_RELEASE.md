# Privacy and GitHub release checklist

## Automated gate

Run from the repository root:

```bash
python scripts/check_release.py
```

The command fails when it finds raw/derived data extensions, model checkpoints,
large binary files, files under `data/` other than README documents, local config
files, likely Windows absolute paths, or common secret-file names.

## Manual checks before the first push

- Confirm `git status --short` contains source, configuration examples, and
  documentation only.
- Confirm `git check-ignore -v` excludes the locally downloaded HDF5 file and all
  private data.
- Search for names, hospital numbers, student numbers, dates of birth, phone
  numbers, e-mail addresses, local drive paths, and identifier lookup tables.
- Do not commit `artifacts/`, including hashed manifests and per-subject errors.
- Do not commit `BP_ID_SALT`, shell history, `.env`, or `config/local.json`.
- Confirm screenshots and figures do not contain recognizable participant faces.
- Obtain author approval before releasing aggregate private-cohort results.
- Add the approved author list, repository URL/DOI, citation, and project license.
- Tag the exact paper release and record the commit hash in the manuscript.

## Recommended Git commands

```bash
git init
git add .gitignore README.md README_zh-CN.md environment.yml config src figures scripts docs data artifacts/README.md
git status --short
python scripts/check_release.py
git diff --cached --stat
```

Review the staged files before committing. The privacy checker is a safeguard,
not a replacement for institutional disclosure review.
