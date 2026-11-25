# Contributing to microvasc

Thanks for helping.

## Before you start

- Bugs: open an issue at https://github.com/helia77/MicroVasc-Review/issues with the microvasc
  version (`python -c "import microvasc; print(microvasc.__version__)"`), your Python version and
  OS, a short example that shows the problem, and the full traceback. A small synthetic volume is
  fine as input.
- New features: open an issue first and post your plan (API, new dependencies, how you will split
  the work into PRs). Wait for a reply before writing code.
- Questions: open an issue.

## Setup

```bash
git clone https://github.com/helia77/MicroVasc-Review.git
cd MicroVasc-Review
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test,kline]"
```

The `kline` extra installs scikit-fmm, which pip builds from source on Linux. That needs a C++
compiler and the Python headers. Use `pip install -e ".[test]"` if you do not need Kline.

## Workflow

1. Work on a branch, never on `main`.
2. One topic per PR and one topic per commit.
3. Keep each PR at about 300 changed lines or less, not counting tests. Split bigger work into
   several PRs.
4. Change as few existing files as possible. Do not reformat or rename code you are not working
   on.
5. Keep your branch up to date by rebasing, not merging:

    ```
    git fetch origin
    git rebase origin/main
    git push --force-with-lease
    ```

    Force-push only to your own branch.

6. Open a PR against `main`. Link the issue (`resolves: #12`) and say what changed, why, and how
   you tested it.
7. The maintainer reviews every PR. Fix review comments inside the commit they belong to (fixup
   and rebase), so each commit stays one clean topic.
8. Only the maintainer merges. PRs are merged by rebase, so there are no merge commits.

## Files the maintainer manages

Do not edit these in a PR. If your change needs one of them updated (for example a new dependency
in `pyproject.toml` or a new page in the docs nav), say so in the PR description and the
maintainer will do it.

- `pyproject.toml`, `MANIFEST.in`
- `CITATION.cff`, `LICENSE`, `THIRD_PARTY_NOTICES.md`, `README.md`
- `mkdocs.yml`, `docs/requirements.txt`, `docs/index.md`, `docs/citation.md`
- everything under `.github/`
- the reference files in `tests/data/` and the Palagyi table `microvasc/PK12.npz`
- version numbers, tags and releases

`CHANGELOG.md`: add one line under Unreleased in the right section (Added, Changed, Fixed). Do not
edit other lines.

Changes to existing modules (`binarize.py`, `_oof.py`, `skeleton.py`, `data.py`, `metrics/`,
`optimization/`) need their own issue first.

## Commit messages

```
[tag] Short title

Optional body: why the change is needed and what it achieves.
resolves: #12
```

- Lowercase tags in brackets. Several tags are fine: `[add, fix]`.
- Title: capital letter, imperative ("Add", not "Added"), 50 characters or less after the tags,
  no final period.
- Optional body after a blank line, wrapped at 72 characters. Say why, not how.
- No trailers (`Co-authored-by`, `Signed-off-by` and similar).

| Tag | Use for |
| --- | --- |
| `fix` | bug fix |
| `add` | new feature, file, test or docs page |
| `del` | deleting code or files |
| `mod` | renaming or moving only |
| `opt` | refactor, speedup or API change |
| `clean` | cleanup with no change in behavior |
| `install` | adding a dependency |
| `remove` | removing a dependency |

Examples:

```
[opt] Vectorize Frangi and Beyond Frangi voxel loops
[fix, add] Fix optimization scripts and document the workflow
```

## Code

- PEP 8 names: `CamelCase` classes, `snake_case` functions.
- Type hints on public functions and classes.
- Google style docstrings for everything public. Examples in docstrings must run.
- New public modules get an API page under `docs/api/`.
- Check inputs early and raise clear errors (`ValueError`, `TypeError`) instead of failing deep
  inside.
- Optional dependencies (for example scikit-fmm for Kline, or PyTorch) are imported only where
  they are needed, so `import microvasc` works without them. If one is missing, raise an
  `ImportError` that names the extra to install.
- No large files in git. Model weights and datasets are hosted outside the repo (Hugging Face or
  Zenodo) and downloaded when needed.

## Tests

- Use `pytest`. Every new function has tests.
- A bug fix comes with a test that fails on the old code.
- Golden outputs must still match. Do not update golden files. If your change has to alter one,
  explain why in the PR.
- Tests run on CPU and stay fast. GPU tests are skipped when no GPU is found.
- Before opening a PR, run:

    ```
    pytest -ra
    mkdocs build --strict
    git diff --check
    ```

Without scikit-fmm the Kline tests are skipped, and `-ra` lists them. CI runs the tests on Python
3.12 and 3.14, with and without scikit-fmm. `mkdocs build --strict` fails on warnings such as
broken links.

## Documentation

The site at https://www.helia-gohar.com/MicroVasc-Review/ is built with MkDocs from `docs/` and
`mkdocs.yml`. To install the docs tools and preview the site locally:

```bash
pip install -r docs/requirements.txt
mkdocs serve
```

Then open http://127.0.0.1:8000/MicroVasc-Review/ (the path comes from `site_url` in
`mkdocs.yml`). CI builds the site on every push and pull request and deploys it from `main`.

## Writing style

For code comments, docs, commit messages and PR text:

- Short, plain sentences.
- Only claims you have measured. Speedups name the hardware.
- No em dashes.

## License

By contributing, you agree that your work is released under the MIT license of this project.
