# Contributing

## Development setup

```bash
git clone https://github.com/helia77/MicroVasc-Review.git
cd MicroVasc-Review
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test,kline]"
```

The `kline` extra installs scikit-fmm, which pip builds from source on Linux. That needs a C++
compiler and the Python headers. Use `pip install -e ".[test]"` if you do not need Kline.

## Tests

```bash
pytest -ra
```

Without scikit-fmm the Kline tests are skipped, and `-ra` lists them. CI runs the tests on Python
3.12 and 3.14, with and without scikit-fmm.

`tests/data/` holds reference outputs. If a change breaks one of those tests, find out why before
you rewrite the file with the matching `tests/capture_*.py --force` script, and say so in the pull
request.

## Documentation

The site at https://www.helia-gohar.com/MicroVasc-Review/ is built with MkDocs from `docs/` and
`mkdocs.yml`. To preview it locally:

```bash
pip install -r docs/requirements.txt
mkdocs serve
```

Then open http://127.0.0.1:8000/MicroVasc-Review/ (the path comes from `site_url` in
`mkdocs.yml`). CI runs `mkdocs build --strict`, which fails on warnings such as broken links, and
deploys the site from `main`.

## Commit messages

One line: lowercase tags in brackets, then a capitalized summary in the imperative mood, at most
50 characters after the tags, no final period.

```
[opt] Vectorize Frangi and Beyond Frangi voxel loops
[fix, add] Fix optimization scripts and document the workflow
```

Tags: `fix`, `add`, `del` (delete code or files), `mod` (rename or move), `opt` (refactor or
optimize), `clean`, `install` (add a dependency), `remove` (remove a dependency).

## Reporting bugs

Open an issue at https://github.com/helia77/MicroVasc-Review/issues. Include the microvasc version
(`python -c "import microvasc; print(microvasc.__version__)"`), your Python version and OS, the
command or code you ran, and the full traceback. A small input that shows the problem helps most;
a synthetic volume is fine.
