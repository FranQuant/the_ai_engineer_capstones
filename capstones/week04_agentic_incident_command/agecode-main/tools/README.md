# Tools (utilities for this template)

This folder bundles small, project-agnostic utilities adapted from the main
project. They help validate notebooks, normalize formatting, and keep assets in
sync.

- `check_notebooks.py`
  - Executes notebooks end-to-end with per-cell timeouts; reports failures.
  - Options: `--paths`, `--pattern`, `--exclude`, `--timeout`, `--kernel`,
    `--allow-errors`, `--fail-fast`, `--outdir`, `--normalize-ids`,
    `--write-normalized`, `--no-exec`.
  - Tip: use `--normalize-ids --write-normalized --no-exec` to add missing
    cell IDs without executing.

- `fix_inline_comment_spacing.py`
  - Normalizes inline comment spacing in code listings (AsciiDoc-oriented) and
    can wrap overlong IPython lines by moving trailing comments to a
    continuation line.
  - For LaTeX projects, use it only on code listings exported from AsciiDoc or
    when your code listings follow the same prompt patterns.

- `normalize_blank_lines.py`
  - Collapses multiple blank lines in AsciiDoc sources and fixes open-block
    spacing around titled blocks. For LaTeX, keep it for reference or adapt it
    to your markup as needed.

- `check_assets.py`
  - Scans AsciiDoc chapters for referenced figures and code-includes, and
    checks that the files exist. Useful if you dual-maintain AsciiDoc sources.

- `create_figures.sh`
  - Convenient loop to (re)generate figures from `code/figures/*.py` into
    `figures/`.

- `svg_to_png.py`
  - Batch-convert SVGs to PNGs.

## Typical flows

- Run notebooks (with normalization but no exec):

```
python tools/check_notebooks.py --paths notebooks --normalize-ids --no-exec --write-normalized
```

- Execute all notebooks with 2 min per-cell timeout and stop on first failure:

```
python tools/check_notebooks.py --paths notebooks --timeout 120 --fail-fast
```

- Regenerate figures under `figures/`:

```
make -C .. figures
```

- Normalize inline comments in AsciiDoc sources (optional, if you keep .adoc):

```
python tools/fix_inline_comment_spacing.py --wrap --maxlen 87 path/to/chapters
```
