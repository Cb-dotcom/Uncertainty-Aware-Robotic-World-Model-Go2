# MkDocs configuration snippet

The configuration below is what `mkdocs.yml` should contain at minimum. It includes the math extension required to render the equations in `paper-a-loss-map.md`, plus a small set of standard Markdown extensions that improve code-block and admonition rendering.

```yaml
site_name: Uncertainty-Aware Robotic World Model — Go2

theme:
  name: material
  features:
    - navigation.sections
    - navigation.expand
    - content.code.copy
    - search.highlight

nav:
  - Home: index.md
  - Baseline: baseline.md
  - Repo Map: repo-map.md
  - Paper A:
      - Task Modes: paper-a-task-modes.md
      - Runtime Flow: paper-a-runtime-flow.md
      - Loss Map: paper-a-loss-map.md
  - Project meta:
      - Roadmap: roadmap.md
      - Portability: portability.md
      - Repo Strategy: repo-strategy.md

markdown_extensions:
  - admonition
  - attr_list
  - md_in_html
  - tables
  - toc:
      permalink: true
  - pymdownx.highlight:
      anchor_linenums: true
  - pymdownx.superfences
  - pymdownx.inlinehilite
  - pymdownx.snippets
  - pymdownx.arithmatex:
      generic: true

extra_javascript:
  - https://polyfill.io/v3/polyfill.min.js?features=es6
  - https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js
```

## Notes

- `pymdownx.arithmatex` plus the two `extra_javascript` entries are what makes `$...$` and `$$...$$` math render. Drop them if math support is not needed and rewrite the equations in `paper-a-loss-map.md` as plain text.
- `navigation.sections` keeps top-level groups like *Paper A* and *Project meta* expanded in the sidebar, which is the behavior expected for documentation of this size.
- Add a `paper-b-*` section under `nav` once Paper B documentation begins.
- Required dependencies: `mkdocs-material`. Install with `pip install mkdocs-material`.
