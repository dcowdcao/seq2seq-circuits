# Vendored reference repos

These third-party repos were cloned, lightly edited for the segrev3 project,
and then vendored (their nested .git removed) so the project is a self-contained
snapshot. To restore a live clone, re-clone the upstream at the pinned commit and
re-apply the project edits noted below.

The vendored copies are TRIMMED to what this project imports: tests, notebooks,
experiment scripts, figures and other upstream assets were removed. Each tree keeps
its upstream LICENSE, README and pyproject.toml.

## Automatic-Circuit-Discovery
- upstream: https://github.com/ArthurConmy/Automatic-Circuit-Discovery.git
- pinned commit: bc99ace817974b5584b7ee203d596a8e2bbcd399
- branch: main
- kept: the `acdc/` package (core + the upstream task packages), LICENSE, README.md,
  pyproject.toml, .gitignore, .gitattributes
- local edits vs that commit:
     M acdc/__init__.py      (TransformerLens 3.3.0 import path)
    ?? acdc/segrev3/          (project adapter: segrev3 model + data + KL metric)

## EAP-IG
- upstream: https://github.com/hannamw/EAP-IG.git
- pinned commit: e24bafd3d22af2c59ac5a83e0a6581b89b5c05dd
- branch: main
- kept: `src/eap/`, LICENSE, README.md, pyproject.toml
- local edits vs that commit: none

## Edge-Pruning (single file)
- upstream: https://github.com/princeton-nlp/Edge-Pruning (MIT license)
- only `src/modeling/l0.py` is used; it is vendored verbatim, with the upstream MIT
  license notice prepended, at `pipeline/find_circuits/l0.py` (imported by
  `pipeline/find_circuits/edge_pruning.py`). The rest of the upstream repo is not
  vendored; `edge_pruning.py` ports the method onto TransformerLens models.
