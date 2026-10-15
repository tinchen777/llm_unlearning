"""Dependency direction of the packages under src/ (see the module layout):

    utils  <-  data, model  <-  evals  <-  trainer ;  vis -> utils only ;  METHOD / entry points on top of everything.

`data` and `model` are the bottom layers and must not import each other, `evals`, `trainer` or `vis`.
Pure static check (ast), no model / network:  python -m pytest tests/test_layering.py -q
"""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
ALLOWED = {
    "utils": set(),
    "data": {"utils"},
    "model": {"utils"},
    "evals": {"utils", "data", "model"},
    "trainer": {"utils", "data", "model", "evals"},
    "vis": {"utils"},
}


def _imports(path: Path):
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module.split(".")[0], node.lineno
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0], node.lineno


def test_package_dependencies_point_downwards():
    violations = []
    for package, allowed in ALLOWED.items():
        for path in sorted((SRC / package).rglob("*.py")):
            for target, lineno in _imports(path):
                if target in ALLOWED and target != package and target not in allowed:
                    violations.append(f"{path.relative_to(SRC)}:{lineno}: {package} -> {target}")
    assert not violations, "\n".join(violations)
