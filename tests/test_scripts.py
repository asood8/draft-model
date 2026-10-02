"""Every script in scripts/ parses, is ASCII, and answers --help.

These exist because a script is the one kind of file in this repo that nothing else imports. The
library modules are covered many times over by the tests around them; a script can sit broken in a
commit indefinitely, and one did. `scripts/trim_vocab.py` carried an unterminated string literal from
one commit to the next, in a commit whose own message claimed every script's output had been checked,
because a stray editing step turned `print("\\nWARNING: ...` into a `print("` followed by a real
newline. Parsing it would have taken a millisecond.

ASCII is checked for the same reason it is a rule: this machine's console is cp1252, so a gamma or an
alpha in a print raises UnicodeEncodeError at the worst possible moment, part-way through a long
measurement run.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = sorted((Path(__file__).resolve().parent.parent / "scripts").glob("*.py"))


def test_there_are_scripts_to_check():
    """A glob that silently matches nothing would make every test below vacuously pass."""
    assert len(SCRIPTS) >= 14


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_script_parses(path: Path):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_script_is_ascii(path: Path):
    text = path.read_text(encoding="utf-8")
    offenders = sorted({c for c in text if not c.isascii()})
    assert not offenders, f"{path.name} has non-ASCII characters: {offenders}"


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_script_has_a_docstring_with_an_example(path: Path):
    """Each script's docstring opens with what it measures and how to run it; --help prints it."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    doc = ast.get_docstring(tree)
    assert doc, f"{path.name} has no module docstring"
    assert f"scripts/{path.name}" in doc, f"{path.name} docstring shows no example invocation"


@pytest.mark.slow
@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_script_help_runs(path: Path):
    """The real check: imports resolve and argparse is wired up. Slow only because each one is a
    fresh interpreter importing numpy, and some of them torch."""
    done = subprocess.run([sys.executable, str(path), "--help"], capture_output=True, text=True,
                          timeout=300)
    assert done.returncode == 0, f"{path.name} --help failed:\n{done.stderr[-2000:]}"
    assert "usage:" in done.stdout
