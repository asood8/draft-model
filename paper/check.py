"""Structural checks for main.tex, for when there is no LaTeX toolchain to compile with.

    python paper/check.py

This catches the errors that are expensive to find later: an unescaped percent sign silently eats
the rest of its line, an unbalanced environment or brace fails the build with a line number far from
the cause, and a \\cite key absent from refs.bib compiles to a bare question mark. None of this
replaces a compile -- it is what to run when you cannot do one.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def strip_comments(line: str) -> str:
    """Everything from the first unescaped % onward is a comment."""
    out = []
    i = 0
    while i < len(line):
        if line[i] == "\\" and i + 1 < len(line):
            out.append(line[i:i + 2])
            i += 2
            continue
        if line[i] == "%":
            break
        out.append(line[i])
        i += 1
    return "".join(out)


def check(tex: Path, bib: Path) -> list[str]:
    problems: list[str] = []
    raw = tex.read_text(encoding="utf-8").splitlines()
    body = [strip_comments(line) for line in raw]

    # 1. Environments balance, and nest in the right order.
    stack: list[tuple[str, int]] = []
    for n, line in enumerate(body, 1):
        for kind, name in re.findall(r"\\(begin|end)\{([^}]+)\}", line):
            if kind == "begin":
                stack.append((name, n))
            else:
                if not stack:
                    problems.append(f"{tex.name}:{n}: \\end{{{name}}} with nothing open")
                elif stack[-1][0] != name:
                    opened, where = stack[-1]
                    problems.append(
                        f"{tex.name}:{n}: \\end{{{name}}} closes \\begin{{{opened}}} "
                        f"from line {where}")
                    stack.pop()
                else:
                    stack.pop()
    for name, where in stack:
        problems.append(f"{tex.name}:{where}: \\begin{{{name}}} never closed")

    # 2. Braces balance over the whole file.
    depth = 0
    for n, line in enumerate(body, 1):
        for i, ch in enumerate(line):
            if i and line[i - 1] == "\\":
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth < 0:
                    problems.append(f"{tex.name}:{n}: closing brace with nothing open")
                    depth = 0
    if depth:
        problems.append(f"{tex.name}: {depth} brace(s) left open at end of file")

    # 3. Percent signs that were meant to be literal. A digit or a letter in front of a bare % is
    #    almost always a number like "88.7%" that will silently swallow its line.
    for n, line in enumerate(raw, 1):
        stripped = line.lstrip()
        if stripped.startswith("%"):
            continue  # a real comment line
        for m in re.finditer(r"%", line):
            i = m.start()
            if i and line[i - 1] == "\\":
                continue
            before = line[max(0, i - 1):i]
            if before and (before.isalnum() or before in ")]"):
                problems.append(f"{tex.name}:{n}: bare % after {before!r} "
                                f"-- did you mean \\%? ({line.strip()[:70]})")

    # 4. Every \ref has a \label, and every \cite key is in refs.bib.
    text = "\n".join(body)
    labels = set(re.findall(r"\\label\{([^}]+)\}", text))
    for target in set(re.findall(r"\\(?:eq)?ref\{([^}]+)\}", text)):
        if target not in labels:
            problems.append(f"{tex.name}: \\ref{{{target}}} has no \\label")
    unused = labels - set(re.findall(r"\\(?:eq)?ref\{([^}]+)\}", text))
    for label in sorted(unused):
        problems.append(f"{tex.name}: \\label{{{label}}} is never referenced (harmless)")

    if bib.exists():
        keys = set(re.findall(r"@\w+\{([^,]+),", bib.read_text(encoding="utf-8")))
        cited = {k.strip() for group in re.findall(r"\\cite\{([^}]+)\}", text)
                 for k in group.split(",")}
        for key in sorted(cited - keys):
            problems.append(f"{bib.name}: \\cite{{{key}}} is not in the bibliography")
        for key in sorted(keys - cited):
            problems.append(f"{bib.name}: {key} is in the bibliography but never cited (harmless)")
    else:
        problems.append(f"{bib.name}: missing")

    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Structural checks for the paper source.")
    parser.add_argument("--tex", type=Path, default=Path(__file__).parent / "main.tex")
    parser.add_argument("--bib", type=Path, default=Path(__file__).parent / "refs.bib")
    args = parser.parse_args()

    problems = check(args.tex, args.bib)
    pending = len(re.findall(r"\\pending\{", args.tex.read_text(encoding="utf-8"))) - 1  # minus def

    hard = [p for p in problems if "harmless" not in p]
    for p in problems:
        print(("  " if "harmless" in p else "! ") + p)
    print(f"\n{len(hard)} problem(s), {len(problems) - len(hard)} note(s), "
          f"{pending} \\pending marker(s)")
    sys.exit(1 if hard else 0)


if __name__ == "__main__":
    main()
