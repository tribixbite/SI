"""Parsers that extract proposer/solver outputs into Task / Solution dataclasses.

Gemma 4's instruct outputs are fenced markdown. These extractors are
intentionally strict — anything off-format is treated as a failed generation
(its reward is zero, which is fine for the RL loop).
"""

from __future__ import annotations

import ast
import re

_CODE_FENCE = re.compile(r"```(?:python|py)\s*\n(.*?)\n```", re.DOTALL)
_INPUT_FENCE = re.compile(r"```input\s*\n(.*?)\n```", re.DOTALL)
_OUTPUT_FENCE = re.compile(r"```output\s*\n(.*?)\n```", re.DOTALL)


def extract_program(text: str) -> str | None:
    m = _CODE_FENCE.search(text)
    if m is None:
        return None
    program = m.group(1).strip()
    if "def f(" not in program:
        return None
    try:
        ast.parse(program)
    except SyntaxError:
        return None
    return program


def extract_input(text: str) -> str | None:
    m = _INPUT_FENCE.search(text)
    if m is None:
        return None
    body = m.group(1).strip()
    return body or None


def extract_output(text: str) -> str | None:
    m = _OUTPUT_FENCE.search(text)
    if m is None:
        return None
    body = m.group(1).strip()
    return body or None


def contains_banned_import(program: str, banned: tuple[str, ...]) -> bool:
    """Quick AST walk to reject programs importing banned modules.

    Defense-in-depth alongside the sandbox. The sandbox is the real safety net
    (no network, mem/CPU cap); this just saves a wasted verifier call.
    """
    try:
        tree = ast.parse(program)
    except SyntaxError:
        return True
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] in banned for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in banned:
                return True
    return False
