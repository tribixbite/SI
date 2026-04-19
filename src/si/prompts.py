"""Prompt templates for AZR proposer + solver roles.

Condensed, chat-template-ready versions of AZR's prompts
(absolute_zero_reasoner/data_construction/prompts.py). Adapted for Gemma 4's
instruction-tuned chat format — always wrap in llm.chat(...) messages, never
raw completion. See memory: "Gemma 4 E4B inference baseline".

Only deduction + abduction in Phase 1 (induction needs a seed pool of programs;
deferred to later phases).
"""

from __future__ import annotations

BANNED_IMPORTS = (
    "os",
    "sys",
    "subprocess",
    "socket",
    "shutil",
    "pathlib",
    "requests",
    "urllib",
    "ctypes",
    "multiprocessing",
    "threading",
    "pickle",
    "marshal",
    "builtins",
    "__builtin__",
)


_PROPOSER_COMMON = """## Code Requirements
- Define a deterministic function named `f` with at least one parameter.
- `f` must return a value.
- The snippet must execute in under 5 seconds on a modern CPU.
- ALL imports and helper class definitions must be at the top, before `f`.
- You MUST NOT use: {banned}.
- No I/O (file, network), no date/time, no randomness, no printing.
- End with the definition of `f`; nothing after its `return`.

## Output Format
Emit EXACTLY two fenced blocks:

```python
# imports and helpers at top
def f(...):
    ...
    return ...
```

```input
arg1, arg2, ...
```

Put string arguments in quotes. Separate multiple arguments with commas.
No prose after the blocks.
"""


def proposer_deduction_prompt() -> str:
    """Proposer for deduction: generate (program, input). Solver will predict output."""
    return (
        "Write a small Python function and a matching input such that predicting "
        "the output requires non-trivial reasoning. Prefer state tracking, data "
        "structure manipulation, or control flow over arithmetic tricks.\n\n"
        + _PROPOSER_COMMON.format(banned=", ".join(BANNED_IMPORTS))
    )


def proposer_abduction_prompt() -> str:
    """Proposer for abduction: generate (program, input). Solver will predict THE input that produces the given output."""
    return (
        "Write a small Python function and a matching input such that inferring "
        "the input from the function and its output requires non-trivial "
        "reasoning. The function should be as deterministic and injective as "
        "possible near the chosen input so that a correct answer exists.\n\n"
        + _PROPOSER_COMMON.format(banned=", ".join(BANNED_IMPORTS))
    )


def solver_deduction_prompt(program: str, input_literal: str) -> str:
    return (
        "Given this Python function and a call, predict the exact return value.\n"
        "Output ONLY a single fenced block named `output` containing the Python "
        "repr of the return value. No explanation.\n\n"
        f"```python\n{program}\n```\n\n"
        f"Call: `f({input_literal})`\n\n"
        "Format:\n```output\n<repr>\n```"
    )


def solver_abduction_prompt(program: str, output_literal: str) -> str:
    return (
        "Given this Python function, find input arguments such that the call "
        "returns the exact output shown. Output ONLY a single fenced block "
        "named `input` with the arguments to `f`. No explanation.\n\n"
        f"```python\n{program}\n```\n\n"
        f"Target output (repr): `{output_literal}`\n\n"
        "Format:\n```input\narg1, arg2, ...\n```"
    )
