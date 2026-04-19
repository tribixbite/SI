"""Unit tests for proposer/solver output parsers. No GPU needed."""

from si.parsers import (
    contains_banned_import,
    extract_input,
    extract_output,
    extract_program,
)
from si.prompts import BANNED_IMPORTS


def test_extract_program_happy_path():
    text = """
```python
def f(x):
    return x + 1
```
"""
    assert extract_program(text) == "def f(x):\n    return x + 1"


def test_extract_program_missing_def_f():
    text = "```python\ndef g(x):\n    return x\n```"
    assert extract_program(text) is None


def test_extract_program_syntax_error():
    text = "```python\ndef f(x):\n    return (((\n```"
    assert extract_program(text) is None


def test_extract_program_no_fence():
    assert extract_program("def f(x): return x") is None


def test_extract_input_and_output():
    text = """
```input
1, 2, 'hi'
```

```output
42
```
"""
    assert extract_input(text) == "1, 2, 'hi'"
    assert extract_output(text) == "42"


def test_banned_imports_detected():
    prog = "import os\ndef f(x): return x"
    assert contains_banned_import(prog, BANNED_IMPORTS)


def test_banned_imports_from_form():
    prog = "from subprocess import run\ndef f(x): return x"
    assert contains_banned_import(prog, BANNED_IMPORTS)


def test_banned_imports_clean_program():
    prog = "import math\ndef f(x): return math.sqrt(x)"
    assert not contains_banned_import(prog, BANNED_IMPORTS)


def test_banned_imports_syntax_error_treated_as_banned():
    # Fail-closed: unparseable code is rejected, not trusted.
    prog = "import (((\n"
    assert contains_banned_import(prog, BANNED_IMPORTS)
