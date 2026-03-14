"""
Generic .env file parser.

Pure Python stdlib — no external dependencies. Reusable in any Python project.

Why not python-dotenv?
    Zero-dependency constraint. This module replicates the subset of .env
    parsing that matters: comments, blank lines, quoted values, and tilde
    expansion. It intentionally does NOT support variable interpolation
    (${VAR}) or export prefixes to keep behavior predictable.

Parsing Rules
=============

    - Lines starting with # are comments (skipped).
    - Empty/whitespace-only lines are skipped.
    - Lines without = are skipped.
    - Key is everything before the first =, stripped of whitespace.
    - Value is everything after the first =, stripped of whitespace.
    - Matching quotes (single or double) around value are stripped.
    - Leading ~ in value is expanded to the user's home directory.
    - Inline comments are NOT stripped (to match bash source behavior).

Precedence
==========

    load_dotenv_into_environ() does NOT overwrite existing env vars by
    default. This ensures the standard precedence chain:

        CLI args > environment variables > .env file > code defaults

Usage
=====

    from src.process import load_dotenv, load_dotenv_into_environ

    # Parse into a dict (does not touch os.environ)
    values = load_dotenv("path/to/.env")

    # Load into os.environ (won't overwrite existing vars)
    load_dotenv_into_environ("path/to/.env")

    # Load and overwrite existing vars
    load_dotenv_into_environ("path/to/.env", overwrite=True)
"""

import os
from typing import Dict


def load_dotenv(path: str) -> Dict[str, str]:
    """Parse a .env file into a dictionary.

    Handles:
        - Comments (lines starting with #)
        - Blank lines
        - Inline comments after values are NOT stripped (to match bash behavior)
        - Leading/trailing whitespace on keys and values
        - Tilde (~) expansion to home directory
        - Quoted values (single and double quotes stripped)

    Args:
        path: Path to the .env file.

    Returns:
        Dictionary of key-value pairs. Empty dict if file doesn't exist.
    """
    result = {}

    if not os.path.isfile(path):
        return result

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Only process lines with =
            if "=" not in line:
                continue

            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()

            # Skip if key is empty
            if not key:
                continue

            # Strip matching quotes
            if len(value) >= 2:
                if (value[0] == '"' and value[-1] == '"') or \
                   (value[0] == "'" and value[-1] == "'"):
                    value = value[1:-1]

            # Expand ~ to home directory
            if value.startswith("~"):
                value = os.path.expanduser(value)

            result[key] = value

    return result


def load_dotenv_into_environ(path: str, overwrite: bool = False) -> Dict[str, str]:
    """Load .env values into os.environ.

    Args:
        path: Path to the .env file.
        overwrite: If False (default), existing environment variables
            are NOT overwritten. This preserves CLI > env > .env precedence.

    Returns:
        Dictionary of all parsed key-value pairs (regardless of whether
        they were actually set in environ).
    """
    values = load_dotenv(path)

    for key, value in values.items():
        if overwrite or key not in os.environ:
            os.environ[key] = value

    return values
