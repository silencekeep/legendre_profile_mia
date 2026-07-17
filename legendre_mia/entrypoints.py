from __future__ import annotations

import sys
from typing import Sequence

from .cli import main


def run(command: str, argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    return main(arguments, fixed_command=str(command), prog=sys.argv[0])
