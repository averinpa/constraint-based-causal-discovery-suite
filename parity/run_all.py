"""Run all parity harnesses and report aggregate results.

    uv run python -m parity.run_all

Exits 0 iff every harness reports zero "real disagreements" (cases where
both cbcd and the reference match the d-sep truth but produce different
output endpoint matrices).
"""

from __future__ import annotations

import sys

from parity.fci.run import main as fci_main
from parity.pc.run import main as pc_main
from parity.pcmci.run import main as pcmci_main


def main() -> int:
    rcs = [pc_main(), fci_main(), pcmci_main()]
    print()
    print("=" * 72)
    print(f"AGGREGATE: {sum(rc == 0 for rc in rcs)}/3 harnesses passed")
    print("=" * 72)
    return 0 if all(rc == 0 for rc in rcs) else 1


if __name__ == "__main__":
    sys.exit(main())
